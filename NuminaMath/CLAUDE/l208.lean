import Mathlib

namespace NUMINAMATH_CALUDE_coloring_books_sale_result_l208_20872

/-- The number of coloring books gotten rid of in a store sale -/
def books_gotten_rid_of (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  initial_stock - (shelves * books_per_shelf)

/-- Theorem stating that the number of coloring books gotten rid of is 39 -/
theorem coloring_books_sale_result : 
  books_gotten_rid_of 120 9 9 = 39 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_sale_result_l208_20872


namespace NUMINAMATH_CALUDE_no_natural_solutions_l208_20871

theorem no_natural_solutions : ¬∃ (x y : ℕ), x^4 - 2*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l208_20871


namespace NUMINAMATH_CALUDE_journey_distance_l208_20870

/-- Given a constant speed, if a journey of 120 miles takes 3 hours, 
    then a journey of 5 hours at the same speed covers a distance of 200 miles. -/
theorem journey_distance (speed : ℝ) 
  (h1 : speed * 3 = 120) 
  (h2 : speed > 0) : 
  speed * 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l208_20870


namespace NUMINAMATH_CALUDE_natalia_novels_l208_20869

/-- The number of novels Natalia has in her library -/
def number_of_novels : ℕ := sorry

/-- The number of comics in Natalia's library -/
def comics : ℕ := 271

/-- The number of documentaries in Natalia's library -/
def documentaries : ℕ := 419

/-- The number of albums in Natalia's library -/
def albums : ℕ := 209

/-- The capacity of each crate -/
def crate_capacity : ℕ := 9

/-- The total number of crates used -/
def total_crates : ℕ := 116

theorem natalia_novels :
  number_of_novels = 145 ∧
  comics + documentaries + albums + number_of_novels = crate_capacity * total_crates :=
by sorry

end NUMINAMATH_CALUDE_natalia_novels_l208_20869


namespace NUMINAMATH_CALUDE_other_number_proof_l208_20820

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 6300)
  (h2 : Nat.gcd a b = 15)
  (h3 : a = 210) : 
  b = 450 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l208_20820


namespace NUMINAMATH_CALUDE_sum_of_integers_l208_20848

theorem sum_of_integers (s l : ℤ) : 
  s = 10 → 2 * l = 5 * s - 10 → s + l = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l208_20848


namespace NUMINAMATH_CALUDE_odd_function_extension_l208_20853

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (1 + 3 * x)) :
  ∀ x < 0, f x = x * (1 - 3 * x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l208_20853


namespace NUMINAMATH_CALUDE_cost_per_page_is_five_l208_20810

/-- The cost per page in cents when buying notebooks -/
def cost_per_page (num_notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (num_notebooks * pages_per_notebook)

/-- Theorem: The cost per page is 5 cents when buying 2 notebooks with 50 pages each for $5 -/
theorem cost_per_page_is_five :
  cost_per_page 2 50 5 = 5 := by
  sorry

#eval cost_per_page 2 50 5

end NUMINAMATH_CALUDE_cost_per_page_is_five_l208_20810


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l208_20842

-- Define the parabola and hyperbola
def parabola (x y : ℝ) : Prop := y^2 = -4*x
def hyperbola (x y : ℝ) : Prop := x^2/(1/4) - y^2/(3/4) = 1

-- Define the conditions
def parabola_vertex_origin (p : ℝ → ℝ → Prop) : Prop := p 0 0
def parabola_axis_perpendicular_x (p : ℝ → ℝ → Prop) : Prop := ∀ y, p 1 y
def parabola_hyperbola_intersection (p h : ℝ → ℝ → Prop) : Prop := p (-3/2) (Real.sqrt 6) ∧ h (-3/2) (Real.sqrt 6)
def hyperbola_focus (h : ℝ → ℝ → Prop) : Prop := h 1 0

-- Main theorem
theorem parabola_hyperbola_equations :
  ∀ p h : ℝ → ℝ → Prop,
  parabola_vertex_origin p →
  parabola_axis_perpendicular_x p →
  parabola_hyperbola_intersection p h →
  hyperbola_focus h →
  (∀ x y, p x y ↔ parabola x y) ∧
  (∀ x y, h x y ↔ hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l208_20842


namespace NUMINAMATH_CALUDE_coach_rental_equation_l208_20884

/-- Represents the equation for renting coaches to transport a group of people -/
theorem coach_rental_equation (total_people : ℕ) (school_bus_capacity : ℕ) (coach_capacity : ℕ) (x : ℕ) :
  total_people = 328 →
  school_bus_capacity = 64 →
  coach_capacity = 44 →
  44 * x + 64 = 328 :=
by sorry

end NUMINAMATH_CALUDE_coach_rental_equation_l208_20884


namespace NUMINAMATH_CALUDE_symmetry_and_inverse_l208_20868

/-- A function that is symmetric about the line y = x + 1 -/
def SymmetricAboutXPlus1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (y - 1) = x + 1

/-- Definition of g in terms of f and b -/
def g (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x ↦ f (x + b)

/-- A function that is identical to its inverse -/
def IdenticalToInverse (h : ℝ → ℝ) : Prop :=
  ∀ x, h (h x) = x

theorem symmetry_and_inverse (f : ℝ → ℝ) (b : ℝ) 
  (h_sym : SymmetricAboutXPlus1 f) :
  IdenticalToInverse (g f b) ↔ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_and_inverse_l208_20868


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l208_20821

theorem power_equality_implies_exponent (a : ℝ) (m : ℕ) (h : (a^2)^m = a^6) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l208_20821


namespace NUMINAMATH_CALUDE_truck_capacity_l208_20880

theorem truck_capacity (large small : ℝ) 
  (h1 : 2 * large + 3 * small = 15.5)
  (h2 : 5 * large + 6 * small = 35) :
  3 * large + 2 * small = 17 := by
sorry

end NUMINAMATH_CALUDE_truck_capacity_l208_20880


namespace NUMINAMATH_CALUDE_grinder_purchase_price_l208_20819

theorem grinder_purchase_price
  (mobile_cost : ℝ)
  (grinder_loss_percent : ℝ)
  (mobile_profit_percent : ℝ)
  (total_profit : ℝ)
  (h1 : mobile_cost = 8000)
  (h2 : grinder_loss_percent = 0.05)
  (h3 : mobile_profit_percent = 0.10)
  (h4 : total_profit = 50) :
  ∃ (grinder_cost : ℝ),
    grinder_cost * (1 - grinder_loss_percent) +
    mobile_cost * (1 + mobile_profit_percent) -
    (grinder_cost + mobile_cost) = total_profit ∧
    grinder_cost = 15000 := by
  sorry

end NUMINAMATH_CALUDE_grinder_purchase_price_l208_20819


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l208_20832

theorem cos_alpha_plus_5pi_12 (α : ℝ) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l208_20832


namespace NUMINAMATH_CALUDE_complex_equal_parts_l208_20837

/-- Given a complex number z = (2+ai)/(1+2i) where a is real,
    if the real part of z equals its imaginary part, then a = -6 -/
theorem complex_equal_parts (a : ℝ) :
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  Complex.re z = Complex.im z → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l208_20837


namespace NUMINAMATH_CALUDE_subtraction_digit_sum_l208_20874

theorem subtraction_digit_sum : ∃ (a b : ℕ), 
  (a < 10) ∧ (b < 10) ∧ 
  (a * 10 + 9) - (1800 + b * 10 + 8) = 1 ∧
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_subtraction_digit_sum_l208_20874


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l208_20802

/-- Given a rectangular plot with the following properties:
  - The length is 32 meters more than the breadth
  - The cost of fencing at 26.50 per meter is Rs. 5300
  Prove that the length of the plot is 66 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = breadth + 32 →
  perimeter = 2 * (length + breadth) →
  perimeter * 26.5 = 5300 →
  length = 66 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_plot_length_l208_20802


namespace NUMINAMATH_CALUDE_trip_expenses_l208_20875

def david_initial : ℝ := 1800
def emma_initial : ℝ := 2400
def john_initial : ℝ := 1200

def david_spend_percent : ℝ := 0.60
def emma_spend_percent : ℝ := 0.75
def john_spend_percent : ℝ := 0.50

def david_remaining : ℝ := david_initial * (1 - david_spend_percent)
def emma_spent : ℝ := emma_initial * emma_spend_percent
def emma_remaining : ℝ := emma_spent - 800
def john_remaining : ℝ := john_initial * (1 - john_spend_percent)

theorem trip_expenses :
  david_remaining = 720 ∧
  emma_remaining = 1400 ∧
  john_remaining = 600 ∧
  emma_remaining = emma_spent - 800 :=
by sorry

end NUMINAMATH_CALUDE_trip_expenses_l208_20875


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l208_20800

/-- The slopes of the asymptotes of a hyperbola -/
def asymptote_slopes (a b : ℝ) : Set ℝ :=
  {m : ℝ | m = b / a ∨ m = -b / a}

/-- Theorem: The slopes of the asymptotes of the hyperbola (x^2/16) - (y^2/25) = 1 are ±5/4 -/
theorem hyperbola_asymptote_slopes :
  asymptote_slopes 4 5 = {5/4, -5/4} := by
  sorry

#check hyperbola_asymptote_slopes

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l208_20800


namespace NUMINAMATH_CALUDE_courtyard_length_l208_20882

theorem courtyard_length (stone_length : ℝ) (stone_width : ℝ) (courtyard_width : ℝ) (total_stones : ℕ) :
  stone_length = 2.5 →
  stone_width = 2 →
  courtyard_width = 16.5 →
  total_stones = 198 →
  ∃ courtyard_length : ℝ, courtyard_length = 60 ∧ 
    courtyard_length * courtyard_width = (stone_length * stone_width) * total_stones :=
by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l208_20882


namespace NUMINAMATH_CALUDE_f_equals_g_l208_20881

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l208_20881


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l208_20846

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A function to check if a number is five-digit -/
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function to check if a number is divisible by all numbers in a list -/
def divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ m ∈ list, n % m = 0

theorem smallest_five_digit_divisible_by_smallest_primes :
  (is_five_digit 11550) ∧ 
  (divisible_by_all 11550 smallest_primes) ∧ 
  (∀ n : Nat, is_five_digit n ∧ divisible_by_all n smallest_primes → 11550 ≤ n) := by
  sorry

#check smallest_five_digit_divisible_by_smallest_primes

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l208_20846


namespace NUMINAMATH_CALUDE_intersection_sum_l208_20887

theorem intersection_sum (a b : ℚ) : 
  (3 = (1/3) * 4 + a) → 
  (4 = (1/3) * 3 + b) → 
  a + b = 14/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l208_20887


namespace NUMINAMATH_CALUDE_no_double_by_digit_move_l208_20808

theorem no_double_by_digit_move :
  ¬ ∃ (x : ℕ) (n : ℕ), n ≥ 1 ∧
    (∃ (a : ℕ) (N : ℕ),
      x = a * 10^n + N ∧
      0 < a ∧ a < 10 ∧
      N < 10^n ∧
      10 * N + a = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_no_double_by_digit_move_l208_20808


namespace NUMINAMATH_CALUDE_six_people_arrangement_l208_20836

theorem six_people_arrangement (n : ℕ) (h : n = 6) : 
  (2 : ℕ) * (2 : ℕ) * (Nat.factorial 4) = 96 :=
sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l208_20836


namespace NUMINAMATH_CALUDE_five_student_committees_l208_20811

theorem five_student_committees (n k : ℕ) (h1 : n = 8) (h2 : k = 5) : 
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_l208_20811


namespace NUMINAMATH_CALUDE_winter_olympics_volunteers_l208_20895

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  (k - 1).choose (n - 1) * k.factorial

/-- The problem statement -/
theorem winter_olympics_volunteers : distribute 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_winter_olympics_volunteers_l208_20895


namespace NUMINAMATH_CALUDE_polygon_angle_ratio_l208_20856

theorem polygon_angle_ratio (n : ℕ) : 
  (((n - 2) * 180) / 360 : ℚ) = 9/2 ↔ n = 11 := by sorry

end NUMINAMATH_CALUDE_polygon_angle_ratio_l208_20856


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l208_20834

theorem ellipse_hyperbola_product (A B : ℝ) 
  (h1 : B^2 - A^2 = 25)
  (h2 : A^2 + B^2 = 64) :
  |A * B| = Real.sqrt 867.75 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l208_20834


namespace NUMINAMATH_CALUDE_angle_A_is_30_degrees_max_area_is_3_l208_20859

namespace TriangleProof

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Theorem 1: Angle A is 30° when a = 5/3
theorem angle_A_is_30_degrees (t : Triangle) 
  (h : triangle_conditions t) (ha : t.a = 5/3) : 
  t.A = Real.pi / 6 := by sorry

-- Theorem 2: Maximum area is 3
theorem max_area_is_3 (t : Triangle) 
  (h : triangle_conditions t) : 
  (∃ (max_area : ℝ), ∀ (s : Triangle), 
    triangle_conditions s → 
    (1/2 * s.a * s.c * Real.sin s.B) ≤ max_area ∧ 
    max_area = 3) := by sorry

end TriangleProof

end NUMINAMATH_CALUDE_angle_A_is_30_degrees_max_area_is_3_l208_20859


namespace NUMINAMATH_CALUDE_smaller_factor_of_4851_l208_20807

theorem smaller_factor_of_4851 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4851 → 
  min a b = 53 := by
sorry

end NUMINAMATH_CALUDE_smaller_factor_of_4851_l208_20807


namespace NUMINAMATH_CALUDE_smaller_square_area_equals_larger_l208_20825

/-- A square inscribed in a circle with another smaller square -/
structure SquaresInCircle where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square -/
  s : ℝ
  /-- Side length of the smaller square -/
  x : ℝ
  /-- The larger square is inscribed in the circle -/
  h1 : s = 2 * r
  /-- The smaller square has one side coinciding with the larger square -/
  h2 : x ≤ s
  /-- Two vertices of the smaller square are on the circle -/
  h3 : x^2 + (r + x/2)^2 = r^2

/-- The area of the smaller square is equal to the area of the larger square -/
theorem smaller_square_area_equals_larger (sq : SquaresInCircle) :
  sq.x^2 = sq.s^2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_equals_larger_l208_20825


namespace NUMINAMATH_CALUDE_trapezoid_angle_sequence_l208_20843

theorem trapezoid_angle_sequence (a d : ℝ) : 
  (a > 0) →
  (d > 0) →
  (a + 2*d = 105) →
  (4*a + 6*d = 360) →
  (a + d = 85) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_angle_sequence_l208_20843


namespace NUMINAMATH_CALUDE_spongebob_fry_price_l208_20857

/-- Calculates the price of each large fry given the number of burgers sold, 
    price per burger, number of large fries sold, and total earnings. -/
def price_of_large_fry (num_burgers : ℕ) (price_per_burger : ℚ) 
                       (num_fries : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - num_burgers * price_per_burger) / num_fries

/-- Theorem stating that the price of each large fry is $1.50 
    given Spongebob's sales information. -/
theorem spongebob_fry_price : 
  price_of_large_fry 30 2 12 78 = (3/2) := by
  sorry

end NUMINAMATH_CALUDE_spongebob_fry_price_l208_20857


namespace NUMINAMATH_CALUDE_rind_papyrus_fraction_decomposition_l208_20803

theorem rind_papyrus_fraction_decomposition : 
  (2 : ℚ) / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / 365 := by
  sorry

end NUMINAMATH_CALUDE_rind_papyrus_fraction_decomposition_l208_20803


namespace NUMINAMATH_CALUDE_music_books_cost_l208_20855

/-- Calculates the amount spent on music books including tax --/
def amount_spent_on_music_books (total_budget : ℝ) (math_book_price : ℝ) (math_book_count : ℕ) 
  (math_book_discount : ℝ) (science_book_price : ℝ) (art_book_price : ℝ) (art_book_tax : ℝ) 
  (music_book_tax : ℝ) : ℝ :=
  let math_books_cost := math_book_count * math_book_price * (1 - math_book_discount)
  let science_books_cost := (math_book_count + 6) * science_book_price
  let art_books_cost := 2 * math_book_count * art_book_price * (1 + art_book_tax)
  let remaining_budget := total_budget - (math_books_cost + science_books_cost + art_books_cost)
  remaining_budget

/-- Theorem stating that the amount spent on music books including tax is $160 --/
theorem music_books_cost (total_budget : ℝ) (math_book_price : ℝ) (math_book_count : ℕ) 
  (math_book_discount : ℝ) (science_book_price : ℝ) (art_book_price : ℝ) (art_book_tax : ℝ) 
  (music_book_tax : ℝ) :
  total_budget = 500 ∧ 
  math_book_price = 20 ∧ 
  math_book_count = 4 ∧ 
  math_book_discount = 0.1 ∧ 
  science_book_price = 10 ∧ 
  art_book_price = 20 ∧ 
  art_book_tax = 0.05 ∧ 
  music_book_tax = 0.07 → 
  amount_spent_on_music_books total_budget math_book_price math_book_count math_book_discount 
    science_book_price art_book_price art_book_tax music_book_tax = 160 := by
  sorry


end NUMINAMATH_CALUDE_music_books_cost_l208_20855


namespace NUMINAMATH_CALUDE_junior_toy_ratio_l208_20828

theorem junior_toy_ratio :
  let num_rabbits : ℕ := 16
  let monday_toys : ℕ := 6
  let friday_toys : ℕ := 4 * monday_toys
  let wednesday_toys : ℕ := wednesday_toys -- Unknown variable
  let saturday_toys : ℕ := wednesday_toys / 2
  let toys_per_rabbit : ℕ := 3
  
  num_rabbits * toys_per_rabbit = monday_toys + wednesday_toys + friday_toys + saturday_toys →
  wednesday_toys = 2 * monday_toys :=
by sorry

end NUMINAMATH_CALUDE_junior_toy_ratio_l208_20828


namespace NUMINAMATH_CALUDE_rotation_theorem_l208_20818

/-- Triangle in 2D plane -/
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

/-- Rotation of a point around another point by 120° clockwise -/
def rotate120 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Generate the sequence of A points -/
def A (n : ℕ) (t : Triangle) : ℝ × ℝ :=
  match n % 3 with
  | 0 => t.A₃
  | 1 => t.A₁
  | _ => t.A₂

/-- Generate the sequence of P points -/
def P (n : ℕ) (t : Triangle) (P₀ : ℝ × ℝ) : ℝ × ℝ :=
  match n with
  | 0 => P₀
  | n + 1 => rotate120 (A (n + 1) t) (P n t P₀)

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

theorem rotation_theorem (t : Triangle) (P₀ : ℝ × ℝ) :
  P 1986 t P₀ = P₀ → isEquilateral t := by sorry

end NUMINAMATH_CALUDE_rotation_theorem_l208_20818


namespace NUMINAMATH_CALUDE_rectangle_center_line_slope_l208_20892

/-- The slope of a line passing through the origin and the center of a rectangle with given vertices is 1/5 -/
theorem rectangle_center_line_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (9, 0), (1, 2), (9, 2)]
  let center_x : ℝ := (vertices.map Prod.fst).sum / vertices.length
  let center_y : ℝ := (vertices.map Prod.snd).sum / vertices.length
  let slope : ℝ := center_y / center_x
  slope = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_center_line_slope_l208_20892


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_l208_20858

theorem binomial_coefficient_equation (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 8 (x - 2) + Nat.choose 8 (x - 1) + Nat.choose 9 (2 * x - 3)) → 
  (x = 3 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_l208_20858


namespace NUMINAMATH_CALUDE_pyramid_stack_balls_l208_20873

/-- Represents a pyramid-shaped stack of balls -/
structure PyramidStack where
  top_layer : Nat
  layer_diff : Nat
  bottom_layer : Nat

/-- Calculates the number of layers in the pyramid stack -/
def num_layers (p : PyramidStack) : Nat :=
  (p.bottom_layer - p.top_layer) / p.layer_diff + 1

/-- Calculates the total number of balls in the pyramid stack -/
def total_balls (p : PyramidStack) : Nat :=
  let n := num_layers p
  n * (p.top_layer + p.bottom_layer) / 2

/-- Theorem: The total number of balls in the given pyramid stack is 247 -/
theorem pyramid_stack_balls :
  let p := PyramidStack.mk 1 3 37
  total_balls p = 247 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_stack_balls_l208_20873


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l208_20890

theorem max_sum_of_factors (diamond heart : ℕ) : 
  diamond * heart = 48 → (∀ x y : ℕ, x * y = 48 → x + y ≤ diamond + heart) → diamond + heart = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l208_20890


namespace NUMINAMATH_CALUDE_units_digit_of_fib_F_15_l208_20812

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State that the units digit of Fibonacci numbers repeats every 60 terms
axiom fib_units_period (n : ℕ) : fib n % 10 = fib (n % 60) % 10

-- Define F_15
def F_15 : ℕ := fib 15

-- Theorem to prove
theorem units_digit_of_fib_F_15 : fib (fib 15) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fib_F_15_l208_20812


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l208_20809

theorem absolute_value_inequality (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l208_20809


namespace NUMINAMATH_CALUDE_trig_identity_l208_20867

theorem trig_identity (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : 2 * (Real.sin α)^2 - Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = 0) :
  Real.sin (α + π / 4) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = Real.sqrt 26 / 8 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l208_20867


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l208_20839

/-- A rectangular prism with side lengths a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral formed by the intersection of a plane with a rectangular prism -/
structure IntersectionQuadrilateral where
  prism : RectangularPrism
  -- Assume A and C are diagonally opposite vertices
  -- Assume B and D are midpoints of opposite edges not containing A or C

/-- The area of the quadrilateral formed by the intersection -/
noncomputable def intersection_area (quad : IntersectionQuadrilateral) : ℝ := sorry

/-- Theorem stating the area of the specific intersection quadrilateral -/
theorem intersection_area_theorem (quad : IntersectionQuadrilateral) 
  (h1 : quad.prism.a = 2)
  (h2 : quad.prism.b = 3)
  (h3 : quad.prism.c = 5) :
  intersection_area quad = 7 * Real.sqrt 26 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l208_20839


namespace NUMINAMATH_CALUDE_alien_legs_count_l208_20877

/-- Represents the number of limbs for an alien or martian -/
structure Limbs where
  arms : ℕ
  legs : ℕ

/-- Defines the properties of alien limbs -/
def alien_limbs (l : ℕ) : Limbs :=
  { arms := 3, legs := l }

/-- Defines the properties of martian limbs based on alien legs -/
def martian_limbs (l : ℕ) : Limbs :=
  { arms := 2 * 3, legs := l / 2 }

/-- Theorem stating that aliens have 8 legs -/
theorem alien_legs_count : 
  ∃ l : ℕ, 
    (alien_limbs l).legs = 8 ∧ 
    5 * ((alien_limbs l).arms + (alien_limbs l).legs) = 
    5 * ((martian_limbs l).arms + (martian_limbs l).legs) + 5 :=
by
  sorry


end NUMINAMATH_CALUDE_alien_legs_count_l208_20877


namespace NUMINAMATH_CALUDE_both_runners_in_photo_probability_l208_20876

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photography setup -/
structure Photo where
  trackFraction : ℚ
  timeRange : Set ℕ

/-- Calculates the probability of both runners being in the photo -/
def probabilityBothInPhoto (r1 r2 : Runner) (p : Photo) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem both_runners_in_photo_probability
  (rachel : Runner)
  (robert : Runner)
  (photo : Photo)
  (h1 : rachel.name = "Rachel" ∧ rachel.lapTime = 75 ∧ rachel.direction = true)
  (h2 : robert.name = "Robert" ∧ robert.lapTime = 95 ∧ robert.direction = false)
  (h3 : photo.trackFraction = 1/5)
  (h4 : photo.timeRange = {t | 900 ≤ t ∧ t < 960}) :
  probabilityBothInPhoto rachel robert photo = 1/4 :=
sorry

end NUMINAMATH_CALUDE_both_runners_in_photo_probability_l208_20876


namespace NUMINAMATH_CALUDE_ellipse_properties_l208_20888

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the left focus
def LeftFocus : ℝ × ℝ := (-1, 0)

-- Define the line l
def Line (x y : ℝ) : Prop :=
  y = x + 1

-- Theorem statement
theorem ellipse_properties :
  -- Given conditions
  let C := Ellipse
  let e : ℝ := 1/2
  let max_distance : ℝ := 3
  let l := Line

  -- Prove
  (∀ x y, C x y → x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ A B : ℝ × ℝ,
    C A.1 A.2 ∧ C B.1 B.2 ∧
    l A.1 A.2 ∧ l B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 24/7) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l208_20888


namespace NUMINAMATH_CALUDE_quadratic_properties_l208_20806

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - 4

-- Theorem stating the properties of the function
theorem quadratic_properties :
  ∃ a : ℝ, 
    (f a 0 = -3) ∧ 
    (∀ x, f 1 x = (x - 1)^2 - 4) ∧
    (∀ x > 1, ∀ y > x, f 1 y > f 1 x) ∧
    (f 1 (-1) = 0 ∧ f 1 3 = 0) ∧
    (∀ x, f 1 x = 0 → x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l208_20806


namespace NUMINAMATH_CALUDE_square_area_ratio_l208_20805

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 12.5) (h2 : side_D = 18.5) :
  (side_C^2) / (side_D^2) = 625 / 1369 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l208_20805


namespace NUMINAMATH_CALUDE_coin_flip_probability_l208_20823

theorem coin_flip_probability (n : ℕ) : 
  (n.choose 2 : ℚ) / 2^n = 1/32 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l208_20823


namespace NUMINAMATH_CALUDE_unique_division_solution_l208_20878

theorem unique_division_solution :
  ∀ (dividend divisor quotient : ℕ),
    divisor ≥ 100 ∧ divisor < 1000 →
    quotient ≥ 10000 ∧ quotient < 100000 →
    (quotient / 1000) % 10 = 7 →
    dividend = divisor * quotient →
    (dividend, divisor, quotient) = (12128316, 124, 97809) := by
  sorry

end NUMINAMATH_CALUDE_unique_division_solution_l208_20878


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l208_20894

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 16) / 4 ∨ x = (-8 - Real.sqrt 16) / 4) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l208_20894


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l208_20824

/-- The number of unique arrangements of letters in "BALLOON" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of "BALLOON" is 1260 -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l208_20824


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l208_20814

/-- Represents the number of wrapping paper varieties -/
def wrapping_paper : Nat := 10

/-- Represents the number of ribbon colors -/
def ribbons : Nat := 3

/-- Represents the number of gift card types -/
def gift_cards : Nat := 4

/-- Represents the number of gift tag types -/
def gift_tags : Nat := 5

/-- Calculates the total number of gift wrapping combinations -/
def total_combinations : Nat := wrapping_paper * ribbons * gift_cards * gift_tags

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem gift_wrapping_combinations :
  total_combinations = 600 := by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l208_20814


namespace NUMINAMATH_CALUDE_smallest_x_and_corresponding_yzw_l208_20898

theorem smallest_x_and_corresponding_yzw :
  ∀ (x y z w : ℝ),
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0 →
  y = x - 2003 →
  z = 2*y - 2003 →
  w = 3*z - 2003 →
  (x ≥ 10015/3 ∧ 
   (x = 10015/3 → y = 4006/3 ∧ z = 2003/3 ∧ w = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_and_corresponding_yzw_l208_20898


namespace NUMINAMATH_CALUDE_unique_solution_l208_20830

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  lg x + lg (x - 2) = lg 3 + lg (x + 2)

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x > 2 ∧ equation x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l208_20830


namespace NUMINAMATH_CALUDE_sum_at_two_and_minus_two_l208_20835

def cubic_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d

theorem sum_at_two_and_minus_two
  (P : ℝ → ℝ)
  (k : ℝ)
  (h_cubic : cubic_polynomial P)
  (h_zero : P 0 = k)
  (h_one : P 1 = 3 * k)
  (h_neg_one : P (-1) = 4 * k) :
  P 2 + P (-2) = 22 * k :=
sorry

end NUMINAMATH_CALUDE_sum_at_two_and_minus_two_l208_20835


namespace NUMINAMATH_CALUDE_speed_limit_exceeders_l208_20822

/-- Represents the percentage of motorists who receive speeding tickets -/
def speeding_ticket_percentage : ℝ := 10

/-- Represents the percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 30

/-- Represents the total percentage of motorists exceeding the speed limit -/
def exceeding_speed_limit_percentage : ℝ := 14

theorem speed_limit_exceeders (total_motorists : ℝ) (total_motorists_pos : total_motorists > 0) :
  (speeding_ticket_percentage / 100) * total_motorists =
  ((100 - no_ticket_percentage) / 100) * (exceeding_speed_limit_percentage / 100) * total_motorists :=
by sorry

end NUMINAMATH_CALUDE_speed_limit_exceeders_l208_20822


namespace NUMINAMATH_CALUDE_board_numbers_count_l208_20866

theorem board_numbers_count (M : ℝ) : ∃! k : ℕ,
  k > 0 ∧
  (∃ S : ℝ, M = S / k) ∧
  (S + 15) / (k + 1) = M + 2 ∧
  (S + 16) / (k + 2) = M + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_board_numbers_count_l208_20866


namespace NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l208_20801

theorem tan_eleven_pi_sixths : Real.tan (11 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l208_20801


namespace NUMINAMATH_CALUDE_paths_in_4x4_grid_l208_20852

/-- Number of paths in a grid -/
def num_paths (m n : ℕ) : ℕ :=
  if m = 0 ∨ n = 0 then 1
  else num_paths (m - 1) n + num_paths m (n - 1)

/-- Theorem: The number of paths from (0,0) to (3,3) in a 4x4 grid is 23 -/
theorem paths_in_4x4_grid : num_paths 3 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_4x4_grid_l208_20852


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l208_20816

theorem roots_sum_of_squares (p q r s : ℝ) : 
  (r^2 - 3*p*r + 2*q = 0) → 
  (s^2 - 3*p*s + 2*q = 0) → 
  (r^2 + s^2 = 9*p^2 - 4*q) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l208_20816


namespace NUMINAMATH_CALUDE_teresas_age_at_birth_l208_20845

/-- Given the current ages of Teresa and Morio, and Morio's age when their daughter Michiko was born,
    prove that Teresa's age when Michiko was born is 26. -/
theorem teresas_age_at_birth (teresa_current_age morio_current_age morio_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59)
  (h2 : morio_current_age = 71)
  (h3 : morio_age_at_birth = 38) :
  teresa_current_age - (morio_current_age - morio_age_at_birth) = 26 := by
  sorry

end NUMINAMATH_CALUDE_teresas_age_at_birth_l208_20845


namespace NUMINAMATH_CALUDE_oil_drop_probability_l208_20885

/-- The probability of an oil drop falling into a square hole in a circular coin -/
theorem oil_drop_probability (coin_diameter : Real) (hole_side : Real) : 
  coin_diameter = 2 → hole_side = 0.5 → 
  (hole_side^2) / (π * (coin_diameter/2)^2) = 1 / (4 * π) := by
sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l208_20885


namespace NUMINAMATH_CALUDE_count_solutions_3x_plus_5y_equals_501_l208_20849

theorem count_solutions_3x_plus_5y_equals_501 :
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 5 * p.2 = 501 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 168) (Finset.range 101))).card = 34 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_3x_plus_5y_equals_501_l208_20849


namespace NUMINAMATH_CALUDE_max_modulus_complex_l208_20862

theorem max_modulus_complex (z : ℂ) : 
  ∀ z, Complex.abs (z + z⁻¹) = 1 → Complex.abs z ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_modulus_complex_l208_20862


namespace NUMINAMATH_CALUDE_amount_per_friend_is_correct_l208_20831

/-- The amount each friend pays when 6 friends split a $400 bill equally after applying a 5% discount -/
def amount_per_friend : ℚ :=
  let total_bill : ℚ := 400
  let discount_rate : ℚ := 5 / 100
  let num_friends : ℕ := 6
  let discounted_bill : ℚ := total_bill * (1 - discount_rate)
  discounted_bill / num_friends

/-- Theorem stating that the amount each friend pays is $63.33 (repeating) -/
theorem amount_per_friend_is_correct :
  amount_per_friend = 190 / 3 := by
  sorry

#eval amount_per_friend

end NUMINAMATH_CALUDE_amount_per_friend_is_correct_l208_20831


namespace NUMINAMATH_CALUDE_gathering_attendees_l208_20891

theorem gathering_attendees (n : ℕ) : 
  (n * (n - 1) / 2 = 55) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_gathering_attendees_l208_20891


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l208_20897

/-- The equation of the boundary curve -/
def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * |x - y| + 4 * |x + y|

/-- The region bounded by the curve -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | boundary_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area : ℝ := sorry

theorem area_of_bounded_region :
  area = 64 + 32 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l208_20897


namespace NUMINAMATH_CALUDE_second_monday_watching_time_l208_20815

def day_hours : ℕ := 24

def monday_hours : ℕ := day_hours / 2
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := day_hours / 4
def thursday_hours : ℕ := day_hours / 3
def friday_hours : ℕ := 2 * wednesday_hours
def saturday_hours : ℕ := 0

def week_total : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours + saturday_hours

def sunday_hours : ℕ := week_total / 2

def total_watched : ℕ := week_total + sunday_hours

def show_length : ℕ := 75

theorem second_monday_watching_time :
  show_length - total_watched = 12 := by sorry

end NUMINAMATH_CALUDE_second_monday_watching_time_l208_20815


namespace NUMINAMATH_CALUDE_lemonade_mixture_l208_20889

theorem lemonade_mixture (L : ℝ) : 
  -- First solution composition
  let first_lemonade : ℝ := 20
  let first_carbonated : ℝ := 80
  -- Second solution composition
  let second_lemonade : ℝ := L
  let second_carbonated : ℝ := 55
  -- Mixture composition
  let mixture_carbonated : ℝ := 60
  let mixture_first_solution : ℝ := 20
  -- Conditions
  first_lemonade + first_carbonated = 100 →
  second_lemonade + second_carbonated = 100 →
  mixture_first_solution * first_carbonated / 100 + 
    (100 - mixture_first_solution) * second_carbonated / 100 = mixture_carbonated →
  -- Conclusion
  L = 45 := by
sorry

end NUMINAMATH_CALUDE_lemonade_mixture_l208_20889


namespace NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l208_20841

theorem quadratic_roots_always_positive_implies_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, p > 0 →
    ∀ x : ℝ, a * x^2 + b * x + c + p = 0 →
      x > 0 ∧ (∃ y : ℝ, y ≠ x ∧ a * y^2 + b * y + c + p = 0 ∧ y > 0)) :
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l208_20841


namespace NUMINAMATH_CALUDE_congruence_conditions_and_smallest_n_l208_20879

theorem congruence_conditions_and_smallest_n :
  ∀ (r s : ℕ+),
  (2^(r : ℕ) - 16^(s : ℕ)) % 7 = 5 →
  (r : ℕ) % 3 = 1 ∧ (s : ℕ) % 3 = 2 ∧
  (∀ (r' s' : ℕ+),
    (2^(r' : ℕ) - 16^(s' : ℕ)) % 7 = 5 →
    2^(r : ℕ) - 16^(s : ℕ) ≤ 2^(r' : ℕ) - 16^(s' : ℕ)) ∧
  2^(r : ℕ) - 16^(s : ℕ) = 768 :=
by sorry

end NUMINAMATH_CALUDE_congruence_conditions_and_smallest_n_l208_20879


namespace NUMINAMATH_CALUDE_savings_comparison_l208_20840

theorem savings_comparison (last_year_salary : ℝ) 
  (last_year_savings_rate : ℝ) (salary_increase : ℝ) (this_year_savings_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  salary_increase = 0.20 →
  this_year_savings_rate = 0.05 →
  (this_year_savings_rate * (1 + salary_increase) * last_year_salary) / 
  (last_year_savings_rate * last_year_salary) = 1 := by
  sorry

#check savings_comparison

end NUMINAMATH_CALUDE_savings_comparison_l208_20840


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l208_20854

def angle_equation (x : ℝ) : Prop :=
  12 * (Real.sin x)^3 * (Real.cos x)^2 - 12 * (Real.sin x)^2 * (Real.cos x)^3 = 3/2

theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < π/2 ∧ angle_equation x ∧
  ∀ (y : ℝ), y > 0 ∧ y < x → ¬(angle_equation y) ∧
  x = 7.5 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l208_20854


namespace NUMINAMATH_CALUDE_problem_grid_square_count_l208_20851

/-- Represents the grid structure in the problem -/
structure GridStructure where
  width : Nat
  height : Nat
  largeSquares : Nat
  topRowExtraSquares : Nat
  bottomRowExtraSquares : Nat

/-- Counts the number of squares of a given size in the grid -/
def countSquares (g : GridStructure) (size : Nat) : Nat :=
  match size with
  | 1 => g.largeSquares * 6 + g.topRowExtraSquares + g.bottomRowExtraSquares
  | 2 => g.largeSquares * 4 + 4
  | 3 => g.largeSquares * 2 + 1
  | _ => 0

/-- The total number of squares in the grid -/
def totalSquares (g : GridStructure) : Nat :=
  (countSquares g 1) + (countSquares g 2) + (countSquares g 3)

/-- The specific grid structure from the problem -/
def problemGrid : GridStructure := {
  width := 5
  height := 5
  largeSquares := 2
  topRowExtraSquares := 5
  bottomRowExtraSquares := 4
}

theorem problem_grid_square_count :
  totalSquares problemGrid = 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_grid_square_count_l208_20851


namespace NUMINAMATH_CALUDE_max_value_quadratic_l208_20860

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  x^2 + 2*x*y + 3*y^2 ≤ 24 + 12*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l208_20860


namespace NUMINAMATH_CALUDE_triangle_inequality_l208_20850

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  -- Add triangle inequality conditions
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.area ∧
  (t.a^2 + t.b^2 + t.c^2 = 4 * Real.sqrt 3 * t.area ↔ isEquilateral t) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l208_20850


namespace NUMINAMATH_CALUDE_existence_condition_range_l208_20847

theorem existence_condition_range (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, -x₀^2 + 3*x₀ + a > 0) ↔ a > -2 :=
sorry

end NUMINAMATH_CALUDE_existence_condition_range_l208_20847


namespace NUMINAMATH_CALUDE_betty_orange_boxes_l208_20813

theorem betty_orange_boxes (oranges_per_box : ℕ) (total_oranges : ℕ) (h1 : oranges_per_box = 24) (h2 : total_oranges = 72) :
  total_oranges / oranges_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_betty_orange_boxes_l208_20813


namespace NUMINAMATH_CALUDE_not_right_triangle_l208_20899

theorem not_right_triangle : ∀ a b c : ℕ,
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ 
  (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ 
  (a = 7 ∧ b = 8 ∧ c = 13) →
  (a^2 + b^2 ≠ c^2) ↔ (a = 7 ∧ b = 8 ∧ c = 13) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l208_20899


namespace NUMINAMATH_CALUDE_modulus_of_z_values_of_a_and_b_l208_20893

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)

-- Theorem for the modulus of z
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

-- Theorem for the values of a and b
theorem values_of_a_and_b :
  ∀ (a b : ℝ), z^2 + a*z + b = 1 + i → a = -3 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_values_of_a_and_b_l208_20893


namespace NUMINAMATH_CALUDE_M_inter_N_eq_M_l208_20826

open Set

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 3}
def N : Set ℝ := {x : ℝ | x^2 - 3*x - 4 < 0}

theorem M_inter_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_M_l208_20826


namespace NUMINAMATH_CALUDE_tan_half_difference_l208_20896

theorem tan_half_difference (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5) 
  (h2 : Real.sin a + Real.sin b = 2/5) : 
  Real.tan ((a - b)/2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_difference_l208_20896


namespace NUMINAMATH_CALUDE_roots_in_specific_intervals_roots_in_unit_interval_l208_20863

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 2*m + 1

-- Part I
theorem roots_in_specific_intervals (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-1 : ℝ) 0 ∧ 
             x₂ ∈ Set.Ioo 1 2 ∧ 
             f m x₁ = 0 ∧ 
             f m x₂ = 0) →
  m ∈ Set.Ioo (-5/6 : ℝ) (-1/2) :=
sorry

-- Part II
theorem roots_in_unit_interval (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo 0 1 ∧ 
             x₂ ∈ Set.Ioo 0 1 ∧ 
             f m x₁ = 0 ∧ 
             f m x₂ = 0) →
  m ∈ Set.Ico (-1/2 : ℝ) (1 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_roots_in_specific_intervals_roots_in_unit_interval_l208_20863


namespace NUMINAMATH_CALUDE_triangle_tangent_slopes_sum_l208_20827

theorem triangle_tangent_slopes_sum (A B C : ℝ × ℝ) : 
  let triangle_slopes : List ℝ := [63, 73, 97]
  let curve (x : ℝ) := (x + 3) * (x^2 + 3)
  let tangent_slope (x : ℝ) := 3 * x^2 + 6 * x + 3
  (∀ p ∈ [A, B, C], p.1 ≥ 0 ∧ p.2 ≥ 0) →
  (∀ p ∈ [A, B, C], curve p.1 = p.2) →
  (List.zip [A, B, C] (A :: B :: C :: A :: nil)).all 
    (λ (p, q) => (q.2 - p.2) / (q.1 - p.1) ∈ triangle_slopes) →
  (tangent_slope A.1 + tangent_slope B.1 + tangent_slope C.1 = 237) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_slopes_sum_l208_20827


namespace NUMINAMATH_CALUDE_sod_area_calculation_l208_20864

-- Define the dimensions
def yard_width : ℕ := 200
def yard_depth : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def front_flowerbed_depth : ℕ := 4
def front_flowerbed_length : ℕ := 25
def third_flowerbed_width : ℕ := 10
def third_flowerbed_length : ℕ := 12
def fourth_flowerbed_width : ℕ := 7
def fourth_flowerbed_length : ℕ := 8

-- Define the theorem
theorem sod_area_calculation : 
  let total_area := yard_width * yard_depth
  let sidewalk_area := sidewalk_width * sidewalk_length
  let front_flowerbeds_area := 2 * (front_flowerbed_depth * front_flowerbed_length)
  let third_flowerbed_area := third_flowerbed_width * third_flowerbed_length
  let fourth_flowerbed_area := fourth_flowerbed_width * fourth_flowerbed_length
  let non_sod_area := sidewalk_area + front_flowerbeds_area + third_flowerbed_area + fourth_flowerbed_area
  total_area - non_sod_area = 9474 := by
sorry

end NUMINAMATH_CALUDE_sod_area_calculation_l208_20864


namespace NUMINAMATH_CALUDE_prob_two_heads_in_four_tosses_l208_20829

/-- The probability of getting exactly 2 heads when tossing a fair coin 4 times -/
theorem prob_two_heads_in_four_tosses : 
  let n : ℕ := 4  -- number of tosses
  let k : ℕ := 2  -- number of desired heads
  let p : ℚ := 1/2  -- probability of getting heads in a single toss
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 :=
sorry

end NUMINAMATH_CALUDE_prob_two_heads_in_four_tosses_l208_20829


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l208_20804

theorem right_triangle_shorter_leg : ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 39 →           -- Hypotenuse is 39 units
  a ≤ b →            -- a is the shorter leg
  a = 15 :=          -- The shorter leg is 15 units
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l208_20804


namespace NUMINAMATH_CALUDE_max_value_of_expression_upper_bound_achievable_l208_20833

theorem max_value_of_expression (x : ℝ) :
  x^4 / (x^8 + 2*x^6 - 3*x^4 + 5*x^3 + 8*x^2 + 5*x + 25) ≤ 1/15 :=
by sorry

theorem upper_bound_achievable :
  ∃ x : ℝ, x^4 / (x^8 + 2*x^6 - 3*x^4 + 5*x^3 + 8*x^2 + 5*x + 25) = 1/15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_upper_bound_achievable_l208_20833


namespace NUMINAMATH_CALUDE_inverse_of_singular_matrix_l208_20817

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 10, 6]

theorem inverse_of_singular_matrix :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_singular_matrix_l208_20817


namespace NUMINAMATH_CALUDE_fred_grew_38_cantelopes_l208_20865

/-- The number of cantelopes Tim grew -/
def tims_cantelopes : ℕ := 44

/-- The total number of cantelopes Fred and Tim grew together -/
def total_cantelopes : ℕ := 82

/-- The number of cantelopes Fred grew -/
def freds_cantelopes : ℕ := total_cantelopes - tims_cantelopes

/-- Theorem stating that Fred grew 38 cantelopes -/
theorem fred_grew_38_cantelopes : freds_cantelopes = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_38_cantelopes_l208_20865


namespace NUMINAMATH_CALUDE_black_white_pieces_difference_l208_20838

theorem black_white_pieces_difference (B W : ℕ) : 
  (B - 1) / W = 9 / 7 →
  B / (W - 1) = 7 / 5 →
  B - W = 7 :=
by sorry

end NUMINAMATH_CALUDE_black_white_pieces_difference_l208_20838


namespace NUMINAMATH_CALUDE_smaller_number_problem_l208_20861

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 45) : 
  min x y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l208_20861


namespace NUMINAMATH_CALUDE_rabbit_count_l208_20886

theorem rabbit_count (white_rabbits black_rabbits female_rabbits : ℕ) : 
  white_rabbits = 12 → black_rabbits = 9 → female_rabbits = 8 → 
  white_rabbits + black_rabbits - female_rabbits = 13 := by
sorry

end NUMINAMATH_CALUDE_rabbit_count_l208_20886


namespace NUMINAMATH_CALUDE_hiram_age_is_40_l208_20844

/-- Hiram's age in years -/
def hiram_age : ℕ := sorry

/-- Allyson's age in years -/
def allyson_age : ℕ := 28

/-- Theorem stating Hiram's age based on the given conditions -/
theorem hiram_age_is_40 :
  (hiram_age + 12 = 2 * allyson_age - 4) → hiram_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_hiram_age_is_40_l208_20844


namespace NUMINAMATH_CALUDE_prism_lateral_edges_parallel_equal_l208_20883

structure Prism where
  -- A prism is a polyhedron
  is_polyhedron : Bool
  -- A prism has two congruent and parallel bases
  has_congruent_parallel_bases : Bool
  -- The lateral faces of a prism are parallelograms
  lateral_faces_are_parallelograms : Bool

/-- The lateral edges of a prism are parallel and equal in length -/
theorem prism_lateral_edges_parallel_equal (p : Prism) :
  p.is_polyhedron ∧ p.has_congruent_parallel_bases ∧ p.lateral_faces_are_parallelograms →
  (lateral_edges_parallel : Bool) ∧ (lateral_edges_equal_length : Bool) :=
by sorry

end NUMINAMATH_CALUDE_prism_lateral_edges_parallel_equal_l208_20883
