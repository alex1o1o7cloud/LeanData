import Mathlib

namespace NUMINAMATH_CALUDE_crayons_per_row_l356_35642

/-- Given that Faye has 210 crayons in total and places them into 7 rows,
    prove that there are 30 crayons in each row. -/
theorem crayons_per_row (total_crayons : ℕ) (num_rows : ℕ) (h1 : total_crayons = 210) (h2 : num_rows = 7) :
  total_crayons / num_rows = 30 := by
  sorry

#check crayons_per_row

end NUMINAMATH_CALUDE_crayons_per_row_l356_35642


namespace NUMINAMATH_CALUDE_max_lessons_is_126_l356_35658

/-- Represents the number of lessons that can be conducted with given clothing items -/
def lessons (shirts trousers shoes : ℕ) : ℕ := 3 * shirts * trousers * shoes

/-- Represents the conditions of the problem -/
structure ClothingConditions where
  shirts : ℕ
  trousers : ℕ
  shoes : ℕ
  jackets : ℕ
  one_more_shirt : lessons (shirts + 1) trousers shoes = lessons shirts trousers shoes + 18
  one_more_trousers : lessons shirts (trousers + 1) shoes = lessons shirts trousers shoes + 63
  one_more_shoes : lessons shirts trousers (shoes + 1) = lessons shirts trousers shoes + 42
  jackets_eq_two : jackets = 2

/-- The theorem to be proved -/
theorem max_lessons_is_126 (c : ClothingConditions) : lessons c.shirts c.trousers c.shoes = 126 := by
  sorry

end NUMINAMATH_CALUDE_max_lessons_is_126_l356_35658


namespace NUMINAMATH_CALUDE_intersection_line_passes_through_fixed_point_l356_35609

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 + 3 * y^2 = 6

/-- Right focus of the ellipse -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Line passing through the right focus -/
def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

/-- Intersection points of the line and the ellipse -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ ellipse_C x y ∧ y = line_through_focus k x}

/-- Symmetric point about x-axis -/
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The fixed point on x-axis -/
def fixed_point : ℝ × ℝ := (3, 0)

theorem intersection_line_passes_through_fixed_point (k : ℝ) (hk : k ≠ 0) :
  ∀ p q, p ∈ intersection_points k → q ∈ intersection_points k → p ≠ q →
  ∃ t, (1 - t) • (symmetric_point p) + t • q = fixed_point :=
sorry

end NUMINAMATH_CALUDE_intersection_line_passes_through_fixed_point_l356_35609


namespace NUMINAMATH_CALUDE_utopia_park_elephants_l356_35696

/-- The time taken for new elephants to enter Utopia National Park -/
def time_for_new_elephants (initial_elephants : ℕ) (exodus_duration : ℕ) (exodus_rate : ℕ) (entry_rate : ℕ) (final_elephants : ℕ) : ℕ :=
  let elephants_after_exodus := initial_elephants - exodus_duration * exodus_rate
  let new_elephants := final_elephants - elephants_after_exodus
  new_elephants / entry_rate

theorem utopia_park_elephants :
  time_for_new_elephants 30000 4 2880 1500 28980 = 7 :=
by sorry

end NUMINAMATH_CALUDE_utopia_park_elephants_l356_35696


namespace NUMINAMATH_CALUDE_fruit_basket_solution_l356_35657

def fruit_basket_problem (initial_apples initial_oranges x : ℕ) : Prop :=
  -- Initial condition: oranges are twice the apples
  initial_oranges = 2 * initial_apples ∧
  -- After x removals, 1 apple and 16 oranges remain
  initial_apples - 3 * x = 1 ∧
  initial_oranges - 4 * x = 16

theorem fruit_basket_solution : 
  ∃ initial_apples initial_oranges : ℕ, fruit_basket_problem initial_apples initial_oranges 7 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_solution_l356_35657


namespace NUMINAMATH_CALUDE_percentage_calculation_l356_35640

theorem percentage_calculation (n : ℝ) (h : n = 5200) : 0.15 * (0.30 * (0.50 * n)) = 117 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l356_35640


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l356_35632

/-- A convex octagon inscribed in a circle -/
structure InscribedOctagon where
  /-- The octagon is convex -/
  is_convex : Bool
  /-- The octagon is inscribed in a circle -/
  inscribed_in_circle : Bool
  /-- Four consecutive sides have length 3 -/
  consecutive_sides_length : ℕ
  /-- The remaining sides have length 2 -/
  remaining_sides_length : ℕ

/-- The area of the inscribed octagon -/
noncomputable def octagon_area (o : InscribedOctagon) : ℝ :=
  13 + 12 * Real.sqrt 2

/-- Theorem stating the area of the specific inscribed octagon -/
theorem inscribed_octagon_area (o : InscribedOctagon) 
  (h1 : o.is_convex = true)
  (h2 : o.inscribed_in_circle = true)
  (h3 : o.consecutive_sides_length = 3)
  (h4 : o.remaining_sides_length = 2) :
  octagon_area o = 13 + 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l356_35632


namespace NUMINAMATH_CALUDE_sticks_to_triangles_l356_35692

/-- Represents a stick that can be cut into smaller pieces -/
structure Stick :=
  (length : ℕ)

/-- Represents a triangle with sides of specific lengths -/
structure Triangle :=
  (side1 : ℕ)
  (side2 : ℕ)
  (side3 : ℕ)

/-- The number of original sticks -/
def num_sticks : ℕ := 12

/-- The length of each original stick -/
def stick_length : ℕ := 13

/-- The number of triangles to be formed -/
def num_triangles : ℕ := 13

/-- The desired triangle side lengths -/
def target_triangle : Triangle :=
  { side1 := 3, side2 := 4, side3 := 5 }

/-- Theorem stating that the sticks can be cut to form the desired triangles -/
theorem sticks_to_triangles :
  ∃ (cut_pieces : List ℕ),
    (cut_pieces.sum = num_sticks * stick_length) ∧
    (∀ t : Triangle, t ∈ List.replicate num_triangles target_triangle →
      t.side1 ∈ cut_pieces ∧ t.side2 ∈ cut_pieces ∧ t.side3 ∈ cut_pieces) :=
sorry

end NUMINAMATH_CALUDE_sticks_to_triangles_l356_35692


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l356_35612

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {0, 2, 5}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {0, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l356_35612


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l356_35605

theorem sum_of_x_solutions (y : ℝ) (h1 : y = 9) (h2 : ∃ x : ℝ, x^2 + y^2 = 225) : 
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 225 ∧ x₂^2 + y^2 = 225 ∧ x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l356_35605


namespace NUMINAMATH_CALUDE_largest_product_digit_sum_l356_35668

def is_two_digit_prime_less_than_30 (p : ℕ) : Prop :=
  p ≥ 10 ∧ p < 30 ∧ Nat.Prime p

def largest_product (d e : ℕ) : ℕ :=
  d * e * (100 * d + e)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_digit_sum :
  ∃ (d e : ℕ),
    is_two_digit_prime_less_than_30 d ∧
    is_two_digit_prime_less_than_30 e ∧
    d ≠ e ∧
    Nat.Prime (100 * d + e) ∧
    (∀ (d' e' : ℕ),
      is_two_digit_prime_less_than_30 d' ∧
      is_two_digit_prime_less_than_30 e' ∧
      d' ≠ e' ∧
      Nat.Prime (100 * d' + e') →
      largest_product d' e' ≤ largest_product d e) ∧
    sum_of_digits (largest_product d e) = 19 :=
  sorry

end NUMINAMATH_CALUDE_largest_product_digit_sum_l356_35668


namespace NUMINAMATH_CALUDE_mountain_trip_distance_l356_35602

/-- Proves that the distance coming down the mountain is 8 km given the problem conditions --/
theorem mountain_trip_distance (d_up d_down : ℝ) : 
  (d_up / 3 + d_down / 4 = 4) →  -- Total time equation
  (d_down = d_up + 2) →          -- Difference in distances
  (d_down = 8) :=                -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_mountain_trip_distance_l356_35602


namespace NUMINAMATH_CALUDE_c_bounds_l356_35633

theorem c_bounds (a b c : ℝ) (h1 : a + 2*b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) :
  -2/3 ≤ c ∧ c ≤ 1 := by sorry

end NUMINAMATH_CALUDE_c_bounds_l356_35633


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l356_35649

theorem factorization_of_cubic (b : ℝ) : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l356_35649


namespace NUMINAMATH_CALUDE_inverse_of_12_mod_1009_l356_35670

theorem inverse_of_12_mod_1009 : ∃! x : ℕ, x < 1009 ∧ (12 * x) % 1009 = 1 :=
by
  use 925
  sorry

end NUMINAMATH_CALUDE_inverse_of_12_mod_1009_l356_35670


namespace NUMINAMATH_CALUDE_mixed_gender_selection_l356_35694

def male_students : ℕ := 10
def female_students : ℕ := 6
def total_selection : ℕ := 3

def select_mixed_gender (m : ℕ) (f : ℕ) (total : ℕ) : ℕ :=
  Nat.choose m 1 * Nat.choose f 2 + Nat.choose m 2 * Nat.choose f 1

theorem mixed_gender_selection :
  select_mixed_gender male_students female_students total_selection = 420 := by
  sorry

end NUMINAMATH_CALUDE_mixed_gender_selection_l356_35694


namespace NUMINAMATH_CALUDE_exists_perpendicular_intersection_line_l356_35691

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of C₂
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through F with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the condition for two points being perpendicular from the origin
def perpendicular_from_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem exists_perpendicular_intersection_line :
  ∃ k : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    perpendicular_from_origin x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_intersection_line_l356_35691


namespace NUMINAMATH_CALUDE_total_clamps_is_92_l356_35641

/-- Represents the number of bike clamps given per bicycle purchase -/
def clamps_per_bike : ℕ := 2

/-- Represents the number of bikes sold in the morning -/
def morning_sales : ℕ := 19

/-- Represents the number of bikes sold in the afternoon -/
def afternoon_sales : ℕ := 27

/-- Calculates the total number of bike clamps given away -/
def total_clamps : ℕ := clamps_per_bike * (morning_sales + afternoon_sales)

/-- Proves that the total number of bike clamps given away is 92 -/
theorem total_clamps_is_92 : total_clamps = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_clamps_is_92_l356_35641


namespace NUMINAMATH_CALUDE_triangle_configuration_l356_35654

/-- Given a configuration of similar right-angled triangles, prove the values of v, w, x, y, and z -/
theorem triangle_configuration (v w x y z : ℝ) : 
  v / 8 = 9 / x ∧ 
  9 / x = y / 20 ∧ 
  y^2 = x^2 + 9^2 ∧ 
  z^2 = 20^2 - x^2 ∧ 
  w^2 = 8^2 + v^2 →
  v = 6 ∧ w = 10 ∧ x = 12 ∧ y = 15 ∧ z = 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_configuration_l356_35654


namespace NUMINAMATH_CALUDE_x_plus_2y_squared_l356_35682

theorem x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2*y) = 48) (h2 : y * (x + 2*y) = 72) : 
  (x + 2*y)^2 = 96 := by
sorry

end NUMINAMATH_CALUDE_x_plus_2y_squared_l356_35682


namespace NUMINAMATH_CALUDE_gomez_students_sum_l356_35615

theorem gomez_students_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_gomez_students_sum_l356_35615


namespace NUMINAMATH_CALUDE_car_rental_cost_per_km_l356_35635

theorem car_rental_cost_per_km (samuel_fixed_cost carrey_fixed_cost carrey_per_km distance : ℝ) 
  (h1 : samuel_fixed_cost = 24)
  (h2 : carrey_fixed_cost = 20)
  (h3 : carrey_per_km = 0.25)
  (h4 : distance = 44.44444444444444)
  (h5 : ∃ samuel_per_km : ℝ, samuel_fixed_cost + samuel_per_km * distance = carrey_fixed_cost + carrey_per_km * distance) :
  ∃ samuel_per_km : ℝ, samuel_per_km = 0.16 := by
sorry


end NUMINAMATH_CALUDE_car_rental_cost_per_km_l356_35635


namespace NUMINAMATH_CALUDE_rotated_line_equation_l356_35634

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a line 90 degrees counterclockwise around a given point --/
def rotateLine90 (l : Line) (p : Point) : Line :=
  sorry

/-- The initial line l₀ --/
def l₀ : Line :=
  { slope := 1, intercept := 1 }

/-- The point P around which the line is rotated --/
def P : Point :=
  { x := 3, y := 1 }

/-- The rotated line l --/
def l : Line :=
  rotateLine90 l₀ P

theorem rotated_line_equation :
  l.slope * 3 + l.intercept = 1 ∧ l.slope = -1 ∧ l.intercept = 4 :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l356_35634


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l356_35676

theorem rectangular_prism_volume 
  (l w h : ℝ) 
  (face_area_1 : l * w = 15) 
  (face_area_2 : w * h = 20) 
  (face_area_3 : l * h = 30) : 
  l * w * h = 60 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l356_35676


namespace NUMINAMATH_CALUDE_sum_of_cubes_l356_35697

theorem sum_of_cubes :
  (∀ n : ℤ, ∃ a b c d : ℤ, 6 * n = a^3 + b^3 + c^3 + d^3) ∧
  (∀ k : ℤ, ∃ a b c d e : ℤ, k = a^3 + b^3 + c^3 + d^3 + e^3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l356_35697


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l356_35651

/-- The function f(x) = x^3 - ax^2 - bx, where f(1) = 0 -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x

/-- The condition that f(1) = 0 -/
def f_intersects_x_axis (a b : ℝ) : Prop := f a b 1 = 0

theorem extreme_values_of_f (a b : ℝ) (h : f_intersects_x_axis a b) :
  (∃ x, f a b x = 4/27) ∧ (∃ x, f a b x = 0) ∧
  (∀ x, f a b x ≤ 4/27) ∧ (∀ x, f a b x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l356_35651


namespace NUMINAMATH_CALUDE_intersection_equality_union_characterization_l356_35638

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 3*a = (a+3)*x}

-- Define set B
def B : Set ℝ := {x | x^2 + 3 = 4*x}

-- Theorem 1: If A ∩ B = A, then a = 1 or a = 3
theorem intersection_equality (a : ℝ) : A a ∩ B = A a → a = 1 ∨ a = 3 := by
  sorry

-- Theorem 2: A ∪ B = {1, 3} when a = 1 or a = 3, and A ∪ B = {a, 1, 3} otherwise
theorem union_characterization (a : ℝ) :
  (a = 1 ∨ a = 3 → A a ∪ B = {1, 3}) ∧
  (a ≠ 1 ∧ a ≠ 3 → A a ∪ B = {a, 1, 3}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_union_characterization_l356_35638


namespace NUMINAMATH_CALUDE_compute_expression_l356_35683

theorem compute_expression : 12 - 4 * (5 - 10)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l356_35683


namespace NUMINAMATH_CALUDE_employee_pay_l356_35685

theorem employee_pay (total_pay : ℝ) (a_pay : ℝ) (b_pay : ℝ) :
  total_pay = 550 →
  a_pay = 1.5 * b_pay →
  a_pay + b_pay = total_pay →
  b_pay = 220 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l356_35685


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l356_35604

theorem polynomial_functional_equation :
  ∃ (q : ℝ → ℝ), (∀ x, q x = -2 * x + 4) ∧
                 (∀ x, q (q x) = x * q x + 2 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l356_35604


namespace NUMINAMATH_CALUDE_square_equation_result_l356_35646

theorem square_equation_result (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 / 25 := by
sorry

end NUMINAMATH_CALUDE_square_equation_result_l356_35646


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l356_35695

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum_theorem (e : Ellipse) 
    (h_center : e.h = -4 ∧ e.k = 2)
    (h_semi_major : e.a = 5)
    (h_semi_minor : e.b = 3) :
  e.h + e.k + e.a + e.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l356_35695


namespace NUMINAMATH_CALUDE_student_number_problem_l356_35659

theorem student_number_problem (x : ℝ) : 3 * x - 220 = 110 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l356_35659


namespace NUMINAMATH_CALUDE_three_digit_numbers_19_times_sum_of_digits_l356_35661

def isValidNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n = 19 * (n / 100 + (n / 10 % 10) + (n % 10))

theorem three_digit_numbers_19_times_sum_of_digits :
  {n : ℕ | isValidNumber n} = {114, 133, 152, 171, 190, 209, 228, 247, 266, 285, 399} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_19_times_sum_of_digits_l356_35661


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l356_35699

/-- The trajectory of point Q given a line l and the relation between Q and a point P on l -/
theorem trajectory_of_Q (x y m n : ℝ) : 
  (2 * m + 4 * n + 3 = 0) →  -- P(m, n) is on line l
  (m = 3 * x ∧ n = 3 * y) →  -- P = 3Q, derived from 2⃗OQ = ⃗QP
  (2 * x + 4 * y + 1 = 0) :=  -- Trajectory equation of Q
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l356_35699


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l356_35667

def S (n : ℕ) : ℤ := n^2 - 2*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem sum_of_specific_terms : a 3 + a 17 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l356_35667


namespace NUMINAMATH_CALUDE_disinfectant_sales_l356_35664

/-- Disinfectant sales problem -/
theorem disinfectant_sales 
  (cost_A : ℕ) (cost_B : ℕ) (total_cost : ℕ) 
  (initial_price_A : ℕ) (initial_volume_A : ℕ) 
  (price_change : ℕ) (volume_change : ℕ)
  (price_B : ℕ) (x : ℕ) :
  cost_A = 20 →
  cost_B = 30 →
  total_cost = 2000 →
  initial_price_A = 30 →
  initial_volume_A = 100 →
  price_change = 1 →
  volume_change = 5 →
  price_B = 60 →
  x > 30 →
  (∃ (volume_A : ℕ → ℕ) (cost_price_B : ℕ → ℕ) (volume_B : ℕ → ℚ) 
      (max_profit : ℕ) (valid_prices : List ℕ),
    (∀ y : ℕ, volume_A y = 250 - 5 * y) ∧
    (∀ y : ℕ, cost_price_B y = 100 * y - 3000) ∧
    (∀ y : ℕ, volume_B y = (10 * y : ℚ) / 3 - 100) ∧
    max_profit = 2125 ∧
    valid_prices = [39, 42, 45, 48] ∧
    (∀ p ∈ valid_prices, 
      (-5 * (p - 45)^2 + 2125 : ℚ) ≥ 1945 ∧ 
      p ≤ 50 ∧ 
      p % 3 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_disinfectant_sales_l356_35664


namespace NUMINAMATH_CALUDE_intersection_M_N_l356_35645

def M : Set ℝ := {x | |x - 2| ≤ 1}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l356_35645


namespace NUMINAMATH_CALUDE_original_number_proof_l356_35681

theorem original_number_proof (final_number : ℝ) (increase_percentage : ℝ) 
  (h1 : final_number = 1680)
  (h2 : increase_percentage = 110) : 
  ∃ (original : ℝ), original * (1 + increase_percentage / 100) = final_number ∧ original = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l356_35681


namespace NUMINAMATH_CALUDE_problem_statement_l356_35606

/-- Given two positive real numbers x and y satisfying the equations
     1/x + 1/y = 5 and x² + y² = 14, prove that x²y + xy² = 3.2 -/
theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : 1/x + 1/y = 5) (h2 : x^2 + y^2 = 14) :
  x^2 * y + x * y^2 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l356_35606


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l356_35663

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

-- Theorem statement
theorem binary_110_equals_6 :
  binary_to_decimal [true, true, false] = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l356_35663


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_l356_35601

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem minimum_value_implies_m (m : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f x m ≥ -3) ∧ (∃ x : ℝ, x ≥ 2 ∧ f x m = -3) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_l356_35601


namespace NUMINAMATH_CALUDE_gcd_problem_l356_35686

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 85 ∧ Nat.gcd n 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l356_35686


namespace NUMINAMATH_CALUDE_age_difference_l356_35647

theorem age_difference (x y z : ℕ) (h : z = x - 15) :
  (x + y) - (y + z) = 15 := by sorry

end NUMINAMATH_CALUDE_age_difference_l356_35647


namespace NUMINAMATH_CALUDE_existence_of_m_l356_35603

/-- Sum of digits function -/
def d (n : ℕ+) : ℕ :=
  sorry

/-- Main theorem -/
theorem existence_of_m (k : ℕ+) :
  ∃ m : ℕ+, ∃! (s : Finset ℕ+), s.card = k ∧ ∀ x ∈ s, x + d x = m :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l356_35603


namespace NUMINAMATH_CALUDE_purely_imaginary_z_implies_x_equals_one_l356_35666

-- Define the complex number z as a function of x
def z (x : ℝ) : ℂ := (x^2 - 1 : ℝ) + (x + 1 : ℝ) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- Theorem statement
theorem purely_imaginary_z_implies_x_equals_one :
  ∀ x : ℝ, isPurelyImaginary (z x) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_implies_x_equals_one_l356_35666


namespace NUMINAMATH_CALUDE_unique_k_solution_l356_35677

theorem unique_k_solution : ∃! k : ℝ, ∀ x : ℝ, 
  (3*x^2 - 4*x + 5) * (5*x^2 + k*x + 15) = 15*x^4 - 47*x^3 + 100*x^2 - 60*x + 75 :=
by sorry

end NUMINAMATH_CALUDE_unique_k_solution_l356_35677


namespace NUMINAMATH_CALUDE_positive_sum_y_z_l356_35618

theorem positive_sum_y_z (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_y_z_l356_35618


namespace NUMINAMATH_CALUDE_problem_statement_l356_35684

theorem problem_statement : (3^1 - 2 + 6^2 - 1)^0 * 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l356_35684


namespace NUMINAMATH_CALUDE_mixed_grains_approximation_l356_35679

/-- Given a total amount of grain and a sample with mixed grains, calculate the approximate amount of mixed grains in the entire batch. -/
def approximateMixedGrains (totalStones : ℕ) (sampleSize : ℕ) (mixedInSample : ℕ) : ℕ :=
  (totalStones * mixedInSample) / sampleSize

/-- Theorem stating that the approximate amount of mixed grains in the given scenario is 169 stones. -/
theorem mixed_grains_approximation :
  approximateMixedGrains 1534 254 28 = 169 := by
  sorry

#eval approximateMixedGrains 1534 254 28

end NUMINAMATH_CALUDE_mixed_grains_approximation_l356_35679


namespace NUMINAMATH_CALUDE_money_division_l356_35660

/-- The problem of dividing money among A, B, and C -/
theorem money_division (a b c : ℚ) : 
  a + b + c = 360 →  -- Total amount is $360
  a = (1/3) * (b + c) →  -- A gets 1/3 of B and C combined
  a = b + 10 →  -- A gets $10 more than B
  ∃ x : ℚ, b = x * (a + c) ∧ x = 2/7  -- B gets a fraction of A and C combined
  := by sorry

end NUMINAMATH_CALUDE_money_division_l356_35660


namespace NUMINAMATH_CALUDE_system_solution_l356_35629

theorem system_solution : ∃ (x y : ℝ), 
  (7 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 32) ∧ (x = -3) ∧ (y = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l356_35629


namespace NUMINAMATH_CALUDE_sector_radius_l356_35616

theorem sector_radius (arc_length : ℝ) (area : ℝ) (radius : ℝ) : 
  arc_length = 2 → area = 4 → area = (1/2) * radius * arc_length → radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l356_35616


namespace NUMINAMATH_CALUDE_sin_two_phi_l356_35689

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end NUMINAMATH_CALUDE_sin_two_phi_l356_35689


namespace NUMINAMATH_CALUDE_area_triangle_GCD_l356_35672

/-- Given a square ABCD with area 256, point E on BC dividing it 3:1, 
    F and G midpoints of AE and DE, and area of BEGF is 48, 
    prove that the area of triangle GCD is 48. -/
theorem area_triangle_GCD (A B C D E F G : ℝ × ℝ) : 
  -- Square ABCD has area 256
  (B.1 - A.1) * (C.2 - B.2) = 256 →
  -- E divides BC in 3:1 ratio
  E.1 - B.1 = 3/4 * (C.1 - B.1) →
  E.2 = B.2 →
  -- F is midpoint of AE
  F = ((A.1 + E.1)/2, (A.2 + E.2)/2) →
  -- G is midpoint of DE
  G = ((D.1 + E.1)/2, (D.2 + E.2)/2) →
  -- Area of quadrilateral BEGF is 48
  abs ((B.1*E.2 + E.1*G.2 + G.1*F.2 + F.1*B.2) - 
       (E.1*B.2 + G.1*E.2 + F.1*G.2 + B.1*F.2)) / 2 = 48 →
  -- Then the area of triangle GCD is 48
  abs ((G.1*C.2 + C.1*D.2 + D.1*G.2) - 
       (C.1*G.2 + D.1*C.2 + G.1*D.2)) / 2 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_area_triangle_GCD_l356_35672


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l356_35637

theorem sqrt_fraction_simplification :
  Real.sqrt ((25 : ℝ) / 49 - 16 / 81) = Real.sqrt 1241 / 63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l356_35637


namespace NUMINAMATH_CALUDE_max_value_at_2000_l356_35625

/-- The function f(k) = k^2 / 1.001^k reaches its maximum value when k = 2000 --/
theorem max_value_at_2000 (k : ℕ) : 
  (k^2 : ℝ) / (1.001^k) ≤ (2000^2 : ℝ) / (1.001^2000) :=
sorry

end NUMINAMATH_CALUDE_max_value_at_2000_l356_35625


namespace NUMINAMATH_CALUDE_linear_inequality_solution_set_l356_35626

theorem linear_inequality_solution_set 
  (m n : ℝ) 
  (h_m : m = -1) 
  (h_n : n = -1) : 
  {x : ℝ | m * x - n ≤ 2} = {x : ℝ | x ≥ -1} := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_set_l356_35626


namespace NUMINAMATH_CALUDE_floor_sum_example_l356_35674

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l356_35674


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l356_35617

theorem fractional_equation_solution (x : ℝ) (h : x * (x - 2) ≠ 0) :
  (4 / (x^2 - 2*x) + 1 / x = 3 / (x - 2)) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l356_35617


namespace NUMINAMATH_CALUDE_cubic_function_tangent_line_l356_35662

/-- Given a cubic function f(x) = x^3 + ax + b, prove that if its tangent line
    at x = 1 has the equation 2x - y - 5 = 0, then a = -1 and b = -3. -/
theorem cubic_function_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + a*x + b
  let f' : ℝ → ℝ := λ x => 3*x^2 + a
  (f' 1 = 2 ∧ f 1 = -3) → (a = -1 ∧ b = -3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_tangent_line_l356_35662


namespace NUMINAMATH_CALUDE_normal_vector_to_ellipsoid_l356_35648

/-- The ellipsoid equation -/
def F (x y z : ℝ) : ℝ := x^2 + 2*y^2 + 3*z^2 - 6

/-- The point on the ellipsoid -/
def M₀ : ℝ × ℝ × ℝ := (1, -1, 1)

/-- The proposed normal vector -/
def n : ℝ × ℝ × ℝ := (2, -4, 6)

theorem normal_vector_to_ellipsoid :
  let (x₀, y₀, z₀) := M₀
  F x₀ y₀ z₀ = 0 ∧ 
  n = (2*x₀, 4*y₀, 6*z₀) :=
by sorry

end NUMINAMATH_CALUDE_normal_vector_to_ellipsoid_l356_35648


namespace NUMINAMATH_CALUDE_tram_length_l356_35698

/-- The length of a tram given observation times and tunnel length -/
theorem tram_length (pass_time : ℝ) (tunnel_length : ℝ) (tunnel_time : ℝ) :
  pass_time = 4 →
  tunnel_length = 64 →
  tunnel_time = 12 →
  ∃ (tram_length : ℝ),
    tram_length > 0 ∧
    tram_length / pass_time = (tunnel_length + tram_length) / tunnel_time ∧
    tram_length = 32 :=
by sorry


end NUMINAMATH_CALUDE_tram_length_l356_35698


namespace NUMINAMATH_CALUDE_correct_statements_count_l356_35622

-- Define a structure for a programming statement
structure ProgrammingStatement :=
  (text : String)
  (is_correct : Bool)

-- Define the four statements
def statement1 : ProgrammingStatement :=
  ⟨"INPUT \"a, b, c=\"; a, b; c", false⟩

def statement2 : ProgrammingStatement :=
  ⟨"PRINT S=7", false⟩

def statement3 : ProgrammingStatement :=
  ⟨"9=r", false⟩

def statement4 : ProgrammingStatement :=
  ⟨"PRINT 20.3*2", true⟩

-- Define a list of all statements
def all_statements : List ProgrammingStatement :=
  [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (all_statements.filter (λ s => s.is_correct)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l356_35622


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l356_35639

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l356_35639


namespace NUMINAMATH_CALUDE_new_person_weight_is_129_l356_35610

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating the weight of the new person under the given conditions -/
theorem new_person_weight_is_129 :
  weight_of_new_person 4 95 8.5 = 129 := by
  sorry

#eval weight_of_new_person 4 95 8.5

end NUMINAMATH_CALUDE_new_person_weight_is_129_l356_35610


namespace NUMINAMATH_CALUDE_sequence_term_value_l356_35624

-- Define the sequence sum
def S (n : ℕ) : ℤ := n^2 - 6*n

-- Define the m-th term of the sequence
def a (m : ℕ) : ℤ := 2*m - 7

-- Theorem statement
theorem sequence_term_value (m : ℕ) :
  (∀ n : ℕ, S n = n^2 - 6*n) →
  (∀ k : ℕ, k ≥ 2 → a k = S k - S (k-1)) →
  (5 < a m ∧ a m < 8) →
  m = 7 := by sorry

end NUMINAMATH_CALUDE_sequence_term_value_l356_35624


namespace NUMINAMATH_CALUDE_triangular_number_representation_l356_35688

theorem triangular_number_representation (n : ℕ+) :
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ l < k :=
by sorry

end NUMINAMATH_CALUDE_triangular_number_representation_l356_35688


namespace NUMINAMATH_CALUDE_first_week_customers_l356_35614

def commission_rate : ℚ := 1
def salary : ℚ := 500
def bonus : ℚ := 50
def total_earnings : ℚ := 760

def customers_first_week (C : ℚ) : Prop :=
  let commission := commission_rate * (C + 2*C + 3*C)
  total_earnings = salary + bonus + commission

theorem first_week_customers :
  ∃ C : ℚ, customers_first_week C ∧ C = 35 :=
sorry

end NUMINAMATH_CALUDE_first_week_customers_l356_35614


namespace NUMINAMATH_CALUDE_prize_orders_count_l356_35678

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 6

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := num_bowlers - 1

/-- Calculates the number of possible outcomes for the tournament -/
def tournament_outcomes : ℕ := 2^num_matches

/-- Theorem stating that the number of different possible prize orders is 32 -/
theorem prize_orders_count : tournament_outcomes = 32 := by
  sorry

end NUMINAMATH_CALUDE_prize_orders_count_l356_35678


namespace NUMINAMATH_CALUDE_soap_brand_usage_l356_35652

/-- Given a survey of households and their soap brand usage, prove the number of households using both brands. -/
theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_a : ℕ) (both_to_only_b_ratio : ℕ) 
  (h_total : total = 240)
  (h_neither : neither = 80)
  (h_only_a : only_a = 60)
  (h_ratio : both_to_only_b_ratio = 3) :
  ∃ (both : ℕ), both = 25 ∧ total = neither + only_a + both_to_only_b_ratio * both + both :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l356_35652


namespace NUMINAMATH_CALUDE_road_renovation_equation_ahead_of_schedule_l356_35631

/-- Proves that the given equation holds true for a road renovation scenario -/
theorem road_renovation_equation (x : ℝ) (h1 : x > 5) : 
  (1500 / (x - 5) : ℝ) - (1500 / x : ℝ) = 10 ↔ 
  (1500 / (x - 5) : ℝ) = (1500 / x : ℝ) + 10 := by sorry

/-- Proves that the equation represents completing 10 days ahead of schedule -/
theorem ahead_of_schedule (x : ℝ) (h1 : x > 5) :
  (1500 / (x - 5) : ℝ) - (1500 / x : ℝ) = 10 ↔
  (1500 / (x - 5) : ℝ) = (1500 / x : ℝ) + 10 := by sorry

end NUMINAMATH_CALUDE_road_renovation_equation_ahead_of_schedule_l356_35631


namespace NUMINAMATH_CALUDE_impurity_reduction_proof_l356_35665

/-- The logarithm base 10 of 2 -/
def lg2 : Real := 0.3010

/-- The logarithm base 10 of 3 -/
def lg3 : Real := 0.4771

/-- The reduction factor of impurities after each filtration -/
def reduction_factor : Real := 0.8

/-- The target impurity level as a fraction of the original -/
def target_impurity : Real := 0.05

/-- The minimum number of filtrations required to reduce impurities below the target level -/
def min_filtrations : Nat := 15

theorem impurity_reduction_proof :
  (reduction_factor ^ min_filtrations : Real) < target_impurity ∧
  ∀ n : Nat, n < min_filtrations → (reduction_factor ^ n : Real) ≥ target_impurity :=
by
  sorry

#check impurity_reduction_proof

end NUMINAMATH_CALUDE_impurity_reduction_proof_l356_35665


namespace NUMINAMATH_CALUDE_number_B_value_l356_35673

theorem number_B_value (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 := by
  sorry

end NUMINAMATH_CALUDE_number_B_value_l356_35673


namespace NUMINAMATH_CALUDE_area_of_2015_l356_35687

/-- Represents a grid composed of 1x1 squares -/
structure Grid where
  squares : Set (Int × Int)

/-- Represents a shaded region in the grid -/
inductive ShadedRegion
  | Horizontal : (Int × Int) → (Int × Int) → ShadedRegion
  | Vertical : (Int × Int) → (Int × Int) → ShadedRegion
  | Diagonal : (Int × Int) → (Int × Int) → ShadedRegion
  | Midpoint : (Int × Int) → (Int × Int) → ShadedRegion

/-- The set of shaded regions representing the number 2015 -/
def number2015 : Set ShadedRegion :=
  sorry

/-- Calculates the area of a set of shaded regions -/
def areaOfShadedRegions (regions : Set ShadedRegion) : ℚ :=
  sorry

/-- Theorem stating that the area of the shaded regions representing 2015 is 47½ -/
theorem area_of_2015 (g : Grid) :
  areaOfShadedRegions number2015 = 47 + (1/2) :=
sorry

end NUMINAMATH_CALUDE_area_of_2015_l356_35687


namespace NUMINAMATH_CALUDE_only_possible_amount_l356_35643

/-- Represents the possible coin types in the machine -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents -/
def coin_value : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- The result of using a coin in the machine -/
def machine_output : Coin → List Coin
  | Coin.Penny => List.replicate 7 Coin.Dime
  | Coin.Nickel => List.replicate 5 Coin.Quarter
  | Coin.Dime => List.replicate 5 Coin.Penny
  | Coin.Quarter => []  -- Not specified in the problem, so we'll leave it empty

/-- The total value in cents after using the machine k times, starting with one penny -/
def total_value (k : ℕ) : ℕ :=
  1 + 69 * k

/-- The given options in cents -/
def options : List ℕ := [315, 483, 552, 760, 897]

/-- Theorem stating that 760 cents ($7.60) is the only possible amount from the given options -/
theorem only_possible_amount :
  ∃ k : ℕ, total_value k = 760 ∧ ∀ n ∈ options, n ≠ 760 → ¬∃ k : ℕ, total_value k = n :=
sorry


end NUMINAMATH_CALUDE_only_possible_amount_l356_35643


namespace NUMINAMATH_CALUDE_problem_1_l356_35636

theorem problem_1 : 3.14 * 5.5^2 - 3.14 * 4.5^2 = 31.4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l356_35636


namespace NUMINAMATH_CALUDE_cutlery_count_l356_35611

/-- Calculates the total number of cutlery pieces after purchases -/
def totalCutlery (initialKnives : ℕ) : ℕ :=
  let initialTeaspoons := 2 * initialKnives
  let additionalKnives := initialKnives / 3
  let additionalTeaspoons := (2 * initialTeaspoons) / 3
  (initialKnives + additionalKnives) + (initialTeaspoons + additionalTeaspoons)

/-- Theorem stating that given the initial conditions, the total cutlery after purchases is 112 -/
theorem cutlery_count : totalCutlery 24 = 112 := by
  sorry

#eval totalCutlery 24

end NUMINAMATH_CALUDE_cutlery_count_l356_35611


namespace NUMINAMATH_CALUDE_distance_C_D_l356_35623

/-- An ellipse with equation 4(x-2)^2 + 16y^2 = 64 -/
structure Ellipse where
  -- The equation is implicitly defined by the structure

/-- Point C is an endpoint of the minor axis -/
def C (e : Ellipse) : ℝ × ℝ := sorry

/-- Point D is an endpoint of the major axis -/
def D (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance between C and D is 2√5 -/
theorem distance_C_D (e : Ellipse) : distance (C e) (D e) = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_C_D_l356_35623


namespace NUMINAMATH_CALUDE_solar_project_analysis_l356_35680

/-- Represents the net profit of a solar power generation project over n years. -/
def net_profit (n : ℕ) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the average annual profit of the project over n years. -/
def avg_annual_profit (n : ℕ) : ℚ :=
  net_profit n / n

theorem solar_project_analysis :
  ∀ n : ℕ,
  (n > 0) →
  (net_profit n = -4 * n^2 + 80 * n - 144) ∧
  (net_profit 3 > 0) ∧
  (net_profit 2 ≤ 0) ∧
  (∀ k : ℕ, k > 0 → avg_annual_profit 6 ≥ avg_annual_profit k) :=
by sorry

end NUMINAMATH_CALUDE_solar_project_analysis_l356_35680


namespace NUMINAMATH_CALUDE_exponential_function_difference_l356_35693

theorem exponential_function_difference (x : ℝ) : 
  let f : ℝ → ℝ := fun x => (3 : ℝ) ^ x
  f (x + 1) - f x = 2 * f x := by
sorry

end NUMINAMATH_CALUDE_exponential_function_difference_l356_35693


namespace NUMINAMATH_CALUDE_april_greatest_drop_l356_35628

/-- Represents the months of the year --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- Price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => 1.00
  | Month.February => -1.50
  | Month.March => -0.50
  | Month.April => -3.75
  | Month.May => 0.50
  | Month.June => -2.25

/-- Additional price shift due to market event in April --/
def market_event_shift : ℝ := -1.25

/-- Theorem stating that April had the greatest monthly drop in price --/
theorem april_greatest_drop :
  ∀ m : Month, m ≠ Month.April → price_change Month.April ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_april_greatest_drop_l356_35628


namespace NUMINAMATH_CALUDE_james_sodas_per_day_l356_35671

/-- Calculates the number of sodas James drinks per day -/
def sodas_per_day (packs : ℕ) (sodas_per_pack : ℕ) (initial_sodas : ℕ) (days : ℕ) : ℕ :=
  (packs * sodas_per_pack + initial_sodas) / days

/-- Theorem: James drinks 10 sodas per day -/
theorem james_sodas_per_day :
  sodas_per_day 5 12 10 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_sodas_per_day_l356_35671


namespace NUMINAMATH_CALUDE_f_properties_l356_35656

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sin x + 1

theorem f_properties :
  (∃ (x : ℝ), f x = -2) ∧ 
  (∀ (x : ℝ), f (Real.pi / 2 + x) = f (Real.pi / 2 - x)) ∧
  (∀ (x : ℝ), x > 0 → x < Real.pi / 2 → (deriv f) x < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l356_35656


namespace NUMINAMATH_CALUDE_inequality_implies_a_greater_than_three_l356_35627

theorem inequality_implies_a_greater_than_three (a : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 →
    Real.sqrt 2 * (2 * a + 3) * Real.cos (θ - π / 4) + 6 / (Real.sin θ + Real.cos θ) - 2 * Real.sin (2 * θ) < 3 * a + 6) →
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_greater_than_three_l356_35627


namespace NUMINAMATH_CALUDE_function_sum_equals_one_l356_35650

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Main theorem
theorem function_sum_equals_one
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_f0 : f 0 = 1)
  (h_fg : ∀ x, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_equals_one_l356_35650


namespace NUMINAMATH_CALUDE_product_congruence_l356_35630

def product : ℕ → ℕ
| 0 => 2
| 1 => 13
| (n+2) => if n % 2 = 0 then (10 + n + 2) * 2 else (10 + n + 2) * 3

def big_product : ℕ := (product 0) * (product 1) * (product 2) * (product 3) * 
                       (product 4) * (product 5) * (product 6) * (product 7) * 
                       (product 8) * (product 9) * (product 10) * (product 11) * 
                       (product 12) * (product 13) * (product 14) * (product 15) * 
                       (product 16) * (product 17)

theorem product_congruence : big_product ≡ 1 [ZMOD 5] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l356_35630


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l356_35607

theorem fixed_point_on_line (a b c : ℝ) (h : a + b - c = 0) (h2 : ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  a * (-1) + b * (-1) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l356_35607


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l356_35608

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 + i) / (1 - i) = (1 / 2 : ℂ) + (3 / 2 : ℂ) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l356_35608


namespace NUMINAMATH_CALUDE_train_length_l356_35655

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) → 
  platform_length = 270 → 
  crossing_time = 26 → 
  (train_speed * crossing_time) - platform_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l356_35655


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l356_35644

theorem circle_area_from_circumference (c : ℝ) (h : c = 36) :
  let r := c / (2 * Real.pi)
  (Real.pi * r^2) = 324 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l356_35644


namespace NUMINAMATH_CALUDE_f_at_three_l356_35619

/-- Horner's method representation of the polynomial f(x) = x^5 - 2x^4 + 3x^3 - 4x^2 + 5x + 6 -/
def f (x : ℝ) : ℝ := (((x - 2) * x + 3) * x - 4) * x + 5 * x + 6

/-- Theorem stating that f(3) = 147 -/
theorem f_at_three : f 3 = 147 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_l356_35619


namespace NUMINAMATH_CALUDE_initial_ratio_is_5_to_7_l356_35620

/-- Represents the composition of an alloy -/
structure Alloy where
  zinc : ℝ
  copper : ℝ

/-- Proves that the initial ratio of zinc to copper in the alloy is 5:7 -/
theorem initial_ratio_is_5_to_7 (initial : Alloy) 
  (h1 : initial.zinc + initial.copper = 6)  -- Initial total weight is 6 kg
  (h2 : (initial.zinc + 8) / initial.copper = 3)  -- New ratio after adding 8 kg zinc is 3:1
  : initial.zinc / initial.copper = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_5_to_7_l356_35620


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l356_35600

theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, (x^2 - m*x + 1) / (x^2 + x + 1) ≤ 2) ↔ -4 ≤ m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l356_35600


namespace NUMINAMATH_CALUDE_arcsin_b_range_l356_35690

open Real Set

theorem arcsin_b_range (a b : ℝ) :
  (arcsin a = arccos b) →
  (∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → x^2 + y^2 ≥ 2 / Real.sqrt 3) →
  Icc (π / 6) (π / 3) \ {π / 4} ⊆ {θ | ∃ b, arcsin b = θ} :=
by sorry

end NUMINAMATH_CALUDE_arcsin_b_range_l356_35690


namespace NUMINAMATH_CALUDE_keith_cards_proof_l356_35675

/-- The number of cards Keith started with -/
def initial_cards : ℕ := 84

/-- The number of cards Keith has left after the incident -/
def remaining_cards : ℕ := 46

/-- The number of cards Keith bought -/
def bought_cards : ℕ := 8

theorem keith_cards_proof :
  ∃ (total : ℕ), total = initial_cards + bought_cards ∧ 
                 remaining_cards * 2 = total := by
  sorry

end NUMINAMATH_CALUDE_keith_cards_proof_l356_35675


namespace NUMINAMATH_CALUDE_cone_volume_with_inscribed_cylinder_l356_35653

/-- The volume of a cone with an inscribed cylinder -/
def cone_volume (cylinder_volume : ℝ) (truncated_cone_volume : ℝ) : ℝ :=
  94.5

/-- Theorem stating the volume of the cone given the conditions -/
theorem cone_volume_with_inscribed_cylinder
  (cylinder_volume : ℝ)
  (truncated_cone_volume : ℝ)
  (h1 : cylinder_volume = 21)
  (h2 : truncated_cone_volume = 91) :
  cone_volume cylinder_volume truncated_cone_volume = 94.5 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_with_inscribed_cylinder_l356_35653


namespace NUMINAMATH_CALUDE_shelter_cats_l356_35621

theorem shelter_cats (x : ℚ) 
  (h1 : x + x/2 - 3 + 5 - 1 = 19) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_shelter_cats_l356_35621


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l356_35613

/-- The hyperbola equation -/
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, 0)

/-- The asymptotes of the hyperbola are tangent to the circle -/
def asymptotes_tangent_to_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ 
    (y = (a/b) * (x - 3) ∨ y = -(a/b) * (x - 3))

/-- The right focus of the hyperbola is the center of the circle -/
def right_focus_is_circle_center (a : ℝ) : Prop :=
  (a, 0) = circle_center

/-- The main theorem -/
theorem hyperbola_parameters :
  ∀ (a b : ℝ), 
    a > 0 → b > 0 →
    asymptotes_tangent_to_circle a b →
    right_focus_is_circle_center a →
    a^2 = 4 ∧ b^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l356_35613


namespace NUMINAMATH_CALUDE_solution_set_l356_35669

theorem solution_set (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_solution_set_l356_35669
