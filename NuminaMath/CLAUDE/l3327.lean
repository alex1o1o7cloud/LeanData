import Mathlib

namespace sqrt_square_eq_abs_l3327_332752

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt ((x + 3)^2) = |x + 3| := by sorry

end sqrt_square_eq_abs_l3327_332752


namespace dining_bill_calculation_l3327_332784

theorem dining_bill_calculation (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_bill = 198)
  (h2 : tax_rate = 0.1)
  (h3 : tip_rate = 0.2) :
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total_bill ∧ 
    food_price = 150 := by
  sorry

end dining_bill_calculation_l3327_332784


namespace colors_wash_time_l3327_332741

/-- Represents the time in minutes for a laundry load in the washing machine and dryer -/
structure LaundryTime where
  wash : ℕ
  dry : ℕ

/-- The total time for all three loads of laundry -/
def totalTime : ℕ := 344

/-- The laundry time for the whites -/
def whites : LaundryTime := { wash := 72, dry := 50 }

/-- The laundry time for the darks -/
def darks : LaundryTime := { wash := 58, dry := 65 }

/-- The drying time for the colors -/
def colorsDryTime : ℕ := 54

/-- Theorem stating that the washing time for the colors is 45 minutes -/
theorem colors_wash_time :
  ∃ (colorsWashTime : ℕ),
    colorsWashTime = totalTime - (whites.wash + whites.dry + darks.wash + darks.dry + colorsDryTime) ∧
    colorsWashTime = 45 := by
  sorry

end colors_wash_time_l3327_332741


namespace monochromatic_unit_area_triangle_exists_l3327_332705

/-- A color representing red, green, or blue -/
inductive Color
| Red
| Green
| Blue

/-- A point with integer coordinates on a plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring of points on a plane -/
def Coloring := Point → Color

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : Point) : ℚ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

theorem monochromatic_unit_area_triangle_exists (c : Coloring) :
  ∃ (p1 p2 p3 : Point), c p1 = c p2 ∧ c p2 = c p3 ∧ triangleArea p1 p2 p3 = 1 := by
  sorry

end monochromatic_unit_area_triangle_exists_l3327_332705


namespace max_value_of_S_l3327_332734

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 :=
by sorry

end max_value_of_S_l3327_332734


namespace min_cubes_to_remove_l3327_332703

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular block given its dimensions. -/
def blockVolume (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the side length of the largest cube that can fit within the block. -/
def largestCubeSide (d : BlockDimensions) : ℕ :=
  min d.length (min d.width d.height)

/-- Calculates the volume of the largest cube that can fit within the block. -/
def largestCubeVolume (d : BlockDimensions) : ℕ :=
  let side := largestCubeSide d
  side * side * side

/-- The main theorem stating the minimum number of cubes to remove. -/
theorem min_cubes_to_remove (d : BlockDimensions) 
    (h1 : d.length = 4) (h2 : d.width = 5) (h3 : d.height = 6) : 
    blockVolume d - largestCubeVolume d = 56 := by
  sorry

end min_cubes_to_remove_l3327_332703


namespace torn_pages_fine_l3327_332718

/-- Calculates the fine for tearing out pages from a book -/
def calculate_fine (start_page end_page : ℕ) (cost_per_sheet : ℕ) : ℕ :=
  let total_pages := end_page - start_page + 1
  let total_sheets := (total_pages + 1) / 2
  total_sheets * cost_per_sheet

/-- The fine for tearing out pages 15 to 30 is 128 yuan -/
theorem torn_pages_fine :
  calculate_fine 15 30 16 = 128 := by
  sorry

end torn_pages_fine_l3327_332718


namespace polar_to_cartesian_equivalence_l3327_332792

/-- Proves the equivalence between a polar equation and its Cartesian form --/
theorem polar_to_cartesian_equivalence (x y ρ θ : ℝ) :
  (ρ = -4 * Real.cos θ + Real.sin θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  x^2 + y^2 + 4*x - y = 0 :=
by sorry

end polar_to_cartesian_equivalence_l3327_332792


namespace domain_of_shifted_f_l3327_332706

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 2

-- Define the property that f is defined only on its domain
axiom f_defined_on_domain : ∀ x, x ∈ domain_f → f x ≠ 0

-- State the theorem
theorem domain_of_shifted_f :
  {x | f (x + 1) ≠ 0} = Set.Icc 0 1 :=
sorry

end domain_of_shifted_f_l3327_332706


namespace largest_similar_triangle_exists_l3327_332724

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the properties of the triangles
axiom similar_triangles (T1 T2 : Triangle) : Prop
axiom point_on_line (P Q R : Point) : Prop
axiom triangle_area (T : Triangle) : ℝ

-- Define the given triangles
variable (A B : Triangle)

-- Define the constructed triangle
variable (M : Triangle)

-- Define the conditions
variable (h1 : point_on_line (A.1) (M.2.1) (M.2.2))
variable (h2 : point_on_line (A.2.1) (A.1) (A.2.2))
variable (h3 : point_on_line (A.2.2) (A.1) (A.2.1))
variable (h4 : similar_triangles M B)

-- State the theorem
theorem largest_similar_triangle_exists :
  ∃ (M : Triangle), 
    point_on_line (A.1) (M.2.1) (M.2.2) ∧
    point_on_line (A.2.1) (A.1) (A.2.2) ∧
    point_on_line (A.2.2) (A.1) (A.2.1) ∧
    similar_triangles M B ∧
    ∀ (M' : Triangle), 
      (point_on_line (A.1) (M'.2.1) (M'.2.2) ∧
       point_on_line (A.2.1) (A.1) (A.2.2) ∧
       point_on_line (A.2.2) (A.1) (A.2.1) ∧
       similar_triangles M' B) →
      triangle_area M ≥ triangle_area M' :=
sorry

end largest_similar_triangle_exists_l3327_332724


namespace shaded_area_fraction_l3327_332776

theorem shaded_area_fraction (total_squares : ℕ) (shaded_squares : ℕ) :
  total_squares = 6 →
  shaded_squares = 2 →
  (shaded_squares : ℚ) / total_squares = 1 / 3 := by
sorry

end shaded_area_fraction_l3327_332776


namespace polynomial_root_equivalence_l3327_332723

theorem polynomial_root_equivalence : ∀ r : ℝ, 
  r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end polynomial_root_equivalence_l3327_332723


namespace x_difference_l3327_332764

theorem x_difference (x₁ x₂ : ℝ) : 
  ((x₁ + 3)^2 / (2*x₁ + 15) = 3) →
  ((x₂ + 3)^2 / (2*x₂ + 15) = 3) →
  x₁ ≠ x₂ →
  |x₁ - x₂| = 12 :=
by
  sorry

end x_difference_l3327_332764


namespace book_arrangement_theorem_l3327_332717

/-- Represents the number of ways to arrange books of two subjects -/
def arrange_books (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

/-- Theorem: The number of ways to arrange 3 math books and 3 Chinese books
    on a shelf, such that no two books of the same subject are adjacent,
    is equal to 72 -/
theorem book_arrangement_theorem :
  arrange_books 3 = 72 := by
  sorry

#eval arrange_books 3  -- This should output 72

end book_arrangement_theorem_l3327_332717


namespace builders_hired_for_houses_l3327_332730

/-- The number of builders hired to build houses given specific conditions -/
def builders_hired (days_per_floor : ℕ) (builders_per_floor : ℕ) (pay_per_builder : ℕ) 
  (num_houses : ℕ) (floors_per_house : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_per_floor := days_per_floor * builders_per_floor * pay_per_builder
  let total_floors := num_houses * floors_per_house
  total_cost / cost_per_floor

/-- Theorem stating the number of builders hired under given conditions -/
theorem builders_hired_for_houses :
  builders_hired 30 3 100 5 6 270000 = 30 := by
  sorry

end builders_hired_for_houses_l3327_332730


namespace cyclic_inequality_l3327_332780

theorem cyclic_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

#check cyclic_inequality

end cyclic_inequality_l3327_332780


namespace mod_twelve_power_six_l3327_332712

theorem mod_twelve_power_six (n : ℕ) : 12^6 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 0 := by
  sorry

end mod_twelve_power_six_l3327_332712


namespace negation_of_proposition_l3327_332704

theorem negation_of_proposition (P : ℝ → Prop) :
  (∀ x : ℝ, x^2 + 1 > 1) ↔ ¬(∃ x₀ : ℝ, x₀^2 + 1 ≤ 1) := by
  sorry

end negation_of_proposition_l3327_332704


namespace average_student_height_l3327_332747

/-- The average height of all students given specific conditions -/
theorem average_student_height 
  (avg_female_height : ℝ) 
  (avg_male_height : ℝ) 
  (male_to_female_ratio : ℝ) 
  (h1 : avg_female_height = 170) 
  (h2 : avg_male_height = 182) 
  (h3 : male_to_female_ratio = 5) : 
  (male_to_female_ratio * avg_male_height + avg_female_height) / (male_to_female_ratio + 1) = 180 := by
  sorry

end average_student_height_l3327_332747


namespace B_equals_D_l3327_332772

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

-- Define set D
def D : Set ℝ := {y : ℝ | y ≥ 1}

-- Theorem stating that B and D are equal
theorem B_equals_D : B = D := by sorry

end B_equals_D_l3327_332772


namespace characterize_solutions_l3327_332781

/-- Given a system of equations with real parameters a, b, c and variables x, y, z,
    this theorem characterizes all possible solutions. -/
theorem characterize_solutions (a b c x y z : ℝ) :
  x^2 * y^2 + x^2 * z^2 = a * x * y * z ∧
  y^2 * z^2 + y^2 * x^2 = b * x * y * z ∧
  z^2 * x^2 + z^2 * y^2 = c * x * y * z →
  (∃ t : ℝ, (x = t ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = t ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = t)) ∨
  (∃ s : ℝ, s = (a + b + c) / 2 ∧
    ((x^2 = (s - b) * (s - c) ∧ y^2 = (s - a) * (s - c) ∧ z^2 = (s - a) * (s - b)) ∧
     (0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∨
     (0 < -a ∧ 0 < -b ∧ 0 < -c ∧ -a + -b > -c ∧ -a + -c > -b ∧ -b + -c > -a))) :=
by sorry

end characterize_solutions_l3327_332781


namespace not_perfect_square_l3327_332700

theorem not_perfect_square : 
  (∃ x : ℕ, 6^3024 = x^2) ∧
  (∀ y : ℕ, 7^3025 ≠ y^2) ∧
  (∃ z : ℕ, 8^3026 = z^2) ∧
  (∃ w : ℕ, 9^3027 = w^2) ∧
  (∃ v : ℕ, 10^3028 = v^2) := by
  sorry

end not_perfect_square_l3327_332700


namespace range_of_a_for_max_and_min_l3327_332737

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- The discriminant of the quadratic equation f'(x) = 0 -/
def discriminant (a : ℝ) : ℝ := 36*a^2 - 36*(a+2)

/-- The condition for f to have both maximum and minimum -/
def has_max_and_min (a : ℝ) : Prop := discriminant a > 0

theorem range_of_a_for_max_and_min :
  ∀ a : ℝ, has_max_and_min a ↔ (a < -1 ∨ a > 2) :=
by sorry

end range_of_a_for_max_and_min_l3327_332737


namespace wechat_group_size_l3327_332761

theorem wechat_group_size :
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) / 2 = 72 ∧ n = 9 := by
  sorry

end wechat_group_size_l3327_332761


namespace square_of_cube_third_smallest_prime_l3327_332746

def third_smallest_prime : Nat := 5

theorem square_of_cube_third_smallest_prime : 
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by
  sorry

end square_of_cube_third_smallest_prime_l3327_332746


namespace roses_age_l3327_332759

theorem roses_age (rose_age mother_age : ℕ) : 
  rose_age = mother_age / 3 →
  rose_age + mother_age = 100 →
  rose_age = 25 := by
sorry

end roses_age_l3327_332759


namespace sams_ribbon_length_l3327_332714

/-- The total length of a ribbon cut into equal pieces -/
def total_ribbon_length (piece_length : ℕ) (num_pieces : ℕ) : ℕ :=
  piece_length * num_pieces

/-- Theorem: The total length of Sam's ribbon is 3723 cm -/
theorem sams_ribbon_length : 
  total_ribbon_length 73 51 = 3723 := by
  sorry

end sams_ribbon_length_l3327_332714


namespace quadratic_inequality_solutions_l3327_332755

theorem quadratic_inequality_solutions (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end quadratic_inequality_solutions_l3327_332755


namespace condition_relationship_l3327_332751

theorem condition_relationship :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end condition_relationship_l3327_332751


namespace simplify_expression_l3327_332728

theorem simplify_expression : 1 + 1 / (1 + Real.sqrt 5) + 1 / (1 - Real.sqrt 5) = 1 / 2 := by
  sorry

end simplify_expression_l3327_332728


namespace larrys_coincidence_l3327_332702

theorem larrys_coincidence (a b c d e : ℝ) 
  (ha : a = 5) (hb : b = 3) (hc : c = 6) (hd : d = 4) :
  a - b + c + d - e = a - (b - (c + (d - e))) :=
by sorry

end larrys_coincidence_l3327_332702


namespace jenna_reading_schedule_l3327_332711

/-- Represents Jenna's reading schedule for September --/
structure ReadingSchedule where
  total_days : Nat
  total_pages : Nat
  busy_days : Nat
  special_day_pages : Nat

/-- Calculates the number of pages Jenna needs to read per day on regular reading days --/
def pages_per_day (schedule : ReadingSchedule) : Nat :=
  let regular_reading_days := schedule.total_days - schedule.busy_days - 1
  let regular_pages := schedule.total_pages - schedule.special_day_pages
  regular_pages / regular_reading_days

/-- Theorem stating that Jenna needs to read 20 pages per day on regular reading days --/
theorem jenna_reading_schedule :
  let schedule := ReadingSchedule.mk 30 600 4 100
  pages_per_day schedule = 20 := by
  sorry

end jenna_reading_schedule_l3327_332711


namespace percentage_calculation_l3327_332777

theorem percentage_calculation (x : ℝ) : 
  (0.08 : ℝ) * x = (0.6 : ℝ) * ((0.3 : ℝ) * x) - (0.1 : ℝ) * x := by
  sorry

end percentage_calculation_l3327_332777


namespace floor_product_equals_48_l3327_332767

theorem floor_product_equals_48 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 48 ↔ x ∈ Set.Icc 8 (49/6) :=
sorry

end floor_product_equals_48_l3327_332767


namespace ice_cream_distribution_l3327_332715

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143) (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 := by
  sorry

end ice_cream_distribution_l3327_332715


namespace vacuum_tube_alignment_l3327_332790

theorem vacuum_tube_alignment (f : Fin 7 → Fin 7) (h : Function.Bijective f) :
  ∃ x : Fin 7, f x = x := by
  sorry

end vacuum_tube_alignment_l3327_332790


namespace square_containing_circle_l3327_332774

/-- The area and perimeter of the smallest square containing a circle --/
theorem square_containing_circle (r : ℝ) (h : r = 6) :
  ∃ (area perimeter : ℝ),
    area = (2 * r) ^ 2 ∧
    perimeter = 4 * (2 * r) ∧
    area = 144 ∧
    perimeter = 48 := by
  sorry

end square_containing_circle_l3327_332774


namespace percent_of_self_l3327_332778

theorem percent_of_self (y : ℝ) (h1 : y > 0) (h2 : y * (y / 100) = 9) : y = 30 := by
  sorry

end percent_of_self_l3327_332778


namespace quadratic_factorization_l3327_332709

theorem quadratic_factorization (p : ℕ+) :
  (∃ a b : ℤ, ∀ x : ℤ, x^2 - 5*x + p.val = (x - a) * (x - b)) →
  p.val = 4 ∨ p.val = 6 := by
sorry

end quadratic_factorization_l3327_332709


namespace prob_second_day_restaurant_A_l3327_332749

/-- Represents the restaurants in the Olympic Village -/
inductive Restaurant
| A  -- Smart restaurant
| B  -- Manual restaurant

/-- The probability of choosing restaurant A on the second day -/
def prob_second_day_A (first_day_choice : Restaurant) : ℝ :=
  match first_day_choice with
  | Restaurant.A => 0.6
  | Restaurant.B => 0.5

/-- The probability of choosing a restaurant on the first day -/
def prob_first_day (r : Restaurant) : ℝ := 0.5

/-- The theorem stating the probability of going to restaurant A on the second day -/
theorem prob_second_day_restaurant_A :
  (prob_first_day Restaurant.A * prob_second_day_A Restaurant.A +
   prob_first_day Restaurant.B * prob_second_day_A Restaurant.B) = 0.55 := by
  sorry


end prob_second_day_restaurant_A_l3327_332749


namespace milk_packet_cost_l3327_332768

theorem milk_packet_cost (total_packets : Nat) (remaining_packets : Nat) 
  (avg_price_all : ℚ) (avg_price_remaining : ℚ) :
  total_packets = 10 →
  remaining_packets = 7 →
  avg_price_all = 25 →
  avg_price_remaining = 20 →
  (total_packets * avg_price_all - remaining_packets * avg_price_remaining : ℚ) = 110 := by
  sorry

end milk_packet_cost_l3327_332768


namespace solve_equation_l3327_332701

theorem solve_equation (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) 
  (h2 : y = 3*x) (h3 : x ≠ 3) : x = 3/2 := by
  sorry

end solve_equation_l3327_332701


namespace union_equals_A_l3327_332783

def A : Set ℤ := {-1, 0, 1}
def B (a : ℤ) : Set ℤ := {a, a^2}

theorem union_equals_A (a : ℤ) : A ∪ B a = A ↔ a = -1 := by
  sorry

end union_equals_A_l3327_332783


namespace intersection_of_S_and_T_l3327_332789

-- Define the sets S and T
def S : Set ℝ := {0, 1, 2, 3}
def T : Set ℝ := {x | |x - 1| ≤ 1}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {0, 1, 2} := by
  sorry

end intersection_of_S_and_T_l3327_332789


namespace cubic_roots_sum_series_l3327_332740

def cubic_polynomial (x : ℝ) : ℝ := 30 * x^3 - 50 * x^2 + 22 * x - 1

theorem cubic_roots_sum_series : 
  ∃ (a b c : ℝ),
    (∀ x : ℝ, cubic_polynomial x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
    (∑' n : ℕ, (a^n + b^n + c^n)) = 12 := by
  sorry

end cubic_roots_sum_series_l3327_332740


namespace square_polygon_area_l3327_332729

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon represented by a list of its vertices -/
def Polygon := List Point

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (p : Polygon) : ℝ := sorry

/-- The specific polygon described in the problem -/
def squarePolygon : Polygon := [
  { x := 0, y := 0 },
  { x := 6, y := 0 },
  { x := 6, y := 6 },
  { x := 0, y := 6 }
]

/-- Theorem stating that the area of the given polygon is 36 square units -/
theorem square_polygon_area :
  polygonArea squarePolygon = 36 := by sorry

end square_polygon_area_l3327_332729


namespace polynomial_identity_sum_of_squares_l3327_332799

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 512 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := by
sorry

end polynomial_identity_sum_of_squares_l3327_332799


namespace function_value_at_negative_two_l3327_332762

theorem function_value_at_negative_two :
  Real.sqrt (4 * (-2) + 9) = 1 := by
  sorry

end function_value_at_negative_two_l3327_332762


namespace ratio_sum_problem_l3327_332721

theorem ratio_sum_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  x / y = 3 / 4 → x + y + 100 = 500 → y = 1600 / 7 := by
  sorry

end ratio_sum_problem_l3327_332721


namespace high_school_sampling_l3327_332716

/-- Represents a stratified sampling scenario in a high school -/
structure StratifiedSampling where
  total_students : ℕ
  freshmen : ℕ
  sampled_freshmen : ℕ

/-- Calculates the total number of students to be sampled in a stratified sampling scenario -/
def total_sampled (s : StratifiedSampling) : ℚ :=
  (s.total_students : ℚ) * s.sampled_freshmen / s.freshmen

/-- Theorem stating that for the given high school scenario, the total number of students
    to be sampled is 80 -/
theorem high_school_sampling :
  let s : StratifiedSampling := { total_students := 2400, freshmen := 600, sampled_freshmen := 20 }
  total_sampled s = 80 := by
  sorry


end high_school_sampling_l3327_332716


namespace min_value_theorem_min_value_achieved_l3327_332748

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  x^4 * y^3 * z^2 ≥ 1/9^9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 9 ∧
  x^4 * y^3 * z^2 < 1/9^9 + ε :=
by sorry

end min_value_theorem_min_value_achieved_l3327_332748


namespace last_s_replacement_l3327_332787

/-- Represents the rules of the cryptographic code --/
structure CryptoRules where
  firstShift : ℕ
  vowels : List Char
  vowelSequence : List ℕ

/-- Counts the occurrences of a character in a string --/
def countOccurrences (c : Char) (s : String) : ℕ := sorry

/-- Calculates the triangular number for a given n --/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Applies the shift to a character based on the rules --/
def applyShift (c : Char) (count : ℕ) (rules : CryptoRules) : Char := sorry

/-- Main theorem to prove --/
theorem last_s_replacement (message : String) (rules : CryptoRules) :
  let lastSCount := countOccurrences 's' message
  let shift := triangularNumber lastSCount % 26
  let newPos := (('s'.toNat - 'a'.toNat + 1 + shift) % 26) + 'a'.toNat - 1
  Char.ofNat newPos = 'g' := by sorry

end last_s_replacement_l3327_332787


namespace calculator_cost_proof_l3327_332742

theorem calculator_cost_proof (basic scientific graphing : ℝ) 
  (h1 : scientific = 2 * basic)
  (h2 : graphing = 3 * scientific)
  (h3 : basic + scientific + graphing = 72) :
  basic = 8 := by
sorry

end calculator_cost_proof_l3327_332742


namespace quadratic_properties_l3327_332775

/-- A quadratic function y = ax² + bx + c with given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neq_0 : a ≠ 0
  h_point_neg1 : a * (-1)^2 + b * (-1) + c = -1
  h_point_0 : c = 3
  h_point_1 : a + b + c = 5
  h_point_3 : 9 * a + 3 * b + c = 3

/-- Theorem stating the properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.a * f.c < 0 ∧ f.a * 3^2 + (f.b - 1) * 3 + f.c = 0 := by
  sorry

end quadratic_properties_l3327_332775


namespace expansion_equality_l3327_332769

theorem expansion_equality (x y : ℝ) : 25 * (3 * x + 7 - 4 * y) = 75 * x + 175 - 100 * y := by
  sorry

end expansion_equality_l3327_332769


namespace inequality_range_l3327_332708

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end inequality_range_l3327_332708


namespace range_of_a_l3327_332770

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) → a ≤ 1 := by
  sorry

end range_of_a_l3327_332770


namespace prime_remainder_theorem_l3327_332765

theorem prime_remainder_theorem (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (p^3 + 17) % 24 = 0 ∨ (p^3 + 17) % 24 = 16 := by
  sorry

end prime_remainder_theorem_l3327_332765


namespace like_terms_imply_exponents_l3327_332732

/-- Two terms are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∃ (c₁ c₂ : ℚ), ∀ (x y : ℕ), term1 x y = c₁ * term2 x y

/-- The first term in our problem -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^m * y^2

/-- The second term in our problem -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := (2/3) * x * y^n

/-- If term1 and term2 are like terms, then m = 1 and n = 2 -/
theorem like_terms_imply_exponents (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m = 1 ∧ n = 2 := by
  sorry

end like_terms_imply_exponents_l3327_332732


namespace solution_set_part1_range_of_a_l3327_332736

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_l3327_332736


namespace least_possible_value_l3327_332763

theorem least_possible_value (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 5 → (∀ w, w - x ≥ 9 → w ≥ z) → x = 2 := by
  sorry

end least_possible_value_l3327_332763


namespace sum_of_first_50_terms_l3327_332794

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |> List.sum

theorem sum_of_first_50_terms (a : ℕ → ℕ) :
  a 1 = 7 ∧ (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = 20) →
  sequence_sum a 50 = 500 := by
sorry

end sum_of_first_50_terms_l3327_332794


namespace system_solutions_l3327_332758

theorem system_solutions :
  let eq1 := (fun (x y : ℝ) => x + 3*y + 3*x*y = -1)
  let eq2 := (fun (x y : ℝ) => x^2*y + 3*x*y^2 = -4)
  ∃! (s : Set (ℝ × ℝ)), s = {(-3, -1/3), (-1, -1), (-1, 4/3), (4, -1/3)} ∧
    ∀ (p : ℝ × ℝ), p ∈ s ↔ (eq1 p.1 p.2 ∧ eq2 p.1 p.2) := by
  sorry

end system_solutions_l3327_332758


namespace polynomial_coefficient_F_l3327_332757

def polynomial (x E F G H : ℤ) : ℤ := x^6 - 14*x^5 + E*x^4 + F*x^3 + G*x^2 + H*x + 36

def roots : List ℤ := [3, 3, 2, 2, 2, 2]

theorem polynomial_coefficient_F (E F G H : ℤ) :
  (∀ r ∈ roots, polynomial r E F G H = 0) →
  (List.sum roots = 14) →
  (∀ r ∈ roots, r > 0) →
  F = -248 := by sorry

end polynomial_coefficient_F_l3327_332757


namespace expression_equals_zero_l3327_332756

theorem expression_equals_zero (x : ℝ) : 
  (((x^2 - 2*x + 1) / (x^2 - 1)) / ((x - 1) / (x^2 + x))) - x = 0 :=
by sorry

end expression_equals_zero_l3327_332756


namespace four_digit_divisible_by_eleven_l3327_332719

theorem four_digit_divisible_by_eleven : 
  ∃ (B : ℕ), B < 10 ∧ (4000 + 100 * B + 10 * B + 2) % 11 = 0 :=
by
  -- The proof would go here
  sorry

end four_digit_divisible_by_eleven_l3327_332719


namespace geometric_sequence_monotonicity_l3327_332798

/-- An infinite geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- A sequence is monotonically increasing -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The first three terms of a sequence are in ascending order -/
def FirstThreeAscending (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_monotonicity (a : ℕ → ℝ) 
  (h : GeometricSequence a) : 
  MonotonicallyIncreasing a ↔ FirstThreeAscending a := by
  sorry

end geometric_sequence_monotonicity_l3327_332798


namespace line_ellipse_intersection_length_l3327_332750

theorem line_ellipse_intersection_length : ∃ A B : ℝ × ℝ,
  (∀ x y : ℝ, y = x - 1 → x^2 / 4 + y^2 / 3 = 1 → (x, y) = A ∨ (x, y) = B) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24 / 7 :=
sorry

end line_ellipse_intersection_length_l3327_332750


namespace gcd_18_30_l3327_332722

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l3327_332722


namespace tangent_circle_equation_l3327_332791

/-- A circle tangent to the parabola y^2 = 2x (y > 0), its axis, and the x-axis -/
structure TangentCircle where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the parabola y^2 = 2x (y > 0) -/
  tangent_to_parabola : center.2^2 = 2 * center.1
  /-- The circle is tangent to the x-axis -/
  tangent_to_x_axis : center.2 = radius
  /-- The circle's center is on the axis of the parabola (x-axis) -/
  on_parabola_axis : center.1 ≥ 0

/-- The equation of the circle is x^2 + y^2 - x - 2y + 1/4 = 0 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
  x^2 + y^2 - x - 2*y + 1/4 = 0 := by
  sorry


end tangent_circle_equation_l3327_332791


namespace shaded_area_circle_in_square_l3327_332773

/-- The area of the shaded region between a circle inscribed in a square,
    where the circle touches the midpoints of the square's sides. -/
theorem shaded_area_circle_in_square (side_length : ℝ) (h : side_length = 12) :
  side_length ^ 2 - π * (side_length / 2) ^ 2 = side_length ^ 2 - π * 36 :=
by sorry

end shaded_area_circle_in_square_l3327_332773


namespace kate_change_l3327_332720

def candy1_cost : ℚ := 54/100
def candy2_cost : ℚ := 35/100
def candy3_cost : ℚ := 68/100
def amount_paid : ℚ := 5

theorem kate_change : 
  amount_paid - (candy1_cost + candy2_cost + candy3_cost) = 343/100 := by
  sorry

end kate_change_l3327_332720


namespace equation_solution_l3327_332760

theorem equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 2)) = Real.sqrt 12 → y = 22 := by
sorry

end equation_solution_l3327_332760


namespace least_positive_angle_theta_l3327_332743

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0) → 
  (Real.cos (15 * Real.pi / 180) = Real.sin (45 * Real.pi / 180) + Real.sin θ) → 
  θ = 15 * Real.pi / 180 :=
by
  sorry

end least_positive_angle_theta_l3327_332743


namespace ship_passengers_l3327_332745

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 4 : ℚ) + (P / 9 : ℚ) + (P / 6 : ℚ) + 42 = P →
  P = 108 := by
  sorry

end ship_passengers_l3327_332745


namespace base_conversion_sum_l3327_332796

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : Nat) (c d : Nat) : Nat :=
  sorry

theorem base_conversion_sum :
  let c : Nat := 12
  let d : Nat := 13
  base8ToBase10 356 + base14ToBase10 4 c d = 1203 := by
  sorry

end base_conversion_sum_l3327_332796


namespace part_one_part_two_l3327_332797

-- Part 1
theorem part_one (f h : ℝ → ℝ) (m : ℝ) :
  (∀ x > 1, f x = x^2 - m * Real.log x) →
  (∀ x > 1, h x = x^2 - x) →
  (∀ x > 1, f x ≥ h x) →
  m ≤ Real.exp 1 :=
sorry

-- Part 2
theorem part_two (f h k : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^2 - 2 * Real.log x) →
  (∀ x, h x = x^2 - x + a) →
  (∀ x, k x = f x - h x) →
  (∃ x y, x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧ x < y ∧ k x = 0 ∧ k y = 0 ∧ 
    ∀ z ∈ Set.Icc 1 3, k z = 0 → (z = x ∨ z = y)) →
  2 - 2 * Real.log 2 < a ∧ a ≤ 3 - 2 * Real.log 3 :=
sorry

end part_one_part_two_l3327_332797


namespace milburg_grown_ups_l3327_332735

/-- The number of grown-ups in Milburg -/
def grown_ups (total_population children : ℕ) : ℕ :=
  total_population - children

/-- Proof that the number of grown-ups in Milburg is 5256 -/
theorem milburg_grown_ups :
  grown_ups 8243 2987 = 5256 := by
  sorry

end milburg_grown_ups_l3327_332735


namespace sum_with_rearrangement_not_all_nines_l3327_332725

def digit_sum (n : ℕ) : ℕ := sorry

def is_digit_rearrangement (n m : ℕ) : Prop :=
  digit_sum n = digit_sum m

def repeated_nines (k : ℕ) : ℕ := sorry

theorem sum_with_rearrangement_not_all_nines (n : ℕ) :
  ∀ m : ℕ, is_digit_rearrangement n m → n + m ≠ repeated_nines 125 := by sorry

end sum_with_rearrangement_not_all_nines_l3327_332725


namespace quadratic_value_l3327_332793

/-- A quadratic function with specific properties -/
def f (a b c : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c

/-- Theorem stating the properties of the quadratic function and its value at x = 5 -/
theorem quadratic_value (a b c : ℚ) :
  (f a b c (-2) = 10) →  -- Maximum value is 10 at x = -2
  ((2 * a * (-2) + b) = 0) →  -- Derivative is 0 at x = -2 (maximum condition)
  (f a b c 0 = -8) →  -- Passes through (0, -8)
  (f a b c 1 = 0) →  -- Passes through (1, 0)
  (f a b c 5 = -400/9) :=  -- Value at x = 5
by sorry

end quadratic_value_l3327_332793


namespace fence_cost_per_foot_l3327_332710

/-- The cost per foot of fencing a square plot -/
theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 25) 
  (h2 : total_cost = 1160) : 
  total_cost / (4 * Real.sqrt area) = 58 := by
sorry


end fence_cost_per_foot_l3327_332710


namespace probability_of_exact_successes_l3327_332779

def probability_of_success : ℚ := 1/3

def number_of_trials : ℕ := 3

def number_of_successes : ℕ := 2

theorem probability_of_exact_successes :
  (number_of_trials.choose number_of_successes) *
  probability_of_success ^ number_of_successes *
  (1 - probability_of_success) ^ (number_of_trials - number_of_successes) =
  2/9 :=
sorry

end probability_of_exact_successes_l3327_332779


namespace starting_lineup_combinations_l3327_332713

/-- The number of ways to choose k elements from a set of n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players on the team --/
def totalPlayers : ℕ := 15

/-- The number of All-Star players --/
def allStars : ℕ := 3

/-- The size of the starting lineup --/
def lineupSize : ℕ := 5

theorem starting_lineup_combinations :
  choose (totalPlayers - allStars) (lineupSize - allStars) = 66 := by
  sorry

end starting_lineup_combinations_l3327_332713


namespace problem_statement_l3327_332733

theorem problem_statement (a b c d k m : ℕ) 
  (h1 : d * a = b * c)
  (h2 : a + d = 2^k)
  (h3 : b + c = 2^m) :
  a = 1 := by sorry

end problem_statement_l3327_332733


namespace count_six_digit_numbers_with_at_least_two_zeros_l3327_332795

/-- The number of 6-digit numbers with at least two zeros -/
def six_digit_numbers_with_at_least_two_zeros : ℕ :=
  900000 - (9^6 + 5 * 9^5)

/-- Proof that the number of 6-digit numbers with at least two zeros is 73,314 -/
theorem count_six_digit_numbers_with_at_least_two_zeros :
  six_digit_numbers_with_at_least_two_zeros = 73314 := by
  sorry

#eval six_digit_numbers_with_at_least_two_zeros

end count_six_digit_numbers_with_at_least_two_zeros_l3327_332795


namespace personal_planners_count_l3327_332744

/-- The cost of a spiral notebook in dollars -/
def spiral_notebook_cost : ℝ := 15

/-- The cost of a personal planner in dollars -/
def personal_planner_cost : ℝ := 10

/-- The number of spiral notebooks bought -/
def num_spiral_notebooks : ℕ := 4

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.2

/-- The total discounted cost in dollars -/
def total_discounted_cost : ℝ := 112

/-- The number of personal planners bought -/
def num_personal_planners : ℕ := 8

theorem personal_planners_count :
  ∃ (x : ℕ),
    (1 - discount_rate) * (spiral_notebook_cost * num_spiral_notebooks + personal_planner_cost * x) = total_discounted_cost ∧
    x = num_personal_planners :=
by sorry

end personal_planners_count_l3327_332744


namespace parabola_equation_l3327_332782

/-- Represents a parabola in standard form -/
structure Parabola where
  p : ℝ
  axis : Bool  -- True for vertical axis, False for horizontal axis

/-- The hyperbola from the problem statement -/
def hyperbola : Set (ℝ × ℝ) :=
  {(x, y) | 16 * x^2 - 9 * y^2 = 144}

/-- The theorem statement -/
theorem parabola_equation (P : Parabola) :
  P.axis = true ∧  -- Vertical axis of symmetry
  (∀ (x y : ℝ), (x, y) ∈ hyperbola → x^2 = 9 ∧ y^2 = 16) ∧  -- Hyperbola properties
  (0, 0) ∈ hyperbola ∧  -- Vertex at origin
  (-3, 0) ∈ hyperbola ∧  -- Left vertex of hyperbola
  P.p = 6  -- Distance from vertex to directrix is 3
  →
  ∀ (x y : ℝ), y^2 = 2 * P.p * x ↔ y^2 = 12 * x := by
  sorry

end parabola_equation_l3327_332782


namespace arc_length_for_36_degree_angle_l3327_332788

theorem arc_length_for_36_degree_angle (d : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  d = 4 → θ_deg = 36 → l = (θ_deg * π / 180) * (d / 2) → l = 2 * π / 5 := by
  sorry

end arc_length_for_36_degree_angle_l3327_332788


namespace rainfall_problem_l3327_332754

/-- Rainfall problem -/
theorem rainfall_problem (day1 day2 day3 : ℝ) : 
  day1 = 4 →
  day2 = 5 * day1 →
  day3 = day1 + day2 - 6 →
  day3 = 18 := by
sorry

end rainfall_problem_l3327_332754


namespace condition_necessary_not_sufficient_l3327_332727

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a^2 + b^2 = 2*a*b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a^2 + b^2 ≠ 2*a*b) := by sorry

end condition_necessary_not_sufficient_l3327_332727


namespace conic_section_type_l3327_332785

/-- The equation √((x-2)² + y²) + √((x+2)² + y²) = 12 represents an ellipse -/
theorem conic_section_type : ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b),
  {(x, y) : ℝ × ℝ | Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 12} =
  {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / b^2) = 1} :=
by sorry

end conic_section_type_l3327_332785


namespace munchausen_polygon_exists_l3327_332726

/-- A polygon in a 2D plane --/
structure Polygon :=
  (vertices : Set (ℝ × ℝ))

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A line in a 2D plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Predicate to check if a point is inside a polygon --/
def IsInside (p : Point) (poly : Polygon) : Prop := sorry

/-- Predicate to check if a line passes through a point --/
def PassesThrough (l : Line) (p : Point) : Prop := sorry

/-- Predicate to check if a line divides a polygon into three parts --/
def DividesIntoThree (l : Line) (poly : Polygon) : Prop := sorry

/-- The main theorem --/
theorem munchausen_polygon_exists :
  ∃ (poly : Polygon) (p : Point),
    IsInside p poly ∧
    ∀ (l : Line), PassesThrough l p → DividesIntoThree l poly := by
  sorry

end munchausen_polygon_exists_l3327_332726


namespace remainder_seven_to_63_mod_8_l3327_332753

theorem remainder_seven_to_63_mod_8 :
  7^63 % 8 = 7 := by
  sorry

end remainder_seven_to_63_mod_8_l3327_332753


namespace find_d_l3327_332786

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 3

-- State the theorem
theorem find_d (c : ℝ) (d : ℝ) :
  (∀ x, f c (g c x) = 15 * x + d) → d = 18 := by
  sorry

end find_d_l3327_332786


namespace intersection_count_l3327_332771

/-- The number of distinct intersection points between a circle and a parabola -/
def numIntersectionPoints (r : ℝ) (a b : ℝ) : ℕ :=
  let circle (x y : ℝ) := x^2 + y^2 = r^2
  let parabola (x y : ℝ) := y = a * x^2 + b
  -- Definition of the function to count intersection points
  sorry

/-- Theorem stating that the number of intersection points is 3 for the given equations -/
theorem intersection_count : numIntersectionPoints 4 1 (-4) = 3 := by
  sorry

end intersection_count_l3327_332771


namespace inequality_proof_l3327_332739

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l3327_332739


namespace oranges_per_box_l3327_332707

/-- Given 56 oranges and 8 boxes, prove that the number of oranges per box is 7 -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 56) (h2 : num_boxes = 8) :
  total_oranges / num_boxes = 7 := by
  sorry

end oranges_per_box_l3327_332707


namespace sum_of_ages_l3327_332738

/-- Viggo's age when his brother was 2 years old -/
def viggos_age_when_brother_was_2 (brothers_age_when_2 : ℕ) : ℕ :=
  10 + 2 * brothers_age_when_2

/-- The current age of Viggo's younger brother -/
def brothers_current_age : ℕ := 10

/-- Viggo's current age -/
def viggos_current_age : ℕ :=
  brothers_current_age + (viggos_age_when_brother_was_2 2 - 2)

theorem sum_of_ages : 
  viggos_current_age + brothers_current_age = 32 := by
  sorry

end sum_of_ages_l3327_332738


namespace difference_at_negative_five_l3327_332766

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g (k : ℤ) (x : ℝ) : ℝ := x^3 - k * x - 10

-- State the theorem
theorem difference_at_negative_five (k : ℤ) : f (-5) - g k (-5) = -24 → k = 61 := by
  sorry

end difference_at_negative_five_l3327_332766


namespace probability_at_least_one_two_l3327_332731

def num_sides : ℕ := 8

def total_outcomes : ℕ := num_sides * num_sides

def outcomes_without_two : ℕ := (num_sides - 1) * (num_sides - 1)

def outcomes_with_at_least_one_two : ℕ := total_outcomes - outcomes_without_two

theorem probability_at_least_one_two :
  (outcomes_with_at_least_one_two : ℚ) / total_outcomes = 15 / 64 := by
  sorry

end probability_at_least_one_two_l3327_332731
