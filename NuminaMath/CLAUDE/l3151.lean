import Mathlib

namespace min_printers_purchase_l3151_315131

theorem min_printers_purchase (cost1 cost2 : ℕ) (h1 : cost1 = 350) (h2 : cost2 = 200) :
  ∃ (x y : ℕ), 
    x * cost1 = y * cost2 ∧ 
    x + y = 11 ∧
    ∀ (a b : ℕ), a * cost1 = b * cost2 → a + b ≥ 11 :=
by sorry

end min_printers_purchase_l3151_315131


namespace vertex_coordinates_l3151_315185

def f (x : ℝ) := (x - 1)^2 - 2

theorem vertex_coordinates :
  ∃ (x y : ℝ), (x = 1 ∧ y = -2) ∧
  ∀ (t : ℝ), f t ≥ f x :=
by sorry

end vertex_coordinates_l3151_315185


namespace max_min_product_l3151_315161

theorem max_min_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a + b + c = 8) (h5 : a * b + b * c + c * a = 16) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 16 / 9 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 16 / 9 :=
sorry

end max_min_product_l3151_315161


namespace expand_and_simplify_l3151_315122

theorem expand_and_simplify (x : ℝ) : (x + 3) * (4 * x - 8) + x^2 = 5 * x^2 + 4 * x - 24 := by
  sorry

end expand_and_simplify_l3151_315122


namespace obtuse_triangle_x_range_l3151_315158

/-- A triangle with sides a, b, and c is obtuse if and only if 
    the square of the longest side is greater than the sum of 
    squares of the other two sides. -/
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ a^2 + b^2 < c^2) ∨
  (a ≤ c ∧ c ≤ b ∧ a^2 + c^2 < b^2) ∨
  (b ≤ a ∧ a ≤ c ∧ b^2 + a^2 < c^2) ∨
  (b ≤ c ∧ c ≤ a ∧ b^2 + c^2 < a^2) ∨
  (c ≤ a ∧ a ≤ b ∧ c^2 + a^2 < b^2) ∨
  (c ≤ b ∧ b ≤ a ∧ c^2 + b^2 < a^2)

theorem obtuse_triangle_x_range :
  ∀ x : ℝ, is_obtuse_triangle x (x + 1) (x + 2) → 1 < x ∧ x < 3 :=
by sorry

end obtuse_triangle_x_range_l3151_315158


namespace revenue_not_increased_l3151_315134

/-- The revenue function for the current year -/
def revenue (x : ℝ) : ℝ := 4*x^3 - 20*x^2 + 33*x - 17

/-- The previous year's revenue -/
def previous_revenue : ℝ := 20

theorem revenue_not_increased (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : 
  revenue x ≤ previous_revenue := by
  sorry

#check revenue_not_increased

end revenue_not_increased_l3151_315134


namespace bucket_fill_time_l3151_315140

/-- The time taken to fill a bucket completely, given that two-thirds of it is filled in 90 seconds at a constant rate. -/
theorem bucket_fill_time (fill_rate : ℝ) (h1 : fill_rate > 0) : 
  (2 / 3 : ℝ) / fill_rate = 90 → 1 / fill_rate = 135 := by sorry

end bucket_fill_time_l3151_315140


namespace triangle_centroid_incenter_relation_l3151_315168

open Real

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define functions to calculate centroid and incenter
def centroid (t : Triangle) : ℝ × ℝ := sorry

def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to calculate squared distance between two points
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_centroid_incenter_relation :
  ∃ k : ℝ, ∀ t : Triangle, ∀ P : ℝ × ℝ,
    let G := centroid t
    let I := incenter t
    dist_squared P t.A + dist_squared P t.B + dist_squared P t.C + dist_squared P I =
    k * (dist_squared P G + dist_squared G t.A + dist_squared G t.B + dist_squared G t.C + dist_squared G I) :=
by sorry

end triangle_centroid_incenter_relation_l3151_315168


namespace same_color_socks_probability_l3151_315104

def total_red_socks : ℕ := 12
def total_blue_socks : ℕ := 10

theorem same_color_socks_probability :
  let total_socks := total_red_socks + total_blue_socks
  let same_color_combinations := (total_red_socks.choose 2) + (total_blue_socks.choose 2)
  let total_combinations := total_socks.choose 2
  (same_color_combinations : ℚ) / total_combinations = 37 / 77 := by
  sorry

end same_color_socks_probability_l3151_315104


namespace intersection_implies_z_value_l3151_315199

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, 2, z * i}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem intersection_implies_z_value (z : ℂ) : 
  M z ∩ N = {4} → z = -4 * i :=
by
  sorry

end intersection_implies_z_value_l3151_315199


namespace function_properties_l3151_315189

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_not_constant : ∃ x y, f x ≠ f y)
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_symmetry : ∀ x, f (x + 1) = f (1 - x)) :
  is_even_function f ∧ is_periodic_function f 2 :=
sorry

end function_properties_l3151_315189


namespace green_shirt_pairs_l3151_315187

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ)
  (h1 : red_students = 63)
  (h2 : green_students = 69)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 66)
  (h5 : red_red_pairs = 25)
  (h6 : total_students = 2 * total_pairs) :
  2 * red_red_pairs + (red_students - 2 * red_red_pairs) + 2 * ((green_students - (red_students - 2 * red_red_pairs)) / 2) = total_students :=
by sorry

end green_shirt_pairs_l3151_315187


namespace reconstruct_diagonals_l3151_315138

/-- Represents a convex polygon with labeled vertices -/
structure LabeledPolygon where
  vertices : Finset ℕ
  labels : vertices → ℕ

/-- Represents a set of non-intersecting diagonals in a polygon -/
def Diagonals (p : LabeledPolygon) := Finset (Finset ℕ)

/-- Checks if a set of diagonals divides a polygon into triangles -/
def divides_into_triangles (p : LabeledPolygon) (d : Diagonals p) : Prop := sorry

/-- Checks if a set of diagonals matches the vertex labels -/
def matches_labels (p : LabeledPolygon) (d : Diagonals p) : Prop := sorry

/-- Main theorem: For any labeled convex polygon, there exists a unique set of diagonals
    that divides it into triangles and matches the labels -/
theorem reconstruct_diagonals (p : LabeledPolygon) : 
  ∃! d : Diagonals p, divides_into_triangles p d ∧ matches_labels p d := by sorry

end reconstruct_diagonals_l3151_315138


namespace problem_solution_l3151_315100

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 24^(2/7) := by
sorry

end problem_solution_l3151_315100


namespace left_shoe_probability_l3151_315129

/-- The probability of randomly picking a left shoe from a shoe cabinet with 3 pairs of shoes is 1/2. -/
theorem left_shoe_probability (num_pairs : ℕ) (h : num_pairs = 3) :
  (num_pairs : ℚ) / (2 * num_pairs : ℚ) = 1 / 2 := by
  sorry

end left_shoe_probability_l3151_315129


namespace inscribed_rectangle_area_coefficient_l3151_315162

/-- Triangle ABC with inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  /-- Side lengths of triangle ABC --/
  AB : ℝ
  BC : ℝ
  CA : ℝ
  /-- Coefficient of x in the area formula --/
  a : ℝ
  /-- Coefficient of x^2 in the area formula --/
  b : ℝ
  /-- The area of rectangle PQRS is given by a * x - b * x^2 --/
  area_formula : ∀ x, 0 ≤ x → x ≤ BC → 0 ≤ a * x - b * x^2

/-- The main theorem --/
theorem inscribed_rectangle_area_coefficient
  (t : TriangleWithRectangle)
  (h1 : t.AB = 13)
  (h2 : t.BC = 24)
  (h3 : t.CA = 15) :
  t.b = 13 / 48 :=
sorry

end inscribed_rectangle_area_coefficient_l3151_315162


namespace bottles_per_day_l3151_315159

def total_bottles : ℕ := 355
def total_days : ℕ := 71

theorem bottles_per_day : 
  total_bottles / total_days = 5 := by sorry

end bottles_per_day_l3151_315159


namespace min_value_fraction_sum_l3151_315169

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 1 / a + 2 / b ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 2 / y = 8) :=
by sorry

end min_value_fraction_sum_l3151_315169


namespace opposite_of_negative_three_l3151_315128

theorem opposite_of_negative_three : -((-3) : ℤ) = 3 := by sorry

end opposite_of_negative_three_l3151_315128


namespace total_spent_equals_79_09_l3151_315137

def shorts_price : Float := 15.00
def jacket_price : Float := 14.82
def shirt_price : Float := 12.51
def shoes_price : Float := 21.67
def hat_price : Float := 8.75
def belt_price : Float := 6.34

def total_spent : Float := shorts_price + jacket_price + shirt_price + shoes_price + hat_price + belt_price

theorem total_spent_equals_79_09 : total_spent = 79.09 := by
  sorry

end total_spent_equals_79_09_l3151_315137


namespace greatest_five_digit_multiple_l3151_315166

theorem greatest_five_digit_multiple : ∃ n : ℕ, 
  n ≤ 99999 ∧ 
  n ≥ 10000 ∧
  n % 9 = 0 ∧ 
  n % 6 = 0 ∧ 
  n % 2 = 0 ∧
  ∀ m : ℕ, m ≤ 99999 ∧ m ≥ 10000 ∧ m % 9 = 0 ∧ m % 6 = 0 ∧ m % 2 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

end greatest_five_digit_multiple_l3151_315166


namespace fraction_of_number_l3151_315120

theorem fraction_of_number : (7 : ℚ) / 25 * 89473 = 25052.44 := by
  sorry

end fraction_of_number_l3151_315120


namespace warehouse_storage_problem_l3151_315103

/-- Represents the warehouse storage problem -/
theorem warehouse_storage_problem 
  (second_floor_space : ℝ) 
  (h1 : second_floor_space > 0) 
  (h2 : 3 * second_floor_space - (1/4) * second_floor_space = 55000) : 
  (1/4) * second_floor_space = 5000 := by
  sorry

end warehouse_storage_problem_l3151_315103


namespace set_operations_and_subsets_l3151_315186

def U : Finset ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12}
def A : Finset ℕ := {6, 8, 10, 12}
def B : Finset ℕ := {1, 6, 8}

theorem set_operations_and_subsets :
  (A ∪ B = {1, 6, 8, 10, 12}) ∧
  (U \ A = {4, 5, 7, 9, 11}) ∧
  (Finset.powerset (A ∩ B)).card = 4 := by sorry

end set_operations_and_subsets_l3151_315186


namespace product_45_360_trailing_zeros_l3151_315136

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The product of 45 and 360 has 2 trailing zeros -/
theorem product_45_360_trailing_zeros : trailingZeros (45 * 360) = 2 := by
  sorry

end product_45_360_trailing_zeros_l3151_315136


namespace complex_magnitude_product_l3151_315163

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 4 * Complex.I)) = 10 * Real.sqrt 21 := by
  sorry

end complex_magnitude_product_l3151_315163


namespace mod_seven_equality_l3151_315154

theorem mod_seven_equality : (47 ^ 2049 - 18 ^ 2049) % 7 = 4 := by
  sorry

end mod_seven_equality_l3151_315154


namespace sqrt_six_div_sqrt_three_eq_sqrt_two_l3151_315150

theorem sqrt_six_div_sqrt_three_eq_sqrt_two :
  Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2 := by
  sorry

end sqrt_six_div_sqrt_three_eq_sqrt_two_l3151_315150


namespace problem_solution_l3151_315175

theorem problem_solution (a b : ℝ) (h1 : b - a = -6) (h2 : a * b = 7) :
  a^2 * b - a * b^2 = -42 := by
  sorry

end problem_solution_l3151_315175


namespace find_x_l3151_315171

theorem find_x : ∃ x : ℚ, (3 * x - 6 + 4) / 7 = 15 ∧ x = 107 / 3 := by
  sorry

end find_x_l3151_315171


namespace equation_three_solutions_l3151_315141

theorem equation_three_solutions :
  let f : ℝ → ℝ := λ x => (x^2 - 4) * (x^2 - 1) - (x^2 + 3*x + 2) * (x^2 - 8*x + 7)
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x, f x = 0 → x ∈ s :=
by sorry

end equation_three_solutions_l3151_315141


namespace jasmine_laps_l3151_315157

/-- Calculates the total number of laps swum in a given number of weeks -/
def total_laps (laps_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * num_weeks

/-- Proves that Jasmine swims 300 laps in five weeks -/
theorem jasmine_laps : total_laps 12 5 5 = 300 := by
  sorry

end jasmine_laps_l3151_315157


namespace savings_calculation_l3151_315155

/-- Calculates a person's savings given their income and the ratio of income to expenditure. -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that for a person with an income of 36000 and an income to expenditure ratio of 9:8, their savings are 4000. -/
theorem savings_calculation :
  calculate_savings 36000 9 8 = 4000 := by
  sorry

end savings_calculation_l3151_315155


namespace sequence_matches_given_terms_sequence_satisfies_conditions_l3151_315114

/-- The sequence a_n is defined as 10^n + n -/
def a (n : ℕ) : ℕ := 10^n + n

/-- The first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  a 1 = 11 ∧ a 2 = 102 ∧ a 3 = 1003 ∧ a 4 = 10004 := by
  sorry

/-- The sequence a_n satisfies the given first four terms -/
theorem sequence_satisfies_conditions : ∃ f : ℕ → ℕ, 
  (f 1 = 11 ∧ f 2 = 102 ∧ f 3 = 1003 ∧ f 4 = 10004) ∧
  (∀ n : ℕ, f n = a n) := by
  sorry

end sequence_matches_given_terms_sequence_satisfies_conditions_l3151_315114


namespace a_minus_b_greater_than_one_l3151_315111

theorem a_minus_b_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hf : ∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    r₁^3 + a*r₁^2 + 2*b*r₁ - 1 = 0 ∧
    r₂^3 + a*r₂^2 + 2*b*r₂ - 1 = 0 ∧
    r₃^3 + a*r₃^2 + 2*b*r₃ - 1 = 0)
  (hg : ∀ x : ℝ, 2*x^2 + 2*b*x + a ≠ 0) :
  a - b > 1 := by
sorry

end a_minus_b_greater_than_one_l3151_315111


namespace sum_geq_three_cube_root_three_l3151_315191

theorem sum_geq_three_cube_root_three (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^3 + b^3 + c^3 = a^2 * b^2 * c^2) : 
  a + b + c ≥ 3 * (3 : ℝ)^(1/3) := by
  sorry

end sum_geq_three_cube_root_three_l3151_315191


namespace eg_length_l3151_315172

/-- A quadrilateral with specific side lengths -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℕ)

/-- The theorem stating the length of EG in the specific quadrilateral -/
theorem eg_length (q : Quadrilateral) 
  (h1 : q.EF = 7)
  (h2 : q.FG = 13)
  (h3 : q.GH = 7)
  (h4 : q.HE = 11) :
  q.EG = 13 := by
  sorry


end eg_length_l3151_315172


namespace smallest_part_of_proportional_division_l3151_315110

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 150)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prop : ∃ (k : ℚ), a = 3 * k ∧ b = 5 * k ∧ c = (7/2) * k)
  (h_sum : a + b + c = total) : 
  min a (min b c) = 900 / 23 := by
sorry

end smallest_part_of_proportional_division_l3151_315110


namespace sams_water_buckets_l3151_315101

-- Define the initial amount of water
def initial_water : Real := 1

-- Define the additional amount of water
def additional_water : Real := 8.8

-- Define the total amount of water
def total_water : Real := initial_water + additional_water

-- Theorem statement
theorem sams_water_buckets : total_water = 9.8 := by
  sorry

end sams_water_buckets_l3151_315101


namespace simple_random_sampling_problem_l3151_315127

/-- Prove that in a simple random sampling where 13 individuals are drawn one by one
    from a group of n individuals (n > 13), if the probability for each of the remaining
    individuals to be drawn on the second draw is 1/3, then n = 37. -/
theorem simple_random_sampling_problem (n : ℕ) (h1 : n > 13) :
  (12 : ℝ) / (n - 1 : ℝ) = (1 : ℝ) / 3 → n = 37 := by
  sorry

end simple_random_sampling_problem_l3151_315127


namespace expected_deliveries_l3151_315144

theorem expected_deliveries (packages_yesterday : ℕ) (success_rate : ℚ) :
  packages_yesterday = 80 →
  success_rate = 90 / 100 →
  (packages_yesterday * 2 : ℚ) * success_rate = 144 :=
by
  sorry

end expected_deliveries_l3151_315144


namespace triangle_area_ratio_l3151_315179

/-- Given two triangles AEF and AFC sharing a common vertex A, 
    where EF:FC = 3:5 and the area of AEF is 27, 
    prove that the area of AFC is 45. -/
theorem triangle_area_ratio (EF FC : ℝ) (area_AEF area_AFC : ℝ) : 
  EF / FC = 3 / 5 → 
  area_AEF = 27 → 
  area_AEF / area_AFC = EF / FC → 
  area_AFC = 45 := by
sorry

end triangle_area_ratio_l3151_315179


namespace roger_coins_count_l3151_315126

/-- Calculates the total number of coins given the number of piles of quarters,
    piles of dimes, and coins per pile. -/
def totalCoins (quarterPiles dimePiles coinsPerPile : ℕ) : ℕ :=
  (quarterPiles + dimePiles) * coinsPerPile

/-- Theorem stating that with 3 piles of quarters, 3 piles of dimes,
    and 7 coins per pile, the total number of coins is 42. -/
theorem roger_coins_count :
  totalCoins 3 3 7 = 42 := by
  sorry

end roger_coins_count_l3151_315126


namespace inscribed_ngon_existence_l3151_315132

/-- An n-gon inscribed in a circle with sides parallel to n given lines -/
structure InscribedNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ
  lines : Fin n → ℝ × ℝ → Prop

/-- The measure of the angle at a vertex of the n-gon -/
def angle (ngon : InscribedNGon n) (i : Fin n) : ℝ := sorry

/-- The sum of odd-indexed angles -/
def sumOddAngles (ngon : InscribedNGon n) : ℝ := sorry

/-- The sum of even-indexed angles -/
def sumEvenAngles (ngon : InscribedNGon n) : ℝ := sorry

/-- The existence of an inscribed n-gon with sides parallel to given lines -/
def existsInscribedNGon (n : ℕ) (center : ℝ × ℝ) (radius : ℝ) (lines : Fin n → ℝ × ℝ → Prop) : Prop := sorry

theorem inscribed_ngon_existence (n : ℕ) (center : ℝ × ℝ) (radius : ℝ) (lines : Fin n → ℝ × ℝ → Prop) :
  (n % 2 = 1 ∧ existsInscribedNGon n center radius lines) ∨
  (n % 2 = 0 ∧ (existsInscribedNGon n center radius lines ↔
    ∃ (ngon : InscribedNGon n), sumOddAngles ngon = sumEvenAngles ngon)) :=
sorry

end inscribed_ngon_existence_l3151_315132


namespace tangent_ratio_bounds_l3151_315177

noncomputable def f (x : ℝ) : ℝ := |Real.exp x - 1|

theorem tangent_ratio_bounds (x₁ x₂ : ℝ) (h₁ : x₁ < 0) (h₂ : x₂ > 0) :
  let A := (x₁, f x₁)
  let B := (x₂, f x₂)
  let M := (0, (1 - Real.exp x₁) + x₁ * Real.exp x₁)
  let N := (0, (Real.exp x₂ - 1) - x₂ * Real.exp x₂)
  let tangent_slope_A := -Real.exp x₁
  let tangent_slope_B := Real.exp x₂
  tangent_slope_A * tangent_slope_B = -1 →
  let AM := Real.sqrt ((x₁ - 0)^2 + (f x₁ - M.2)^2)
  let BN := Real.sqrt ((x₂ - 0)^2 + (f x₂ - N.2)^2)
  0 < AM / BN ∧ AM / BN < 1 :=
by sorry

end tangent_ratio_bounds_l3151_315177


namespace outer_wheel_speed_double_radii_difference_outer_wheel_circumference_l3151_315124

/-- Represents a car driving in a circle -/
structure CircularDrivingCar where
  inner_radius : ℝ
  outer_radius : ℝ
  wheel_distance : ℝ

/-- The properties of the car as described in the problem -/
def problem_car : CircularDrivingCar :=
  { inner_radius := 1.5,  -- This value is derived from the solution, not given directly
    outer_radius := 3,    -- This value is derived from the solution, not given directly
    wheel_distance := 1.5 }

/-- The theorem stating the relationship between the outer and inner wheel speeds -/
theorem outer_wheel_speed_double (car : CircularDrivingCar) :
  car.outer_radius = 2 * car.inner_radius :=
sorry

/-- The theorem stating the relationship between the radii and the wheel distance -/
theorem radii_difference (car : CircularDrivingCar) :
  car.outer_radius - car.inner_radius = car.wheel_distance :=
sorry

/-- The main theorem to prove -/
theorem outer_wheel_circumference (car : CircularDrivingCar) :
  2 * π * car.outer_radius = π * 6 :=
sorry

end outer_wheel_speed_double_radii_difference_outer_wheel_circumference_l3151_315124


namespace eddie_pies_l3151_315173

/-- The number of pies Eddie's sister can bake per day -/
def sister_pies : ℕ := 6

/-- The number of pies Eddie's mother can bake per day -/
def mother_pies : ℕ := 8

/-- The total number of pies they can bake in 7 days -/
def total_pies : ℕ := 119

/-- The number of days they bake pies -/
def days : ℕ := 7

/-- Eddie can bake 3 pies a day -/
theorem eddie_pies : ∃ (eddie_pies : ℕ), 
  eddie_pies = 3 ∧ 
  days * (eddie_pies + sister_pies + mother_pies) = total_pies := by
  sorry

end eddie_pies_l3151_315173


namespace decrease_interval_of_f_shifted_l3151_315108

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the interval of decrease for f(x+1)
def interval_of_decrease : Set ℝ := Set.Ioo 0 2

-- Theorem statement
theorem decrease_interval_of_f_shifted :
  ∀ x ∈ interval_of_decrease, f' (x + 1) < 0 :=
sorry

end decrease_interval_of_f_shifted_l3151_315108


namespace inequality_proof_l3151_315151

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end inequality_proof_l3151_315151


namespace rectangular_room_shorter_side_l3151_315130

theorem rectangular_room_shorter_side
  (perimeter : ℝ)
  (area : ℝ)
  (h_perimeter : perimeter = 42)
  (h_area : area = 108)
  (h_rect : ∃ (length width : ℝ), length > 0 ∧ width > 0 ∧
            2 * (length + width) = perimeter ∧
            length * width = area) :
  ∃ (shorter_side : ℝ), shorter_side = 9 ∧
    ∃ (longer_side : ℝ), longer_side > shorter_side ∧
      2 * (shorter_side + longer_side) = perimeter ∧
      shorter_side * longer_side = area :=
by sorry

end rectangular_room_shorter_side_l3151_315130


namespace f_range_characterization_l3151_315188

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_range_characterization :
  ∀ x : ℝ, f x ≥ 1 ↔ ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi :=
sorry

end f_range_characterization_l3151_315188


namespace duck_cow_problem_l3151_315176

theorem duck_cow_problem (D C : ℕ) : 
  (2 * D + 4 * C = 2 * (D + C) + 40) → C = 20 := by
  sorry

end duck_cow_problem_l3151_315176


namespace f_is_direct_proportion_l3151_315107

/-- A function f : ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = 2x -/
def f : ℝ → ℝ := fun x ↦ 2 * x

/-- Theorem: The function f(x) = 2x is a direct proportion function -/
theorem f_is_direct_proportion : IsDirectProportion f := by
  sorry

end f_is_direct_proportion_l3151_315107


namespace difference_of_squares_factorization_l3151_315167

theorem difference_of_squares_factorization (a : ℝ) : a^2 - 6 = (a + Real.sqrt 6) * (a - Real.sqrt 6) := by
  sorry

end difference_of_squares_factorization_l3151_315167


namespace ways_1800_eq_partitions_300_l3151_315133

/-- The number of ways to write a positive integer as a sum of ones, twos, and threes, ignoring order -/
def numWays (n : ℕ) : ℕ := sorry

/-- The number of ways to partition a positive integer into four non-negative integer parts -/
def numPartitions4 (n : ℕ) : ℕ := sorry

/-- Theorem stating the equivalence between the two counting problems for n = 1800 -/
theorem ways_1800_eq_partitions_300 : numWays 1800 = numPartitions4 300 := by sorry

end ways_1800_eq_partitions_300_l3151_315133


namespace ghost_paths_count_l3151_315123

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways a ghost can enter and exit the mansion -/
def ghost_paths : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that there are exactly 56 ways for a ghost to enter and exit the mansion -/
theorem ghost_paths_count : ghost_paths = 56 := by
  sorry

end ghost_paths_count_l3151_315123


namespace no_function_satisfies_condition_l3151_315190

theorem no_function_satisfies_condition : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f x * f y - f (x * y) = 2 * x + 2 * y := by
  sorry

end no_function_satisfies_condition_l3151_315190


namespace intersection_point_of_parabolas_l3151_315102

-- Define the parabolas
def C₁ (x y : ℝ) : Prop :=
  (x - (Real.sqrt 2 - 1))^2 = 2 * (y - 1)^2

def C₂ (a b x y : ℝ) : Prop :=
  x^2 - a*y + x + 2*b = 0

-- Define the perpendicular tangents condition
def perpendicularTangents (a : ℝ) (x y : ℝ) : Prop :=
  (2*y - 2) * (2*y - a) = -1

-- Theorem statement
theorem intersection_point_of_parabolas
  (a b : ℝ) (h : ∃ x y, C₁ x y ∧ C₂ a b x y ∧ perpendicularTangents a x y) :
  ∃ x y, C₁ x y ∧ C₂ a b x y ∧ x = Real.sqrt 2 - 1/2 ∧ y = 1 := by
  sorry

end intersection_point_of_parabolas_l3151_315102


namespace angle_sum_properties_l3151_315112

/-- Given two obtuse angles α and β whose terminal sides intersect the unit circle at points
    with x-coordinates -√2/10 and -2√5/5 respectively, prove that tan(α+β) = -5/3 and α+2β = 9π/4 -/
theorem angle_sum_properties (α β : Real) (hα : α > π/2) (hβ : β > π/2)
  (hA : Real.cos α = -Real.sqrt 2 / 10)
  (hB : Real.cos β = -2 * Real.sqrt 5 / 5) :
  Real.tan (α + β) = -5/3 ∧ α + 2*β = 9*π/4 := by
  sorry

end angle_sum_properties_l3151_315112


namespace two_x_plus_y_equals_seven_l3151_315183

theorem two_x_plus_y_equals_seven 
  (h1 : (x + y) / 3 = 1.6666666666666667)
  (h2 : x + 2 * y = 8) : 
  2 * x + y = 7 := by
  sorry

end two_x_plus_y_equals_seven_l3151_315183


namespace two_color_theorem_l3151_315198

-- Define a type for regions in the plane
def Region : Type := ℕ

-- Define a type for colors
inductive Color
| Red : Color
| Blue : Color

-- Define a function type for coloring the map
def Coloring := Region → Color

-- Define a relation for adjacent regions
def Adjacent : Region → Region → Prop := sorry

-- Define a property for a valid coloring
def ValidColoring (coloring : Coloring) : Prop :=
  ∀ r1 r2 : Region, Adjacent r1 r2 → coloring r1 ≠ coloring r2

-- Define a type for the map configuration
structure MapConfiguration :=
  (num_lines : ℕ)
  (num_circles : ℕ)

-- State the theorem
theorem two_color_theorem :
  ∀ (config : MapConfiguration), ∃ (coloring : Coloring), ValidColoring coloring :=
sorry

end two_color_theorem_l3151_315198


namespace same_heads_probability_l3151_315156

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^keiko_pennies * 2^ephraim_pennies

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 3

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem same_heads_probability :
  probability = 3 / 32 := by sorry

end same_heads_probability_l3151_315156


namespace circle_radius_from_area_circumference_ratio_l3151_315160

/-- Given a circle with area M and circumference N, if M/N = 15, then the radius is 30 -/
theorem circle_radius_from_area_circumference_ratio (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end circle_radius_from_area_circumference_ratio_l3151_315160


namespace friends_team_assignment_l3151_315119

theorem friends_team_assignment :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_assign := num_teams ^ num_friends
  ways_to_assign = 65536 := by
  sorry

end friends_team_assignment_l3151_315119


namespace intersection_of_M_and_N_l3151_315125

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_of_M_and_N_l3151_315125


namespace exists_function_sum_one_not_exists_function_diff_one_l3151_315164

-- Part a
theorem exists_function_sum_one : 
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 1) ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f a ≠ f b) :=
sorry

-- Part b
theorem not_exists_function_diff_one : 
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, f (Real.sin x) - f (Real.cos x) = 1) ∧ 
  (∃ a b : ℝ, a ≠ b ∧ f a ≠ f b) :=
sorry

end exists_function_sum_one_not_exists_function_diff_one_l3151_315164


namespace quadratic_inequality_l3151_315184

-- Define the quadratic function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) 
  (h : ∀ x, f b c (3 + x) = f b c (3 - x)) : 
  f b c 4 < f b c 1 ∧ f b c 1 < f b c (-1) := by
  sorry

end quadratic_inequality_l3151_315184


namespace only_extend_line_segment_valid_l3151_315106

-- Define the geometric objects
structure StraightLine
structure LineSegment where
  endpoint1 : Point
  endpoint2 : Point
structure Ray where
  endpoint : Point

-- Define the statements
inductive GeometricStatement
  | ExtendStraightLine
  | ExtendLineSegment
  | ExtendRay
  | DrawStraightLineWithLength
  | CutOffSegmentOnRay

-- Define a predicate for valid operations
def is_valid_operation (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.ExtendLineSegment => true
  | _ => false

-- Theorem statement
theorem only_extend_line_segment_valid :
  ∀ s : GeometricStatement, is_valid_operation s ↔ s = GeometricStatement.ExtendLineSegment := by
  sorry

end only_extend_line_segment_valid_l3151_315106


namespace inverse_inequality_l3151_315115

theorem inverse_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l3151_315115


namespace total_shells_formula_l3151_315197

/-- The total number of shells picked up in two hours -/
def total_shells (x : ℚ) : ℚ :=
  x + (x + 32)

/-- Theorem stating that the total number of shells is equal to 2x + 32 -/
theorem total_shells_formula (x : ℚ) : total_shells x = 2 * x + 32 := by
  sorry

end total_shells_formula_l3151_315197


namespace sophist_statements_l3151_315149

/-- Represents the types of inhabitants on the Isle of Logic. -/
inductive Inhabitant
  | Knight
  | Liar
  | Sophist

/-- The total number of knights on the island. -/
def num_knights : ℕ := 40

/-- The total number of liars on the island. -/
def num_liars : ℕ := 25

/-- A function that determines if a statement about the number of knights is valid for a sophist. -/
def valid_knight_statement (n : ℕ) : Prop :=
  n ≠ num_knights ∧ n = num_knights

/-- A function that determines if a statement about the number of liars is valid for a sophist. -/
def valid_liar_statement (n : ℕ) : Prop :=
  n ≠ num_liars ∧ n = num_liars + 1

/-- The main theorem stating that the only valid sophist statements are 40 knights and 26 liars. -/
theorem sophist_statements :
  (∃! k : ℕ, valid_knight_statement k) ∧
  (∃! l : ℕ, valid_liar_statement l) ∧
  valid_knight_statement 40 ∧
  valid_liar_statement 26 := by
  sorry

end sophist_statements_l3151_315149


namespace some_number_value_l3151_315105

theorem some_number_value (n : ℝ) : 9 / (1 + n / 0.5) = 1 → n = 4 := by
  sorry

end some_number_value_l3151_315105


namespace new_year_day_frequency_new_year_day_sunday_more_frequent_l3151_315121

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the day of the week for a given date -/
noncomputable def getDayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Function to count occurrences of a specific day of the week as New Year's Day over 400 years -/
noncomputable def countNewYearDay (day : DayOfWeek) (startYear : ℕ) : ℕ :=
  sorry

/-- Theorem stating that New Year's Day falls on Sunday more frequently than on Monday over a 400-year cycle -/
theorem new_year_day_frequency (startYear : ℕ) :
  countNewYearDay DayOfWeek.Sunday startYear > countNewYearDay DayOfWeek.Monday startYear :=
by
  sorry

/-- Given condition: 23 October 1948 was a Saturday -/
axiom oct_23_1948_saturday : getDayOfWeek ⟨1948, 10, 23⟩ = DayOfWeek.Saturday

/-- Theorem to prove the frequency of New Year's Day on Sunday vs Monday -/
theorem new_year_day_sunday_more_frequent :
  ∃ startYear, countNewYearDay DayOfWeek.Sunday startYear > countNewYearDay DayOfWeek.Monday startYear :=
by
  sorry

end new_year_day_frequency_new_year_day_sunday_more_frequent_l3151_315121


namespace unique_five_numbers_l3151_315152

def triple_sums (a b c d e : ℝ) : List ℝ :=
  [a + b + c, a + b + d, a + b + e, a + c + d, a + c + e, a + d + e,
   b + c + d, b + c + e, b + d + e, c + d + e]

theorem unique_five_numbers :
  ∃! (a b c d e : ℝ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧
    triple_sums a b c d e = [3, 4, 6, 7, 9, 10, 11, 14, 15, 17] :=
by
  sorry

end unique_five_numbers_l3151_315152


namespace students_liking_sports_l3151_315145

theorem students_liking_sports (B C : Finset Nat) : 
  (B.card = 9) → 
  (C.card = 8) → 
  ((B ∩ C).card = 6) → 
  ((B ∪ C).card = 11) := by
sorry

end students_liking_sports_l3151_315145


namespace parabola_directrix_l3151_315170

/-- Given a parabola y = -4x^2 + 8x - 1, its directrix is y = 49/16 -/
theorem parabola_directrix (x y : ℝ) :
  y = -4 * x^2 + 8 * x - 1 →
  ∃ (k : ℝ), k = 49/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = -4 * x₀^2 + 8 * x₀ - 1 →
    ∃ (x₁ : ℝ), (x₀ - x₁)^2 + (y₀ - k)^2 = (y₀ - k)^2 / 4) :=
by sorry


end parabola_directrix_l3151_315170


namespace least_five_digit_congruent_to_7_mod_21_l3151_315178

theorem least_five_digit_congruent_to_7_mod_21 :
  ∀ n : ℕ, 
    n ≥ 10000 ∧ n ≤ 99999 ∧ n % 21 = 7 → n ≥ 10003 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_21_l3151_315178


namespace min_difference_of_roots_l3151_315135

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x - t else 2 * (x + 1) - t

theorem min_difference_of_roots (t : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f t x₁ = 0 ∧ f t x₂ = 0 →
  ∃ min_diff : ℝ, (∀ y₁ y₂ : ℝ, y₁ > y₂ → f t y₁ = 0 → f t y₂ = 0 → y₁ - y₂ ≥ min_diff) ∧
               min_diff = 15/16 :=
sorry

end min_difference_of_roots_l3151_315135


namespace solution_set_equivalence_l3151_315109

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_equivalence 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by sorry

end solution_set_equivalence_l3151_315109


namespace andrey_gifts_l3151_315174

theorem andrey_gifts :
  ∀ (n : ℕ) (a : ℕ),
    n > 2 →
    n * (n - 2) = a * (n - 1) + 16 →
    n = 18 :=
by sorry

end andrey_gifts_l3151_315174


namespace at_least_two_black_balls_count_l3151_315139

def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 4
def balls_drawn : ℕ := 4

theorem at_least_two_black_balls_count :
  (Finset.sum (Finset.range 3) (λ i => 
    Nat.choose total_black_balls (i + 2) * Nat.choose total_white_balls (balls_drawn - (i + 2)))) = 115 := by
  sorry

end at_least_two_black_balls_count_l3151_315139


namespace complement_of_union_l3151_315146

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4}

theorem complement_of_union :
  (Aᶜ ∩ Bᶜ) ∩ U = {3, 5} :=
sorry

end complement_of_union_l3151_315146


namespace expected_boy_girl_pairs_l3151_315182

theorem expected_boy_girl_pairs (n_boys n_girls : ℕ) (h_boys : n_boys = 8) (h_girls : n_girls = 12) :
  let total := n_boys + n_girls
  let inner_boys := n_boys - 2
  let inner_pairs := total - 1
  let inner_prob := (inner_boys * n_girls) / ((inner_boys + n_girls) * (inner_boys + n_girls - 1))
  let end_prob := n_girls / total
  (inner_pairs - 2) * (2 * inner_prob) + 2 * end_prob = 144/17 + 24/19 := by
  sorry

end expected_boy_girl_pairs_l3151_315182


namespace sum_of_ages_is_32_l3151_315143

/-- Viggo's age when his brother was 2 years old -/
def viggos_initial_age : ℕ := 2 * 2 + 10

/-- The current age of Viggo's younger brother -/
def brothers_current_age : ℕ := 10

/-- The number of years that have passed since the initial condition -/
def years_passed : ℕ := brothers_current_age - 2

/-- Viggo's current age -/
def viggos_current_age : ℕ := viggos_initial_age + years_passed

/-- The sum of Viggo's and his younger brother's current ages -/
def sum_of_ages : ℕ := viggos_current_age + brothers_current_age

theorem sum_of_ages_is_32 : sum_of_ages = 32 := by
  sorry

end sum_of_ages_is_32_l3151_315143


namespace other_sides_equations_l3151_315194

/-- An isosceles right triangle with one leg on the line 2x - y = 0 and hypotenuse midpoint (4, 2) -/
structure IsoscelesRightTriangle where
  /-- The line containing one leg of the triangle -/
  leg_line : Set (ℝ × ℝ)
  /-- The midpoint of the hypotenuse -/
  hypotenuse_midpoint : ℝ × ℝ
  /-- The triangle is isosceles and right-angled -/
  is_isosceles_right : Bool
  /-- The leg line equation is 2x - y = 0 -/
  leg_line_eq : leg_line = {(x, y) : ℝ × ℝ | 2 * x - y = 0}
  /-- The hypotenuse midpoint is (4, 2) -/
  midpoint_coords : hypotenuse_midpoint = (4, 2)

/-- The theorem stating the equations of the other two sides -/
theorem other_sides_equations (t : IsoscelesRightTriangle) :
  ∃ (side1 side2 : Set (ℝ × ℝ)),
    side1 = {(x, y) : ℝ × ℝ | x + 2 * y - 2 = 0} ∧
    side2 = {(x, y) : ℝ × ℝ | x + 2 * y - 14 = 0} :=
  sorry

end other_sides_equations_l3151_315194


namespace max_n_for_T_sum_less_than_2023_l3151_315118

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def geometric_sequence (n : ℕ) : ℕ := 2^(n - 1)

def c_sequence (n : ℕ) : ℕ := arithmetic_sequence (geometric_sequence n)

def T_sum (n : ℕ) : ℕ := 2^(n + 1) - n - 2

theorem max_n_for_T_sum_less_than_2023 :
  ∀ n : ℕ, T_sum n < 2023 → n ≤ 9 ∧ T_sum 9 < 2023 ∧ T_sum 10 ≥ 2023 :=
by sorry

end max_n_for_T_sum_less_than_2023_l3151_315118


namespace find_g_of_x_l3151_315116

theorem find_g_of_x (x : ℝ) (g : ℝ → ℝ) : 
  (2 * x^5 + 4 * x^3 - 3 * x + 5 + g x = 3 * x^4 + 7 * x^2 - 2 * x - 4) → 
  (g x = -2 * x^5 + 3 * x^4 - 4 * x^3 + 7 * x^2 - x - 9) := by
sorry

end find_g_of_x_l3151_315116


namespace two_numbers_difference_l3151_315153

theorem two_numbers_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 20000) 
  (h3 : a % 9 = 0 ∨ b % 9 = 0) (h4 : 2 * a + 6 = b) : b - a = 6670 := by
  sorry

end two_numbers_difference_l3151_315153


namespace students_spend_two_dollars_l3151_315196

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- The total amount spent by the students in dollars -/
def total_spent : ℚ := (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100

theorem students_spend_two_dollars : total_spent = 2 := by
  sorry

end students_spend_two_dollars_l3151_315196


namespace iv_bottle_capacity_l3151_315147

/-- Calculates the total capacity of an IV bottle given initial volume, flow rate, and elapsed time. -/
def totalCapacity (initialVolume : ℝ) (flowRate : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialVolume + flowRate * elapsedTime

/-- Theorem stating that given the specified conditions, the total capacity of the IV bottle is 150 mL. -/
theorem iv_bottle_capacity :
  let initialVolume : ℝ := 100
  let flowRate : ℝ := 2.5
  let elapsedTime : ℝ := 12
  totalCapacity initialVolume flowRate elapsedTime = 150 := by
  sorry

#eval totalCapacity 100 2.5 12

end iv_bottle_capacity_l3151_315147


namespace box_dimensions_l3151_315181

theorem box_dimensions (x y z : ℝ) 
  (volume : x * y * z = 160)
  (face1 : y * z = 80)
  (face2 : x * z = 40)
  (face3 : x * y = 32) :
  x = 4 ∧ y = 8 ∧ z = 10 := by
  sorry

end box_dimensions_l3151_315181


namespace inscribed_circle_radius_l3151_315195

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The angle at the base of the trapezoid in radians -/
  baseAngle : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The area is positive -/
  area_pos : 0 < area
  /-- The base angle is 30° (π/6 radians) -/
  angle_is_30deg : baseAngle = Real.pi / 6
  /-- The radius is positive -/
  radius_pos : 0 < radius

/-- Theorem: The radius of the inscribed circle in an isosceles trapezoid -/
theorem inscribed_circle_radius (t : IsoscelesTrapezoid) : 
  t.radius = Real.sqrt (2 * t.area) / 4 := by
  sorry

end inscribed_circle_radius_l3151_315195


namespace pascal_triangle_row17_element5_l3151_315180

theorem pascal_triangle_row17_element5 : Nat.choose 17 4 = 2380 := by
  sorry

end pascal_triangle_row17_element5_l3151_315180


namespace difference_of_squares_l3151_315117

theorem difference_of_squares (x y : ℕ+) 
  (sum_eq : x + y = 18)
  (product_eq : x * y = 80) :
  x^2 - y^2 = 36 := by
  sorry

end difference_of_squares_l3151_315117


namespace geometric_sequence_product_l3151_315192

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 4 → a 2 * a 6 = 16 := by
  sorry

end geometric_sequence_product_l3151_315192


namespace sqrt_calculation_and_algebraic_expression_l3151_315193

theorem sqrt_calculation_and_algebraic_expression :
  (∃ x : ℝ, x^2 = 18) ∧ 
  (∃ y : ℝ, y^2 = 8) ∧ 
  (∃ z : ℝ, z^2 = 1/2) ∧
  (∃ a : ℝ, a^2 = 5) ∧
  (∃ b : ℝ, b^2 = 3) ∧
  (∃ c : ℝ, c^2 = 12) ∧
  (∃ d : ℝ, d^2 = 27) →
  (∃ x y z : ℝ, x^2 = 18 ∧ y^2 = 8 ∧ z^2 = 1/2 ∧ x - y + z = 3 * Real.sqrt 2 / 2) ∧
  (∃ a b c d : ℝ, a^2 = 5 ∧ b^2 = 3 ∧ c^2 = 12 ∧ d^2 = 27 ∧
    (2*a - 1) * (1 + 2*a) + b * (c - d) = 16) := by
  sorry

end sqrt_calculation_and_algebraic_expression_l3151_315193


namespace complete_collection_probability_l3151_315113

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def needed_stickers : ℕ := 6
def collected_stickers : ℕ := 8

theorem complete_collection_probability :
  (Nat.choose needed_stickers needed_stickers * Nat.choose collected_stickers (selected_stickers - needed_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
  sorry

end complete_collection_probability_l3151_315113


namespace value_of_expression_l3151_315165

theorem value_of_expression (x y : ℝ) (h : x - 2*y = 3) : x - 2*y + 4 = 7 := by sorry

end value_of_expression_l3151_315165


namespace three_from_eight_l3151_315142

theorem three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end three_from_eight_l3151_315142


namespace polygon_angles_theorem_l3151_315148

theorem polygon_angles_theorem (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 2 * 360) →
  n = 6 := by
sorry

end polygon_angles_theorem_l3151_315148
