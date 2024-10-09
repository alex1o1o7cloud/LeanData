import Mathlib

namespace imaginary_part_of_z_l1731_173191

theorem imaginary_part_of_z {z : ℂ} (h : (1 + z) / I = 1 - z) : z.im = 1 := 
sorry

end imaginary_part_of_z_l1731_173191


namespace product_of_solutions_abs_eq_40_l1731_173172

theorem product_of_solutions_abs_eq_40 :
  (∃ x1 x2 : ℝ, (|3 * x1 - 5| = 40) ∧ (|3 * x2 - 5| = 40) ∧ ((x1 * x2) = -175)) :=
by
  sorry

end product_of_solutions_abs_eq_40_l1731_173172


namespace triangle_area_correct_l1731_173139

-- Define the points (vertices) of the triangle
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (8, -3)
def point3 : ℝ × ℝ := (2, 7)

-- Function to calculate the area of the triangle given three points (shoelace formula)
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.2 * C.1 - C.2 * A.1 - A.2 * B.1)

-- Prove that the area of the triangle with the given vertices is 18 square units
theorem triangle_area_correct : triangle_area point1 point2 point3 = 18 :=
by
  sorry

end triangle_area_correct_l1731_173139


namespace hyperbola_equation_l1731_173156

theorem hyperbola_equation (a b c : ℝ) (e : ℝ) 
  (h1 : e = (Real.sqrt 6) / 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : (c / a) = e)
  (h5 : (b * c) / (Real.sqrt (b^2 + a^2)) = 1) :
  (∃ a b : ℝ, (a = Real.sqrt 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 2) - y^2 = 1)) :=
by
  sorry

end hyperbola_equation_l1731_173156


namespace sufficient_but_not_necessary_condition_l1731_173159

def vectors_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

def vector_a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  x = 3 → vectors_parallel (vector_a x) (vector_b x) ∧
  vectors_parallel (vector_a 3) (vector_b 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1731_173159


namespace area_of_WXYZ_l1731_173162

structure Quadrilateral (α : Type _) :=
  (W : α) (X : α) (Y : α) (Z : α)
  (WZ ZW' WX XX' XY YY' YZ Z'W : ℝ)
  (area_WXYZ : ℝ)

theorem area_of_WXYZ' (WXYZ : Quadrilateral ℝ) 
  (h1 : WXYZ.WZ = 10) 
  (h2 : WXYZ.ZW' = 10)
  (h3 : WXYZ.WX = 6)
  (h4 : WXYZ.XX' = 6)
  (h5 : WXYZ.XY = 7)
  (h6 : WXYZ.YY' = 7)
  (h7 : WXYZ.YZ = 12)
  (h8 : WXYZ.Z'W = 12)
  (h9 : WXYZ.area_WXYZ = 15) : 
  ∃ area_WXZY' : ℝ, area_WXZY' = 45 :=
sorry

end area_of_WXYZ_l1731_173162


namespace total_legs_among_animals_l1731_173174

def legs (chickens sheep grasshoppers spiders : Nat) (legs_chicken legs_sheep legs_grasshopper legs_spider : Nat) : Nat :=
  (chickens * legs_chicken) + (sheep * legs_sheep) + (grasshoppers * legs_grasshopper) + (spiders * legs_spider)

theorem total_legs_among_animals :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let legs_chicken := 2
  let legs_sheep := 4
  let legs_grasshopper := 6
  let legs_spider := 8
  legs chickens sheep grasshoppers spiders legs_chicken legs_sheep legs_grasshopper legs_spider = 118 :=
by
  sorry

end total_legs_among_animals_l1731_173174


namespace diagonals_in_convex_polygon_l1731_173101

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end diagonals_in_convex_polygon_l1731_173101


namespace total_travel_time_correct_l1731_173104

-- Define the conditions
def highway_distance : ℕ := 100 -- miles
def mountain_distance : ℕ := 15 -- miles
def break_time : ℕ := 30 -- minutes
def time_on_mountain_road : ℕ := 45 -- minutes
def speed_ratio : ℕ := 5

-- Define the speeds using the given conditions.
def mountain_speed := mountain_distance / time_on_mountain_road -- miles per minute
def highway_speed := speed_ratio * mountain_speed -- miles per minute

-- Prove that total trip time equals 240 minutes
def total_trip_time : ℕ := 2 * (time_on_mountain_road + (highway_distance / highway_speed)) + break_time

theorem total_travel_time_correct : total_trip_time = 240 := 
by
  -- to be proved
  sorry

end total_travel_time_correct_l1731_173104


namespace acute_triangle_l1731_173196

-- Given the lengths of three line segments
def length1 : ℝ := 5
def length2 : ℝ := 6
def length3 : ℝ := 7

-- Conditions (C): The lengths of the three line segments
def triangle_inequality : Prop :=
  length1 + length2 > length3 ∧
  length1 + length3 > length2 ∧
  length2 + length3 > length1

-- Question (Q) and Answer (A): They form an acute triangle
theorem acute_triangle (h : triangle_inequality) : (length1^2 + length2^2 - length3^2 > 0) :=
by
  sorry

end acute_triangle_l1731_173196


namespace sandy_spent_percentage_l1731_173102

theorem sandy_spent_percentage (I R : ℝ) (hI : I = 200) (hR : R = 140) : 
  ((I - R) / I) * 100 = 30 :=
by
  sorry

end sandy_spent_percentage_l1731_173102


namespace median_length_YN_perimeter_triangle_XYZ_l1731_173120

-- Definitions for conditions
noncomputable def length_XY : ℝ := 5
noncomputable def length_XZ : ℝ := 12
noncomputable def is_right_angle_XYZ : Prop := true
noncomputable def midpoint_N : ℝ := length_XZ / 2

-- Theorem statement for the length of the median YN
theorem median_length_YN (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (13 / 2) = 6.5 := by
  sorry

-- Theorem statement for the perimeter of triangle XYZ
theorem perimeter_triangle_XYZ (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (XY + XZ + 13) = 30 := by
  sorry

end median_length_YN_perimeter_triangle_XYZ_l1731_173120


namespace simplify_expression_l1731_173135

theorem simplify_expression (a b : ℝ) (h : a ≠ b) : 
  ((a^3 - b^3) / (a * b)) - ((a * b^2 - b^3) / (a * b - a^3)) = (2 * a * (a - b)) / b :=
by
  sorry

end simplify_expression_l1731_173135


namespace CaitlinIs24_l1731_173134

-- Definition using the given conditions
def AuntAnnaAge : ℕ := 45
def BriannaAge : ℕ := (2 * AuntAnnaAge) / 3
def CaitlinAge : ℕ := BriannaAge - 6

-- Statement to be proved
theorem CaitlinIs24 : CaitlinAge = 24 :=
by
  sorry

end CaitlinIs24_l1731_173134


namespace combined_work_time_l1731_173108

def ajay_completion_time : ℕ := 8
def vijay_completion_time : ℕ := 24

theorem combined_work_time (T_A T_V : ℕ) (h1 : T_A = ajay_completion_time) (h2 : T_V = vijay_completion_time) :
  1 / (1 / (T_A : ℝ) + 1 / (T_V : ℝ)) = 6 :=
by
  rw [h1, h2]
  sorry

end combined_work_time_l1731_173108


namespace roundness_720_eq_7_l1731_173130

def roundness (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := n.factorization
    factors.sum (λ _ k => k)
  else 0

theorem roundness_720_eq_7 : roundness 720 = 7 := by
  sorry

end roundness_720_eq_7_l1731_173130


namespace find_natural_numbers_l1731_173190

theorem find_natural_numbers (n : ℕ) (x : ℕ) (y : ℕ) (hx : n = 10 * x + y) (hy : 10 * x + y = 14 * x) : n = 14 ∨ n = 28 :=
by
  sorry

end find_natural_numbers_l1731_173190


namespace original_mixture_acid_percent_l1731_173142

-- Definitions of conditions as per the original problem
def original_acid_percentage (a w : ℕ) (h1 : 4 * a = a + w + 2) (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : Prop :=
  (a * 100) / (a + w) = 100 / 3

-- Main theorem statement
theorem original_mixture_acid_percent (a w : ℕ) 
  (h1 : 4 * a = a + w + 2)
  (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : original_acid_percentage a w h1 h2 :=
sorry

end original_mixture_acid_percent_l1731_173142


namespace number_at_two_units_right_of_origin_l1731_173187

theorem number_at_two_units_right_of_origin : 
  ∀ (n : ℝ), (n = 0) →
  ∀ (x : ℝ), (x = n + 2) →
  x = 2 := 
by
  sorry

end number_at_two_units_right_of_origin_l1731_173187


namespace blue_lights_count_l1731_173188

def num_colored_lights := 350
def num_red_lights := 85
def num_yellow_lights := 112
def num_green_lights := 65
def num_blue_lights := num_colored_lights - (num_red_lights + num_yellow_lights + num_green_lights)

theorem blue_lights_count : num_blue_lights = 88 := by
  sorry

end blue_lights_count_l1731_173188


namespace bacteria_initial_count_l1731_173115

noncomputable def initial_bacteria (b_final : ℕ) (q : ℕ) : ℕ :=
  b_final / 4^q

theorem bacteria_initial_count : initial_bacteria 262144 4 = 1024 := by
  sorry

end bacteria_initial_count_l1731_173115


namespace isosceles_triangle_area_l1731_173178

theorem isosceles_triangle_area (x : ℤ) (h1 : x > 2) (h2 : x < 4) 
  (h3 : ∃ (a b : ℤ), a = x ∧ b = 8 - 2 * x ∧ a = b) :
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end isosceles_triangle_area_l1731_173178


namespace number_of_girls_at_camp_l1731_173124

theorem number_of_girls_at_camp (total_people : ℕ) (difference_boys_girls : ℕ) (nb_girls : ℕ) :
  total_people = 133 ∧ difference_boys_girls = 33 ∧ 2 * nb_girls + 33 = total_people → nb_girls = 50 := 
by
  intros
  sorry

end number_of_girls_at_camp_l1731_173124


namespace gray_area_correct_l1731_173131

-- Define the side lengths of the squares
variable (a b : ℝ)

-- Define the areas of the larger and smaller squares
def area_large_square : ℝ := (a + b) * (a + b)
def area_small_square : ℝ := a * a

-- Define the gray area
def gray_area : ℝ := area_large_square a b - area_small_square a

-- The proof statement
theorem gray_area_correct (a b : ℝ) : gray_area a b = 2 * a * b + b ^ 2 := by
  sorry

end gray_area_correct_l1731_173131


namespace matrix_determinant_equiv_l1731_173176

variable {x y z w : ℝ}

theorem matrix_determinant_equiv (h : x * w - y * z = 7) :
    (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by
    sorry

end matrix_determinant_equiv_l1731_173176


namespace arc_length_one_radian_l1731_173129

-- Given definitions and conditions
def radius : ℝ := 6370
def angle : ℝ := 1

-- Arc length formula
def arc_length (R α : ℝ) : ℝ := R * α

-- Statement to prove
theorem arc_length_one_radian : arc_length radius angle = 6370 := 
by 
  -- Proof goes here
  sorry

end arc_length_one_radian_l1731_173129


namespace perfect_square_trinomial_l1731_173165

theorem perfect_square_trinomial {m : ℝ} :
  (∃ (a : ℝ), x^2 + 2 * m * x + 9 = (x + a)^2) → (m = 3 ∨ m = -3) :=
sorry

end perfect_square_trinomial_l1731_173165


namespace bianca_points_l1731_173105

theorem bianca_points : 
  let a := 5; let b := 8; let c := 10;
  let A1 := 10; let P1 := 5; let G1 := 5;
  let A2 := 3; let P2 := 2; let G2 := 1;
  (A1 * a - A2 * a) + (P1 * b - P2 * b) + (G1 * c - G2 * c) = 99 := 
by
  sorry

end bianca_points_l1731_173105


namespace purely_periodic_denominator_l1731_173132

theorem purely_periodic_denominator :
  ∀ q : ℕ, (∃ a : ℕ, (∃ b : ℕ, q = 99 ∧ (a < 10) ∧ (b < 10) ∧ (∃ f : ℝ, f = ↑a / (10 * q) ∧ ∃ g : ℝ, g = (0.01 * ↑b / (10 * (99 / q))))) → q = 11 ∨ q = 33 ∨ q = 99) :=
by sorry

end purely_periodic_denominator_l1731_173132


namespace inequality_proof_l1731_173145

variable (x y z : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)
variable (hz : 0 < z)

theorem inequality_proof :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
sorry

end inequality_proof_l1731_173145


namespace right_triangle_area_l1731_173169

theorem right_triangle_area (a b c r : ℝ) (h1 : a = 15) (h2 : r = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_right : a ^ 2 + b ^ 2 = c ^ 2) (h_incircle : r = (a + b - c) / 2) : 
  1 / 2 * a * b = 60 :=
by
  sorry

end right_triangle_area_l1731_173169


namespace solve_equation_l1731_173137

theorem solve_equation : ∃ x : ℝ, (1 + x) / (2 - x) - 1 = 1 / (x - 2) ↔ x = 0 := 
by
  sorry

end solve_equation_l1731_173137


namespace value_of_a_l1731_173154

theorem value_of_a (a : ℝ) : (-2)^2 + 3*(-2) + a = 0 → a = 2 :=
by {
  sorry
}

end value_of_a_l1731_173154


namespace different_result_l1731_173155

theorem different_result :
  let A := -2 - (-3)
  let B := 2 - 3
  let C := -3 + 2
  let D := -3 - (-2)
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B = C ∧ B = D :=
by
  sorry

end different_result_l1731_173155


namespace range_of_m_l1731_173193

theorem range_of_m (m : ℝ) : 
  ((m + 3) * (m - 4) < 0) → 
  (m^2 - 4 * (m + 3) ≤ 0) → 
  (-2 ≤ m ∧ m < 4) :=
by 
  intro h1 h2
  sorry

end range_of_m_l1731_173193


namespace find_some_number_l1731_173112

theorem find_some_number :
  ∃ (some_number : ℝ), (0.0077 * 3.6) / (some_number * 0.1 * 0.007) = 990.0000000000001 ∧ some_number = 0.04 :=
  sorry

end find_some_number_l1731_173112


namespace find_number_l1731_173164

theorem find_number (x : ℝ) (h: 9999 * x = 4690910862): x = 469.1 :=
by
  sorry

end find_number_l1731_173164


namespace find_s_l1731_173183

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end find_s_l1731_173183


namespace problem1_problem2_l1731_173140

-- Problem 1: Simplify the calculation: 6.9^2 + 6.2 * 6.9 + 3.1^2
theorem problem1 : 6.9^2 + 6.2 * 6.9 + 3.1^2 = 100 := 
by
  sorry

-- Problem 2: Simplify and find the value of the expression with given conditions
theorem problem2 (a b : ℝ) (h1 : a = 1) (h2 : b = 0.5) :
  (a^2 * b^3 + 2 * a^3 * b) / (2 * a * b) - (a + 2 * b) * (a - 2 * b) = 9 / 8 :=
by
  sorry

end problem1_problem2_l1731_173140


namespace extreme_value_of_f_range_of_a_l1731_173117

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem extreme_value_of_f (a : ℝ) (ha : 0 < a) : ∃ x, f x a = a - a * Real.log a - 1 :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 a = f x2 a ∧ abs (x1 - x2) ≥ 1 ) →
  (e - 1 < a ∧ a < Real.exp 2 - Real.exp 1) :=
sorry

end extreme_value_of_f_range_of_a_l1731_173117


namespace find_a_l1731_173168

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end find_a_l1731_173168


namespace sum_of_coordinates_l1731_173125

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l1731_173125


namespace exists_natural_pairs_a_exists_natural_pair_b_l1731_173121

open Nat

-- Part (a) Statement
theorem exists_natural_pairs_a (x y : ℕ) :
  x^2 - y^2 = 105 → (x, y) = (53, 52) ∨ (x, y) = (19, 16) ∨ (x, y) = (13, 8) ∨ (x, y) = (11, 4) :=
sorry

-- Part (b) Statement
theorem exists_natural_pair_b (x y : ℕ) :
  2*x^2 + 5*x*y - 12*y^2 = 28 → (x, y) = (8, 5) :=
sorry

end exists_natural_pairs_a_exists_natural_pair_b_l1731_173121


namespace translate_vertex_l1731_173181

/-- Given points A and B and their translations, verify the translated coordinates of B --/
theorem translate_vertex (A A' B B' : ℝ × ℝ)
  (hA : A = (0, 2))
  (hA' : A' = (-1, 0))
  (hB : B = (2, -1))
  (h_translation : A' = (A.1 - 1, A.2 - 2)) :
  B' = (B.1 - 1, B.2 - 2) :=
by
  sorry

end translate_vertex_l1731_173181


namespace ring_toss_total_earnings_l1731_173161

theorem ring_toss_total_earnings :
  let earnings_first_ring_day1 := 761
  let days_first_ring_day1 := 88
  let earnings_first_ring_day2 := 487
  let days_first_ring_day2 := 20
  let earnings_second_ring_day1 := 569
  let days_second_ring_day1 := 66
  let earnings_second_ring_day2 := 932
  let days_second_ring_day2 := 15

  let total_first_ring := (earnings_first_ring_day1 * days_first_ring_day1) + (earnings_first_ring_day2 * days_first_ring_day2)
  let total_second_ring := (earnings_second_ring_day1 * days_second_ring_day1) + (earnings_second_ring_day2 * days_second_ring_day2)
  let total_earnings := total_first_ring + total_second_ring

  total_earnings = 128242 :=
by
  sorry

end ring_toss_total_earnings_l1731_173161


namespace A_time_to_cover_distance_is_45_over_y_l1731_173106

variable (y : ℝ)
variable (h0 : y > 0)
variable (h1 : (45 : ℝ) / (y - 2 / 3) - (45 : ℝ) / y = 3 / 4)

theorem A_time_to_cover_distance_is_45_over_y :
  45 / y = 45 / y :=
by
  sorry

end A_time_to_cover_distance_is_45_over_y_l1731_173106


namespace find_side_y_l1731_173150

noncomputable def side_length_y : ℝ :=
  let AB := 10 / Real.sqrt 2
  let AD := 10
  let CD := AD / 2
  CD * Real.sqrt 3

theorem find_side_y : side_length_y = 5 * Real.sqrt 3 := by
  let AB : ℝ := 10 / Real.sqrt 2
  let AD : ℝ := 10
  let CD : ℝ := AD / 2
  have h1 : CD * Real.sqrt 3 = 5 * Real.sqrt 3 := by sorry
  exact h1

end find_side_y_l1731_173150


namespace compute_expression_l1731_173189

theorem compute_expression : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end compute_expression_l1731_173189


namespace train_length_l1731_173192

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h_speed : speed_kmph = 60) 
  (h_time : time_sec = 7.199424046076314) 
  (h_length : length_m = 120)
  : speed_kmph * (1000 / 3600) * time_sec = length_m :=
by 
  sorry

end train_length_l1731_173192


namespace train_speed_l1731_173118

theorem train_speed (length_m : ℝ) (time_s : ℝ) (h_length : length_m = 133.33333333333334) (h_time : time_s = 8) : 
  let length_km := length_m / 1000
  let time_hr := time_s / 3600
  length_km / time_hr = 60 :=
by
  sorry

end train_speed_l1731_173118


namespace equivalent_operation_l1731_173170

theorem equivalent_operation (x : ℚ) : (x * (2 / 5)) / (4 / 7) = x * (7 / 10) :=
by
  sorry

end equivalent_operation_l1731_173170


namespace recipe_flour_amount_l1731_173153

theorem recipe_flour_amount
  (cups_of_sugar : ℕ) (cups_of_salt : ℕ) (cups_of_flour_added : ℕ)
  (additional_cups_of_flour : ℕ)
  (h1 : cups_of_sugar = 2)
  (h2 : cups_of_salt = 80)
  (h3 : cups_of_flour_added = 7)
  (h4 : additional_cups_of_flour = cups_of_sugar + 1) :
  cups_of_flour_added + additional_cups_of_flour = 10 :=
by {
  sorry
}

end recipe_flour_amount_l1731_173153


namespace required_words_to_learn_l1731_173123

def total_words : ℕ := 500
def required_percentage : ℕ := 85

theorem required_words_to_learn (x : ℕ) :
  (x : ℚ) / total_words ≥ (required_percentage : ℚ) / 100 ↔ x ≥ 425 := 
sorry

end required_words_to_learn_l1731_173123


namespace find_a_l1731_173194

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x / (3 * x + 4)

theorem find_a (a : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : (f a) (f a x) = x → a = -2 := by
  unfold f
  -- Remaining proof steps skipped
  sorry

end find_a_l1731_173194


namespace total_beads_needed_l1731_173144

-- Condition 1: Number of members in the crafts club
def members := 9

-- Condition 2: Number of necklaces each member makes
def necklaces_per_member := 2

-- Condition 3: Number of beads each necklace requires
def beads_per_necklace := 50

-- Total number of beads needed
theorem total_beads_needed :
  (members * (necklaces_per_member * beads_per_necklace)) = 900 := 
by
  sorry

end total_beads_needed_l1731_173144


namespace inequality_proof_l1731_173122

-- Define the inequality problem in Lean 4
theorem inequality_proof (x y : ℝ) (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : x * y = 1) : 
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by
  sorry

end inequality_proof_l1731_173122


namespace sean_bought_3_sodas_l1731_173149

def soda_cost (S : ℕ) : ℕ := S * 1
def soup_cost (S : ℕ) (C : ℕ) : Prop := C = S
def sandwich_cost (C : ℕ) (X : ℕ) : Prop := X = 3 * C
def total_cost (S C X : ℕ) : Prop := S + 2 * C + X = 18

theorem sean_bought_3_sodas (S C X : ℕ) (h1 : soup_cost S C) (h2 : sandwich_cost C X) (h3 : total_cost S C X) : S = 3 :=
by
  sorry

end sean_bought_3_sodas_l1731_173149


namespace second_smallest_palindromic_prime_l1731_173146

-- Three digit number definition
def three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Palindromic number definition
def is_palindromic (n : ℕ) : Prop := 
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds = ones 

-- Prime number definition
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Second-smallest three-digit palindromic prime
theorem second_smallest_palindromic_prime :
  ∃ n : ℕ, three_digit_number n ∧ is_palindromic n ∧ is_prime n ∧ 
  ∃ m : ℕ, three_digit_number m ∧ is_palindromic m ∧ is_prime m ∧ m > 101 ∧ m < n ∧ 
  n = 131 := 
by
  sorry

end second_smallest_palindromic_prime_l1731_173146


namespace sum_of_reciprocals_l1731_173141

theorem sum_of_reciprocals (a b : ℝ) (h_sum : a + b = 15) (h_prod : a * b = 225) :
  (1 / a) + (1 / b) = 1 / 15 :=
by 
  sorry

end sum_of_reciprocals_l1731_173141


namespace sum_mod_13_l1731_173171

theorem sum_mod_13 (a b c d e : ℤ) (ha : a % 13 = 3) (hb : b % 13 = 5) (hc : c % 13 = 7) (hd : d % 13 = 9) (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by
  -- The proof can be constructed here
  sorry

end sum_mod_13_l1731_173171


namespace determine_no_conditionals_l1731_173173

def problem_requires_conditionals (n : ℕ) : Prop :=
  n = 3 ∨ n = 4

theorem determine_no_conditionals :
  problem_requires_conditionals 1 = false ∧
  problem_requires_conditionals 2 = false ∧
  problem_requires_conditionals 3 = true ∧
  problem_requires_conditionals 4 = true :=
by sorry

end determine_no_conditionals_l1731_173173


namespace range_of_a_l1731_173103

theorem range_of_a (f : ℝ → ℝ) (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) (h_ineq : f (1 - a) < f (2 * a - 1)) : a < 2 / 3 :=
sorry

end range_of_a_l1731_173103


namespace common_points_l1731_173195

variable {R : Type*} [LinearOrderedField R]

def eq1 (x y : R) : Prop := x - y + 2 = 0
def eq2 (x y : R) : Prop := 3 * x + y - 4 = 0
def eq3 (x y : R) : Prop := x + y - 2 = 0
def eq4 (x y : R) : Prop := 2 * x - 5 * y + 7 = 0

theorem common_points : ∃ s : Finset (R × R), 
  (∀ p ∈ s, eq1 p.1 p.2 ∨ eq2 p.1 p.2) ∧ (∀ p ∈ s, eq3 p.1 p.2 ∨ eq4 p.1 p.2) ∧ s.card = 6 :=
by
  sorry

end common_points_l1731_173195


namespace jodi_third_week_miles_l1731_173100

theorem jodi_third_week_miles (total_miles : ℕ) (first_week : ℕ) (second_week : ℕ) (fourth_week : ℕ) (days_per_week : ℕ) (third_week_miles_per_day : ℕ) 
  (H1 : first_week * days_per_week + second_week * days_per_week + third_week_miles_per_day * days_per_week + fourth_week * days_per_week = total_miles)
  (H2 : first_week = 1) 
  (H3 : second_week = 2) 
  (H4 : fourth_week = 4)
  (H5 : total_miles = 60)
  (H6 : days_per_week = 6) :
    third_week_miles_per_day = 3 :=
by sorry

end jodi_third_week_miles_l1731_173100


namespace general_formula_arithmetic_sequence_l1731_173147

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

theorem general_formula_arithmetic_sequence (x : ℝ) (a : ℕ → ℝ) 
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n : ℕ, (a n = 2 * n - 4) ∨ (a n = 4 - 2 * n) :=
by
  sorry

end general_formula_arithmetic_sequence_l1731_173147


namespace find_savings_l1731_173136

-- Definitions and conditions from the problem
def income : ℕ := 36000
def ratio_income_to_expenditure : ℚ := 9 / 8
def expenditure : ℚ := 36000 * (8 / 9)
def savings : ℚ := income - expenditure

-- The theorem statement to prove
theorem find_savings : savings = 4000 := by
  sorry

end find_savings_l1731_173136


namespace max_distance_on_curve_and_ellipse_l1731_173116

noncomputable def max_distance_between_P_and_Q : ℝ :=
  6 * Real.sqrt 2

theorem max_distance_on_curve_and_ellipse :
  ∃ P Q, (P ∈ { p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2 }) ∧ 
         (Q ∈ { q : ℝ × ℝ | q.1^2 / 10 + q.2^2 = 1 }) ∧ 
         (dist P Q = max_distance_between_P_and_Q) := 
sorry

end max_distance_on_curve_and_ellipse_l1731_173116


namespace first_competitor_hotdogs_l1731_173114

theorem first_competitor_hotdogs (x y z : ℕ) (h1 : y = 3 * x) (h2 : z = 2 * y) (h3 : z * 5 = 300) : x = 10 :=
sorry

end first_competitor_hotdogs_l1731_173114


namespace probability_of_three_draws_l1731_173175

noncomputable def box_chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def valid_first_two_draws (a b : ℕ) : Prop :=
  a + b <= 7

def prob_three_draws_to_exceed_seven : ℚ :=
  1 / 6

theorem probability_of_three_draws :
  (∃ (draws : List ℕ), (draws.length = 3) ∧ (draws.sum > 7)
    ∧ (∀ x ∈ draws, x ∈ box_chips)
    ∧ (∀ (a b : ℕ), (a ∈ box_chips ∧ b ∈ box_chips) → valid_first_two_draws a b))
  → prob_three_draws_to_exceed_seven = 1 / 6 :=
sorry

end probability_of_three_draws_l1731_173175


namespace minute_hand_distance_traveled_l1731_173148

noncomputable def radius : ℝ := 8
noncomputable def minutes_in_one_revolution : ℝ := 60
noncomputable def total_minutes : ℝ := 45

theorem minute_hand_distance_traveled :
  (total_minutes / minutes_in_one_revolution) * (2 * Real.pi * radius) = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_traveled_l1731_173148


namespace sum_of_reciprocals_l1731_173167

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + y = 3 * x * y) (h2 : x - y = 2) : (1/x + 1/y) = 4/3 :=
by
  -- Proof omitted
  sorry

end sum_of_reciprocals_l1731_173167


namespace even_function_f_l1731_173133

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem even_function_f (hx : ∀ x : ℝ, f (-x) = f x) 
  (hg : ∀ x : ℝ, g (-x) = -g x)
  (h_pass : g (-1) = 1)
  (hg_eq_f : ∀ x : ℝ, g x = f (x - 1)) 
  : f 7 + f 8 = -1 := 
by
  sorry

end even_function_f_l1731_173133


namespace landmark_postcards_probability_l1731_173111

theorem landmark_postcards_probability :
  let total_postcards := 12
  let landmark_postcards := 4
  let total_arrangements := Nat.factorial total_postcards
  let favorable_arrangements := Nat.factorial (total_postcards - landmark_postcards + 1) * Nat.factorial landmark_postcards
  favorable_arrangements / total_arrangements = (1:ℝ) / 55 :=
by
  sorry

end landmark_postcards_probability_l1731_173111


namespace solve_for_y_l1731_173177

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end solve_for_y_l1731_173177


namespace total_legs_of_animals_l1731_173179

def num_kangaroos := 23
def num_goats := 3 * num_kangaroos
def legs_per_kangaroo := 2
def legs_per_goat := 4

def total_legs := (num_kangaroos * legs_per_kangaroo) + (num_goats * legs_per_goat)

theorem total_legs_of_animals : total_legs = 322 := by
  sorry

end total_legs_of_animals_l1731_173179


namespace g_inv_f_five_l1731_173160

-- Declare the existence of functions f and g and their inverses
variables (f g : ℝ → ℝ)

-- Given condition from the problem
axiom inv_cond : ∀ x, f⁻¹ (g x) = 4 * x - 1

-- Define the specific problem to solve
theorem g_inv_f_five : g⁻¹ (f 5) = 3 / 2 :=
by
  sorry

end g_inv_f_five_l1731_173160


namespace maximum_capacity_of_smallest_barrel_l1731_173184

theorem maximum_capacity_of_smallest_barrel : 
  ∃ (A B C D E F : ℕ), 
    8 ≤ A ∧ A ≤ 16 ∧
    8 ≤ B ∧ B ≤ 16 ∧
    8 ≤ C ∧ C ≤ 16 ∧
    8 ≤ D ∧ D ≤ 16 ∧
    8 ≤ E ∧ E ≤ 16 ∧
    8 ≤ F ∧ F ≤ 16 ∧
    (A + B + C + D + E + F = 72) ∧
    ((C + D) / 2 = 14) ∧ 
    (F = 11 ∨ F = 13) ∧
    (∀ (A' : ℕ), 8 ≤ A' ∧ A' ≤ 16 ∧
      ∃ (B' C' D' E' F' : ℕ), 
      8 ≤ B' ∧ B' ≤ 16 ∧
      8 ≤ C' ∧ C' ≤ 16 ∧
      8 ≤ D' ∧ D' ≤ 16 ∧
      8 ≤ E' ∧ E' ≤ 16 ∧
      8 ≤ F' ∧ F' ≤ 16 ∧
      (A' + B' + C' + D' + E' + F' = 72) ∧
      ((C' + D') / 2 = 14) ∧ 
      (F' = 11 ∨ F' = 13) → A' ≤ A ) :=
sorry

end maximum_capacity_of_smallest_barrel_l1731_173184


namespace tan_of_angle_in_third_quadrant_l1731_173152

theorem tan_of_angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α = -12 / 13) (h2 : π < α ∧ α < 3 * π / 2) : Real.tan α = 12 / 5 := 
sorry

end tan_of_angle_in_third_quadrant_l1731_173152


namespace wine_cost_today_l1731_173128

theorem wine_cost_today (C : ℝ) (h1 : ∀ (new_tariff : ℝ), new_tariff = 0.25) (h2 : ∀ (total_increase : ℝ), total_increase = 25) (h3 : C = 20) : 5 * (1.25 * C - C) = 25 :=
by
  sorry

end wine_cost_today_l1731_173128


namespace problem_1_problem_2_l1731_173119

theorem problem_1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : (a + b) * (a^5 + b^5) ≥ 4 :=
sorry

theorem problem_2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : a + b ≤ 2 :=
sorry

end problem_1_problem_2_l1731_173119


namespace trapezoid_circle_center_l1731_173151

theorem trapezoid_circle_center 
  (EF GH : ℝ)
  (FG HE : ℝ)
  (p q : ℕ) 
  (rel_prime : Nat.gcd p q = 1)
  (EQ GH : ℝ)
  (h1 : EF = 105)
  (h2 : FG = 57)
  (h3 : GH = 22)
  (h4 : HE = 80)
  (h5 : EQ = p / q)
  (h6 : p = 10)
  (h7 : q = 1) :
  p + q = 11 :=
by
  sorry

end trapezoid_circle_center_l1731_173151


namespace ratio_of_linear_combination_l1731_173180

theorem ratio_of_linear_combination (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 :=
by
  sorry

end ratio_of_linear_combination_l1731_173180


namespace julia_played_more_kids_on_monday_l1731_173127

def n_monday : ℕ := 6
def n_tuesday : ℕ := 5

theorem julia_played_more_kids_on_monday : n_monday - n_tuesday = 1 := by
  -- Proof goes here
  sorry

end julia_played_more_kids_on_monday_l1731_173127


namespace netCaloriesConsumedIs1082_l1731_173166

-- Given conditions
def caloriesPerCandyBar : ℕ := 347
def candyBarsEatenInAWeek : ℕ := 6
def caloriesBurnedInAWeek : ℕ := 1000

-- Net calories calculation
def netCaloriesInAWeek (calsPerBar : ℕ) (barsPerWeek : ℕ) (calsBurned : ℕ) : ℕ :=
  calsPerBar * barsPerWeek - calsBurned

-- The theorem to prove
theorem netCaloriesConsumedIs1082 :
  netCaloriesInAWeek caloriesPerCandyBar candyBarsEatenInAWeek caloriesBurnedInAWeek = 1082 :=
by
  sorry

end netCaloriesConsumedIs1082_l1731_173166


namespace nondegenerate_ellipse_iff_l1731_173126

theorem nondegenerate_ellipse_iff (k : ℝ) :
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = k) ↔ k > -117/4 :=
by
  sorry

end nondegenerate_ellipse_iff_l1731_173126


namespace total_votes_l1731_173197

theorem total_votes (total_votes : ℕ) (brenda_votes : ℕ) (fraction : ℚ) (h : brenda_votes = fraction * total_votes) (h_fraction : fraction = 1 / 5) (h_brenda : brenda_votes = 15) : 
  total_votes = 75 := 
by
  sorry

end total_votes_l1731_173197


namespace find_other_polynomial_l1731_173199

variables {a b c d : ℤ}

theorem find_other_polynomial (h : ∀ P Q : ℤ, P - Q = c^2 * d^2 - a^2 * b^2) 
  (P : ℤ) (hP : P = a^2 * b^2 + c^2 * d^2 - 2 * a * b * c * d) : 
  (∃ Q : ℤ, Q = 2 * c^2 * d^2 - 2 * a * b * c * d) ∨ 
  (∃ Q : ℤ, Q = 2 * a^2 * b^2 - 2 * a * b * c * d) :=
by {
  sorry
}

end find_other_polynomial_l1731_173199


namespace pascal_triangle_ratio_l1731_173113

theorem pascal_triangle_ratio (n r : ℕ) :
  (r + 1 = (4 * (n - r)) / 5) ∧ (r + 2 = (5 * (n - r - 1)) / 6) → n = 53 :=
by sorry

end pascal_triangle_ratio_l1731_173113


namespace van_distance_l1731_173158

theorem van_distance
  (D : ℝ)  -- distance the van needs to cover
  (S : ℝ)  -- original speed
  (h1 : D = S * 5)  -- the van takes 5 hours to cover the distance D
  (h2 : D = 62 * 7.5)  -- the van should maintain a speed of 62 kph to cover the same distance in 7.5 hours
  : D = 465 :=         -- prove that the distance D is 465 kilometers
by
  sorry

end van_distance_l1731_173158


namespace total_pairs_of_shoes_l1731_173182

-- Conditions as Definitions
def blue_shoes := 540
def purple_shoes := 355
def green_shoes := purple_shoes  -- The number of green shoes is equal to the number of purple shoes

-- The theorem we need to prove
theorem total_pairs_of_shoes : blue_shoes + green_shoes + purple_shoes = 1250 := by
  sorry

end total_pairs_of_shoes_l1731_173182


namespace eval_operation_l1731_173109

-- Definition of the * operation based on the given table
def op (a b : ℕ) : ℕ :=
  match a, b with
  | 1, 1 => 4
  | 1, 2 => 1
  | 1, 3 => 2
  | 1, 4 => 3
  | 2, 1 => 1
  | 2, 2 => 3
  | 2, 3 => 4
  | 2, 4 => 2
  | 3, 1 => 2
  | 3, 2 => 4
  | 3, 3 => 1
  | 3, 4 => 3
  | 4, 1 => 3
  | 4, 2 => 2
  | 4, 3 => 3
  | 4, 4 => 4
  | _, _ => 0 -- Default case (not needed as per the given problem definition)

-- Statement of the problem in Lean 4
theorem eval_operation : op (op 3 1) (op 4 2) = 3 :=
by {
  sorry -- Proof to be provided
}

end eval_operation_l1731_173109


namespace indigo_restaurant_total_reviews_l1731_173186

-- Define the number of 5-star reviews
def five_star_reviews : Nat := 6

-- Define the number of 4-star reviews
def four_star_reviews : Nat := 7

-- Define the number of 3-star reviews
def three_star_reviews : Nat := 4

-- Define the number of 2-star reviews
def two_star_reviews : Nat := 1

-- Define the total number of reviews
def total_reviews : Nat := five_star_reviews + four_star_reviews + three_star_reviews + two_star_reviews

-- Proof that the total number of customer reviews is 18
theorem indigo_restaurant_total_reviews : total_reviews = 18 :=
by
  -- Direct calculation
  sorry

end indigo_restaurant_total_reviews_l1731_173186


namespace train_B_time_to_destination_l1731_173143

-- Definitions (conditions)
def speed_train_A := 60  -- Train A travels at 60 kmph
def speed_train_B := 90  -- Train B travels at 90 kmph
def time_train_A_after_meeting := 9 -- Train A takes 9 hours after meeting train B

-- Theorem statement
theorem train_B_time_to_destination 
  (speed_A : ℝ)
  (speed_B : ℝ)
  (time_A_after_meeting : ℝ)
  (time_B_to_destination : ℝ) :
  speed_A = speed_train_A ∧
  speed_B = speed_train_B ∧
  time_A_after_meeting = time_train_A_after_meeting →
  time_B_to_destination = 4.5 :=
by
  sorry

end train_B_time_to_destination_l1731_173143


namespace negation_statement_l1731_173198

open Set

variable {S : Set ℝ}

theorem negation_statement (h : ∀ x ∈ S, 3 * x - 5 > 0) : ∃ x ∈ S, 3 * x - 5 ≤ 0 :=
sorry

end negation_statement_l1731_173198


namespace parabola_and_hyperbola_focus_equal_l1731_173163

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) :=
(2, 0)

noncomputable def hyperbola_focus : (ℝ × ℝ) :=
(2, 0)

theorem parabola_and_hyperbola_focus_equal
  (p : ℝ)
  (h_parabola : parabola_focus p = (2, 0))
  (h_hyperbola : hyperbola_focus = (2, 0)) :
  p = 4 := by
  sorry

end parabola_and_hyperbola_focus_equal_l1731_173163


namespace parabola_vertex_coordinate_l1731_173185

theorem parabola_vertex_coordinate :
  ∀ x_P : ℝ, 
  (P : ℝ × ℝ) → 
  (P = (x_P, 1/2 * x_P^2)) → 
  (dist P (0, 1/2) = 3) →
  P.2 = 5 / 2 :=
by sorry

end parabola_vertex_coordinate_l1731_173185


namespace problem_solution_l1731_173110

def p : Prop := ∀ x : ℝ, |x| ≥ 0
def q : Prop := ∃ x : ℝ, x = 2 ∧ x + 2 = 0

theorem problem_solution : p ∧ ¬q :=
by
  -- Here we would provide the proof to show that p ∧ ¬q is true
  sorry

end problem_solution_l1731_173110


namespace product_x_z_l1731_173107

-- Defining the variables x, y, z as positive integers and the given conditions.
theorem product_x_z (x y z : ℕ) (h1 : x = 4 * y) (h2 : z = 2 * x) (h3 : x + y + z = 3 * y ^ 2) : 
    x * z = 5408 / 9 := 
  sorry

end product_x_z_l1731_173107


namespace sixth_term_is_sixteen_l1731_173138

-- Definition of the conditions
def first_term : ℝ := 512
def eighth_term (r : ℝ) : Prop := 512 * r^7 = 2

-- Proving the 6th term is 16 given the conditions
theorem sixth_term_is_sixteen (r : ℝ) (hr : eighth_term r) :
  512 * r^5 = 16 :=
by
  sorry

end sixth_term_is_sixteen_l1731_173138


namespace common_ratio_half_l1731_173157

-- Definitions based on conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n+1) = a n * q
def arith_seq (x y z : ℝ) := x + z = 2 * y

-- Theorem statement
theorem common_ratio_half (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q)
  (h_arith : arith_seq (a 5) (a 6 + a 8) (a 7)) : q = 1 / 2 := 
sorry

end common_ratio_half_l1731_173157
