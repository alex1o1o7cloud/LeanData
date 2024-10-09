import Mathlib

namespace functional_equation_solution_l993_99356

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by
  sorry

end functional_equation_solution_l993_99356


namespace middle_integer_of_sum_is_120_l993_99357

-- Define the condition that three consecutive integers sum to 360
def consecutive_integers_sum_to (n : ℤ) (sum : ℤ) : Prop :=
  (n - 1) + n + (n + 1) = sum

-- The statement to prove
theorem middle_integer_of_sum_is_120 (n : ℤ) :
  consecutive_integers_sum_to n 360 → n = 120 :=
by
  sorry

end middle_integer_of_sum_is_120_l993_99357


namespace expected_value_dodecahedral_die_l993_99362

-- Define the faces of the die
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the scoring rule
def score (n : ℕ) : ℕ :=
  if n ≤ 6 then 2 * n else n

-- The probability of each face
def prob : ℚ := 1 / 12

-- Calculate the expected value
noncomputable def expected_value : ℚ :=
  prob * (score 1 + score 2 + score 3 + score 4 + score 5 + score 6 + 
          score 7 + score 8 + score 9 + score 10 + score 11 + score 12)

-- State the theorem to be proved
theorem expected_value_dodecahedral_die : expected_value = 8.25 := 
  sorry

end expected_value_dodecahedral_die_l993_99362


namespace find_M_l993_99315

theorem find_M : 
  let S := (981 + 983 + 985 + 987 + 989 + 991 + 993 + 995 + 997 + 999)
  let Target := 5100 - M
  S = Target → M = 4800 :=
by
  sorry

end find_M_l993_99315


namespace dvd_cost_l993_99344

-- Given conditions
def vhs_trade_in_value : Int := 2
def number_of_movies : Int := 100
def total_replacement_cost : Int := 800

-- Statement to prove
theorem dvd_cost :
  ((number_of_movies * vhs_trade_in_value) + (number_of_movies * 6) = total_replacement_cost) :=
by
  sorry

end dvd_cost_l993_99344


namespace probability_of_passing_test_l993_99309

theorem probability_of_passing_test (p : ℝ) (h : p + p * (1 - p) + p * (1 - p)^2 = 0.784) : p = 0.4 :=
sorry

end probability_of_passing_test_l993_99309


namespace range_of_b_l993_99385

theorem range_of_b (y : ℝ) (b : ℝ) (h1 : |y - 2| + |y - 5| < b) (h2 : b > 1) : b > 3 := 
sorry

end range_of_b_l993_99385


namespace polygon_perimeter_eq_21_l993_99387

-- Definitions and conditions from the given problem
def rectangle_side_a := 6
def rectangle_side_b := 4
def triangle_hypotenuse := 5

-- The combined polygon perimeter proof statement
theorem polygon_perimeter_eq_21 :
  let rectangle_perimeter := 2 * (rectangle_side_a + rectangle_side_b)
  let adjusted_perimeter := rectangle_perimeter - rectangle_side_b + triangle_hypotenuse
  adjusted_perimeter = 21 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end polygon_perimeter_eq_21_l993_99387


namespace find_weight_of_b_l993_99386

variable (a b c d : ℝ)

def average_weight_of_four : Prop := (a + b + c + d) / 4 = 45

def average_weight_of_a_and_b : Prop := (a + b) / 2 = 42

def average_weight_of_b_and_c : Prop := (b + c) / 2 = 43

def ratio_of_d_to_a : Prop := d / a = 3 / 4

theorem find_weight_of_b (h1 : average_weight_of_four a b c d)
                        (h2 : average_weight_of_a_and_b a b)
                        (h3 : average_weight_of_b_and_c b c)
                        (h4 : ratio_of_d_to_a a d) :
    b = 29.43 :=
  by sorry

end find_weight_of_b_l993_99386


namespace total_worth_of_stock_l993_99351

theorem total_worth_of_stock (X : ℝ) :
  (0.30 * 0.10 * X + 0.40 * -0.05 * X + 0.30 * -0.10 * X = -500) → X = 25000 :=
by
  intro h
  -- Proof to be completed
  sorry

end total_worth_of_stock_l993_99351


namespace total_liters_needed_to_fill_two_tanks_l993_99312

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end total_liters_needed_to_fill_two_tanks_l993_99312


namespace solution_set_of_inequality_l993_99372

variables {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def increasing_on (f : R → R) (S : Set R) : Prop :=
∀ ⦃x y⦄, x ∈ S → y ∈ S → x < y → f x < f y

theorem solution_set_of_inequality
  {f : R → R}
  (h_odd : odd_function f)
  (h_neg_one : f (-1) = 0)
  (h_increasing : increasing_on f {x : R | x > 0}) :
  {x : R | x * f x > 0} = {x : R | x < -1} ∪ {x : R | x > 1} :=
sorry

end solution_set_of_inequality_l993_99372


namespace simplify_expression_l993_99354

theorem simplify_expression (a : ℝ) : (2 * a - 3)^2 - (a + 5) * (a - 5) = 3 * a^2 - 12 * a + 34 :=
by
  sorry

end simplify_expression_l993_99354


namespace sequence_general_formula_l993_99314

theorem sequence_general_formula (a : ℕ → ℚ) 
  (h1 : a 1 = 1 / 2) 
  (h_rec : ∀ n : ℕ, a (n + 2) = 3 * a (n + 1) / (a (n + 1) + 3)) 
  (n : ℕ) : 
  a (n + 1) = 3 / (n + 6) :=
by
  sorry

end sequence_general_formula_l993_99314


namespace property_depreciation_rate_l993_99330

noncomputable def initial_value : ℝ := 25599.08977777778
noncomputable def final_value : ℝ := 21093
noncomputable def annual_depreciation_rate : ℝ := 0.063

theorem property_depreciation_rate :
  final_value = initial_value * (1 - annual_depreciation_rate)^3 :=
sorry

end property_depreciation_rate_l993_99330


namespace range_of_a_monotonically_decreasing_l993_99384

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv (f a) x ≤ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_monotonically_decreasing_l993_99384


namespace sine_of_angle_from_point_l993_99323

theorem sine_of_angle_from_point (x y : ℤ) (r : ℝ) (h : r = Real.sqrt ((x : ℝ)^2 + (y : ℝ)^2)) (hx : x = -12) (hy : y = 5) :
  Real.sin (Real.arctan (y / x)) = y / r := 
by
  sorry

end sine_of_angle_from_point_l993_99323


namespace min_trig_expression_l993_99329

theorem min_trig_expression (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = Real.pi) : 
  ∃ (x : ℝ), (x = 16 - 8 * Real.sqrt 2) ∧ (∀ A B C, 0 < A → 0 < B → 0 < C → A + B + C = Real.pi → 
    (1 / (Real.sin A)^2 + 1 / (Real.sin B)^2 + 4 / (1 + Real.sin C)) ≥ x) := 
sorry

end min_trig_expression_l993_99329


namespace depth_notation_l993_99304

theorem depth_notation (x y : ℤ) (hx : x = 9050) (hy : y = -10907) : -y = x :=
by
  sorry

end depth_notation_l993_99304


namespace point_on_x_axis_l993_99379

theorem point_on_x_axis (m : ℝ) (h : (2 * m + 3) = 0) : m = -3 / 2 :=
sorry

end point_on_x_axis_l993_99379


namespace volume_of_regular_tetrahedron_l993_99346

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 2) / 12

theorem volume_of_regular_tetrahedron (a : ℝ) : 
  volume_of_tetrahedron a = (a ^ 3 * Real.sqrt 2) / 12 := 
by
  sorry

end volume_of_regular_tetrahedron_l993_99346


namespace correct_histogram_height_representation_l993_99320

   def isCorrectHeightRepresentation (heightRep : String) : Prop :=
     heightRep = "ratio of the frequency of individuals in that group within the sample to the class interval"

   theorem correct_histogram_height_representation :
     isCorrectHeightRepresentation "ratio of the frequency of individuals in that group within the sample to the class interval" :=
   by 
     sorry
   
end correct_histogram_height_representation_l993_99320


namespace smallest_four_digit_divisible_by_six_l993_99303

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l993_99303


namespace cos2_alpha_plus_2sin2_alpha_l993_99366

theorem cos2_alpha_plus_2sin2_alpha {α : ℝ} (h : Real.tan α = 3 / 4) : 
    Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by 
  sorry

end cos2_alpha_plus_2sin2_alpha_l993_99366


namespace find_x_l993_99396

theorem find_x (x y : ℝ) (h : y ≠ -5 * x) : (x - 5) / (5 * x + y) = 0 → x = 5 := by
  sorry

end find_x_l993_99396


namespace line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l993_99335

theorem line_through_point_parallel_to_given_line :
  ∃ c : ℤ, (∀ x y : ℤ, 2 * x + 3 * y + c = 0 ↔ (x, y) = (2, 1)) ∧ c = -7 :=
sorry

theorem line_through_point_sum_intercepts_is_minus_four :
  ∃ (a b : ℤ), (∀ x y : ℤ, (x / a) + (y / b) = 1 ↔ (x, y) = (-3, 1)) ∧ (a + b = -4) ∧ 
  ((a = -6 ∧ b = 2) ∨ (a = -2 ∧ b = -2)) ∧ 
  ((∀ x y : ℤ, x - 3 * y + 6 = 0 ↔ (x, y) = (-3, 1)) ∨ 
  (∀ x y : ℤ, x + y + 2 = 0 ↔ (x, y) = (-3, 1))) :=
sorry

end line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l993_99335


namespace Dvaneft_percentage_bounds_l993_99327

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end Dvaneft_percentage_bounds_l993_99327


namespace pau_total_ordered_correct_l993_99365

-- Define the initial pieces of fried chicken ordered by Kobe
def kobe_order : ℝ := 5

-- Define Pau's initial order as twice Kobe's order plus 2.5 pieces
def pau_initial_order : ℝ := (2 * kobe_order) + 2.5

-- Define Shaquille's initial order as 50% more than Pau's initial order
def shaq_initial_order : ℝ := pau_initial_order * 1.5

-- Define the total pieces of chicken Pau will have eaten by the end
def pau_total_ordered : ℝ := 2 * pau_initial_order

-- Prove that Pau will have eaten 25 pieces of fried chicken by the end
theorem pau_total_ordered_correct : pau_total_ordered = 25 := by
  sorry

end pau_total_ordered_correct_l993_99365


namespace one_div_m_plus_one_div_n_l993_99355

theorem one_div_m_plus_one_div_n
  {m n : ℕ} 
  (h1 : Nat.gcd m n = 5) 
  (h2 : Nat.lcm m n = 210)
  (h3 : m + n = 75) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 14 :=
by
  sorry

end one_div_m_plus_one_div_n_l993_99355


namespace base_four_product_l993_99319

def base_four_to_decimal (n : ℕ) : ℕ :=
  -- definition to convert base 4 to decimal, skipping details for now
  sorry

def decimal_to_base_four (n : ℕ) : ℕ :=
  -- definition to convert decimal to base 4, skipping details for now
  sorry

theorem base_four_product : 
  base_four_to_decimal 212 * base_four_to_decimal 13 = base_four_to_decimal 10322 :=
sorry

end base_four_product_l993_99319


namespace angle_in_third_quadrant_l993_99360

-- Define the concept of an angle being in a specific quadrant
def is_in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

-- Prove that -1200° is in the third quadrant
theorem angle_in_third_quadrant :
  is_in_third_quadrant (240) → is_in_third_quadrant (-1200 % 360 + 360 * (if -1200 % 360 ≤ 0 then 1 else 0)) :=
by
  sorry

end angle_in_third_quadrant_l993_99360


namespace min_ab_value_l993_99326

theorem min_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 / a + 20 / b = 4) : ab = 25 :=
sorry

end min_ab_value_l993_99326


namespace perpendicular_condition_centroid_coordinates_l993_99337

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 0}
def B : Point := {x := 4, y := 0}
def C (c : ℝ) : Point := {x := 0, y := c}

def vec (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y}

def dot_product (P Q : Point) : ℝ :=
  P.x * Q.x + P.y * Q.y

theorem perpendicular_condition (c : ℝ) (h : dot_product (vec A (C c)) (vec B (C c)) = 0) :
  c = 2 ∨ c = -2 :=
by
  -- proof to be filled in
  sorry

theorem centroid_coordinates (c : ℝ) (h : c = 2 ∨ c = -2) :
  (c = 2 → Point.mk 1 (2 / 3) = Point.mk 1 (2 / 3)) ∧
  (c = -2 → Point.mk 1 (-2 / 3) = Point.mk 1 (-2 / 3)) :=
by
  -- proof to be filled in
  sorry

end perpendicular_condition_centroid_coordinates_l993_99337


namespace percentage_conversion_l993_99316

-- Define the condition
def decimal_fraction : ℝ := 0.05

-- Define the target percentage
def percentage : ℝ := 5

-- State the theorem
theorem percentage_conversion (df : ℝ) (p : ℝ) (h1 : df = 0.05) (h2 : p = 5) : df * 100 = p :=
by
  rw [h1, h2]
  sorry

end percentage_conversion_l993_99316


namespace find_a_l993_99336

open Real

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the line equation passing through P(2,2)
def line_through_P (m b x y : ℝ) : Prop := y = m * x + b ∧ (2, 2) = (x, y)

-- Define the line equation ax - y + 1 = 0
def perpendicular_line (a x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a : ∃ a : ℝ, ∀ x y m b : ℝ,
    circle x y ∧ line_through_P m b x y ∧
    (line_through_P m b x y → perpendicular_line a x y) → a = 2 :=
by
  intros
  sorry

end find_a_l993_99336


namespace total_phones_in_Delaware_l993_99358

def population : ℕ := 974000
def phones_per_1000 : ℕ := 673

theorem total_phones_in_Delaware : (population / 1000) * phones_per_1000 = 655502 := by
  sorry

end total_phones_in_Delaware_l993_99358


namespace pelican_speed_l993_99364

theorem pelican_speed
  (eagle_speed falcon_speed hummingbird_speed total_distance time : ℕ)
  (eagle_distance falcon_distance hummingbird_distance : ℕ)
  (H1 : eagle_speed = 15)
  (H2 : falcon_speed = 46)
  (H3 : hummingbird_speed = 30)
  (H4 : time = 2)
  (H5 : total_distance = 248)
  (H6 : eagle_distance = eagle_speed * time)
  (H7 : falcon_distance = falcon_speed * time)
  (H8 : hummingbird_distance = hummingbird_speed * time)
  (total_other_birds_distance : ℕ)
  (H9 : total_other_birds_distance = eagle_distance + falcon_distance + hummingbird_distance)
  (pelican_distance : ℕ)
  (H10 : pelican_distance = total_distance - total_other_birds_distance)
  (pelican_speed : ℕ)
  (H11 : pelican_speed = pelican_distance / time) :
  pelican_speed = 33 := 
  sorry

end pelican_speed_l993_99364


namespace xyz_sum_fraction_l993_99398

theorem xyz_sum_fraction (a1 a2 a3 b1 b2 b3 c1 c2 c3 a b c : ℤ) 
  (h1 : a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1) = 9)
  (h2 : a * (b2 * c3 - b3 * c2) - a2 * (b * c3 - b3 * c) + a3 * (b * c2 - b2 * c) = 17)
  (h3 : a1 * (b * c3 - b3 * c) - a * (b1 * c3 - b3 * c1) + a3 * (b1 * c - b * c1) = -8)
  (h4 : a1 * (b2 * c - b * c2) - a2 * (b1 * c - b * c1) + a * (b1 * c2 - b2 * c1) = 7)
  (eq1 : a1 * x + a2 * y + a3 * z = a)
  (eq2 : b1 * x + b2 * y + b3 * z = b)
  (eq3 : c1 * x + c2 * y + c3 * z = c)
  : x + y + z = 16 / 9 := 
sorry

end xyz_sum_fraction_l993_99398


namespace expectation_of_xi_l993_99349

noncomputable def compute_expectation : ℝ := 
  let m : ℝ := 0.3
  let E : ℝ := (1 * 0.5) + (3 * m) + (5 * 0.2)
  E

theorem expectation_of_xi :
  let m: ℝ := 1 - 0.5 - 0.2 
  (0.5 + m + 0.2 = 1) → compute_expectation = 2.4 := 
by
  sorry

end expectation_of_xi_l993_99349


namespace cuboid_count_l993_99313

def length_small (m : ℕ) : ℕ := 6
def width_small (m : ℕ) : ℕ := 4
def height_small (m : ℕ) : ℕ := 3

def length_large (m : ℕ): ℕ := 18
def width_large (m : ℕ) : ℕ := 15
def height_large (m : ℕ) : ℕ := 2

def volume (l : ℕ) (w : ℕ) (h : ℕ) : ℕ := l * w * h

def n_small_cuboids (v_large v_small : ℕ) : ℕ := v_large / v_small

theorem cuboid_count : 
  n_small_cuboids (volume (length_large 1) (width_large 1) (height_large 1)) (volume (length_small 1) (width_small 1) (height_small 1)) = 7 :=
by
  sorry

end cuboid_count_l993_99313


namespace solve_positive_integer_l993_99308

theorem solve_positive_integer (n : ℕ) (h : ∀ m : ℕ, m > 0 → n^m ≥ m^n) : n = 3 :=
sorry

end solve_positive_integer_l993_99308


namespace product_is_even_l993_99341

theorem product_is_even (a b c : ℤ) : Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end product_is_even_l993_99341


namespace primary_school_capacity_l993_99370

variable (x : ℝ)

/-- In a town, there are four primary schools. Two of them can teach 400 students at a time, 
and the other two can teach a certain number of students at a time. These four primary schools 
can teach a total of 1480 students at a time. -/
theorem primary_school_capacity 
  (h1 : 2 * 400 + 2 * x = 1480) : 
  x = 340 :=
sorry

end primary_school_capacity_l993_99370


namespace ellipse_standard_equation_l993_99383

-- Define the conditions
def equation1 (x y : ℝ) : Prop := x^2 + (y^2 / 2) = 1
def equation2 (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def equation3 (x y : ℝ) : Prop := x^2 + (y^2 / 4) = 1
def equation4 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the points
def point1 (x y : ℝ) : Prop := (x = 1 ∧ y = 0)
def point2 (x y : ℝ) : Prop := (x = 0 ∧ y = 2)

-- Define the main theorem
theorem ellipse_standard_equation :
  (equation4 1 0 ∧ equation4 0 2) ↔
  ((equation1 1 0 ∧ equation1 0 2) ∨
   (equation2 1 0 ∧ equation2 0 2) ∨
   (equation3 1 0 ∧ equation3 0 2) ∨
   (equation4 1 0 ∧ equation4 0 2)) :=
by
  sorry

end ellipse_standard_equation_l993_99383


namespace least_number_to_subtract_l993_99342

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (h : 1387 = n + k * 15) : n = 7 :=
by
  sorry

end least_number_to_subtract_l993_99342


namespace find_speed_ratio_l993_99331

noncomputable def circular_track_speed_ratio (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0) : Prop :=
  let t_1 := C / (v_V + v_P)
  let t_2 := (C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r

theorem find_speed_ratio
  (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0)
  (meeting1 : v_V * (C / (v_V + v_P)) + v_P * (C / (v_V + v_P)) = C)
  (lap_vasya : v_V * ((C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))) = C + v_V * (C / (v_V + v_P)))
  (lap_petya : v_P * ((C * (2 * v_P + v_V)) / (v_P * (v_V + v_P))) = C + v_P * (C / (v_V + v_P))) :
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r :=
  sorry

end find_speed_ratio_l993_99331


namespace ratio_of_art_to_math_books_l993_99345

-- The conditions provided
def total_budget : ℝ := 500
def price_math_book : ℝ := 20
def num_math_books : ℕ := 4
def num_art_books : ℕ := num_math_books
def price_art_book : ℝ := 20
def num_science_books : ℕ := num_math_books + 6
def price_science_book : ℝ := 10
def cost_music_books : ℝ := 160

-- Desired proof statement
theorem ratio_of_art_to_math_books : num_art_books / num_math_books = 1 :=
by
  sorry

end ratio_of_art_to_math_books_l993_99345


namespace susan_added_oranges_l993_99378

-- Conditions as definitions
def initial_oranges_in_box : ℝ := 55.0
def final_oranges_in_box : ℝ := 90.0

-- Define the quantity of oranges Susan put into the box
def susan_oranges := final_oranges_in_box - initial_oranges_in_box

-- Theorem statement to prove that the number of oranges Susan put into the box is 35.0
theorem susan_added_oranges : susan_oranges = 35.0 := by
  unfold susan_oranges
  sorry

end susan_added_oranges_l993_99378


namespace tangent_line_hyperbola_eq_l993_99361

noncomputable def tangent_line_ellipse (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0) 
  (h_ell : x0 ^ 2 / a ^ 2 + y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1

noncomputable def tangent_line_hyperbola (a b x0 y0 x y : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h_hyp : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : Prop :=
  (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1

theorem tangent_line_hyperbola_eq (a b x0 y0 x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
  (h_ellipse_tangent : tangent_line_ellipse a b x0 y0 x y h1 h2 h3 (by sorry))
  (h_hyperbola : x0 ^ 2 / a ^ 2 - y0 ^ 2 / b ^ 2 = 1) : 
  tangent_line_hyperbola a b x0 y0 x y h3 h2 h_hyperbola :=
by sorry

end tangent_line_hyperbola_eq_l993_99361


namespace alan_total_spending_l993_99375

-- Define the conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation
def cost_eggs : ℕ := eggs_bought * price_per_egg
def cost_chickens : ℕ := chickens_bought * price_per_chicken
def total_amount_spent : ℕ := cost_eggs + cost_chickens

-- Prove the total amount spent
theorem alan_total_spending : total_amount_spent = 88 := by
  show cost_eggs + cost_chickens = 88
  sorry

end alan_total_spending_l993_99375


namespace LCM_GCD_even_nonnegative_l993_99343

theorem LCM_GCD_even_nonnegative (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  : ∃ (n : ℕ), (n = Nat.lcm a b + Nat.gcd a b - a - b) ∧ (n % 2 = 0) ∧ (0 ≤ n) := 
sorry

end LCM_GCD_even_nonnegative_l993_99343


namespace total_nails_used_l993_99307

-- Given definitions from the conditions
def square_side_length : ℕ := 36
def nails_per_side : ℕ := 40
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

-- Statement of the problem proof
theorem total_nails_used : nails_per_side * sides_of_square - corners_of_square = 156 := by
  sorry

end total_nails_used_l993_99307


namespace abs_neg_two_thirds_l993_99359

theorem abs_neg_two_thirds : abs (-2/3 : ℝ) = 2/3 :=
by
  sorry

end abs_neg_two_thirds_l993_99359


namespace max_distinct_integer_solutions_le_2_l993_99399

def f (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem max_distinct_integer_solutions_le_2 
  (a b c : ℝ) (h₀ : a > 100) :
  ∀ (x : ℤ), |f a b c (x : ℝ)| ≤ 50 → 
  ∃ (x₁ x₂ : ℤ), x = x₁ ∨ x = x₂ :=
by
  sorry

end max_distinct_integer_solutions_le_2_l993_99399


namespace pure_imaginary_condition_fourth_quadrant_condition_l993_99352

theorem pure_imaginary_condition (m : ℝ) (h1: m * (m - 1) = 0) (h2: m ≠ 1) : m = 0 :=
by
  sorry

theorem fourth_quadrant_condition (m : ℝ) (h3: m + 1 > 0) (h4: m^2 - 1 < 0) : -1 < m ∧ m < 1 :=
by
  sorry

end pure_imaginary_condition_fourth_quadrant_condition_l993_99352


namespace inequalities_not_equivalent_l993_99328

theorem inequalities_not_equivalent (x : ℝ) (h1 : x ≠ 1) :
  (x + 3 - (1 / (x - 1)) > -x + 2 - (1 / (x - 1))) ↔ (x + 3 > -x + 2) → False :=
by
  sorry

end inequalities_not_equivalent_l993_99328


namespace cyclic_sum_inequality_l993_99394

theorem cyclic_sum_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 :=
by
  -- TODO: Provide proof here
  sorry

end cyclic_sum_inequality_l993_99394


namespace find_m_for_even_function_l993_99389

def f (x : ℝ) (m : ℝ) := x^2 + (m - 1) * x + 3

theorem find_m_for_even_function : ∃ m : ℝ, (∀ x : ℝ, f (-x) m = f x m) ∧ m = 1 :=
sorry

end find_m_for_even_function_l993_99389


namespace tom_change_l993_99371

theorem tom_change :
  let SNES_value := 150
  let credit_percent := 0.80
  let amount_given := 80
  let game_value := 30
  let NES_sale_price := 160
  let credit_for_SNES := credit_percent * SNES_value
  let amount_to_pay_for_NES := NES_sale_price - credit_for_SNES
  let effective_amount_paid := amount_to_pay_for_NES - game_value
  let change_received := amount_given - effective_amount_paid
  change_received = 70 :=
by
  sorry

end tom_change_l993_99371


namespace power_of_a_point_l993_99306

noncomputable def PA : ℝ := 4
noncomputable def PB : ℝ := 14 + 2 * Real.sqrt 13
noncomputable def PT : ℝ := PB - 8
noncomputable def AB : ℝ := PB - PA

theorem power_of_a_point (PA PB PT : ℝ) (h1 : PA = 4) (h2 : PB = 14 + 2 * Real.sqrt 13) (h3 : PT = PB - 8) : 
  PA * PB = PT * PT :=
by
  rw [h1, h2, h3]
  sorry

end power_of_a_point_l993_99306


namespace geometric_seq_sum_l993_99317

theorem geometric_seq_sum :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    (∀ n, a (n + 1) = a n * q) ∧ 
    (a 4 + a 7 = 2) ∧ 
    (a 5 * a 6 = -8) → 
    a 1 + a 10 = -7 := 
by sorry

end geometric_seq_sum_l993_99317


namespace expression_never_prime_l993_99363

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (n : ℕ) (h : is_prime n) : ¬is_prime (n^2 + 75) :=
sorry

end expression_never_prime_l993_99363


namespace min_value_of_expression_l993_99305

-- Define the conditions in the problem
def conditions (m n : ℝ) : Prop :=
  (2 * m + n = 2) ∧ (m > 0) ∧ (n > 0)

-- Define the problem statement
theorem min_value_of_expression (m n : ℝ) (h : conditions m n) : 
  (∀ m n, conditions m n → (1 / m + 2 / n) ≥ 4) :=
by 
  sorry

end min_value_of_expression_l993_99305


namespace union_of_A_and_B_l993_99318

def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem union_of_A_and_B : A ∪ B = { x | 1 < x ∧ x < 4 } := by
  sorry

end union_of_A_and_B_l993_99318


namespace pavan_travel_time_l993_99332

theorem pavan_travel_time (D : ℝ) (V1 V2 : ℝ) (distance : D = 300) (speed1 : V1 = 30) (speed2 : V2 = 25) : 
  ∃ t : ℝ, t = 11 := 
  by
    sorry

end pavan_travel_time_l993_99332


namespace incorrect_multiplicative_inverse_product_l993_99310

theorem incorrect_multiplicative_inverse_product:
  ∃ (a b : ℝ), a + b = 0 ∧ a * b ≠ 1 :=
by
  sorry

end incorrect_multiplicative_inverse_product_l993_99310


namespace functional_relationship_l993_99369

variable (x y k1 k2 : ℝ)

axiom h1 : y = k1 * x + k2 / (x - 2)
axiom h2 : (y = -1) ↔ (x = 1)
axiom h3 : (y = 5) ↔ (x = 3)

theorem functional_relationship :
  (∀ x y, y = k1 * x + k2 / (x - 2) ∧
    ((x = 1) → y = -1) ∧
    ((x = 3) → y = 5) → y = x + 2 / (x - 2)) :=
by
  sorry

end functional_relationship_l993_99369


namespace find_k_value_l993_99322

theorem find_k_value (k : ℝ) :
  (∃ (x y : ℝ), x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ k = -1/2 := 
by
  sorry

end find_k_value_l993_99322


namespace solve_system_of_equations_l993_99348

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l993_99348


namespace slower_train_speed_l993_99368

-- Define the given conditions
def speed_faster_train : ℝ := 50  -- km/h
def length_faster_train : ℝ := 75.006  -- meters
def passing_time : ℝ := 15  -- seconds

-- Conversion factor
def mps_to_kmph : ℝ := 3.6

-- Define the problem to be proved
theorem slower_train_speed : 
  ∃ speed_slower_train : ℝ, 
    speed_slower_train = speed_faster_train - (75.006 / 15) * mps_to_kmph := 
  by
    exists 31.99856
    sorry

end slower_train_speed_l993_99368


namespace zero_point_exists_between_2_and_3_l993_99325

noncomputable def f (x : ℝ) := 2^(x-1) + x - 5

theorem zero_point_exists_between_2_and_3 :
  ∃ x₀ ∈ Set.Ioo (2 : ℝ) 3, f x₀ = 0 :=
sorry

end zero_point_exists_between_2_and_3_l993_99325


namespace cube_surface_area_l993_99393

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) : 6 * (edge_length * edge_length) = 2400 := by
  -- We state our theorem and assumptions here
  sorry

end cube_surface_area_l993_99393


namespace relation_between_u_and_v_l993_99340

def diameter_circle_condition (AB : ℝ) (r : ℝ) : Prop := AB = 2*r
def chord_tangent_condition (AD BC CD : ℝ) (r : ℝ) : Prop := 
  AD + BC = 2*r ∧ CD*CD = (2*r)*(AD + BC)
def point_selection_condition (AD AF CD : ℝ) : Prop := AD = AF + CD

theorem relation_between_u_and_v (AB AD AF BC CD u v r: ℝ)
  (h1: diameter_circle_condition AB r)
  (h2: chord_tangent_condition AD BC CD r)
  (h3: point_selection_condition AD AF CD)
  (h4: u = AF)
  (h5: v^2 = r^2):
  v^2 = u^3 / (2*r - u) := by
  sorry

end relation_between_u_and_v_l993_99340


namespace kmph_to_mps_l993_99347

theorem kmph_to_mps (s : ℝ) (h : s = 0.975) : s * (1000 / 3600) = 0.2708 := by
  -- We include the assumption s = 0.975 as part of the problem condition.
  -- Import Mathlib to gain access to real number arithmetic.
  -- sorry is added to indicate a place where the proof should go.
  sorry

end kmph_to_mps_l993_99347


namespace root_expression_value_l993_99374

theorem root_expression_value (p q r : ℝ) (hpq : p + q + r = 15) (hpqr : p * q + q * r + r * p = 25) (hpqrs : p * q * r = 10) :
  (p / (2 / p + q * r) + q / (2 / q + r * p) + r / (2 / r + p * q) = 175 / 12) :=
by sorry

end root_expression_value_l993_99374


namespace exists_two_factorizations_in_C_another_number_with_property_l993_99350

def in_set_C (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 1

def is_prime_wrt_C (k : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, in_set_C a ∧ in_set_C b ∧ k = a * b

theorem exists_two_factorizations_in_C : 
  ∃ (a b a' b' : ℕ), 
  in_set_C 4389 ∧ 
  in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
  (4389 = a * b ∧ 4389 = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

theorem another_number_with_property : 
 ∃ (n a b a' b' : ℕ), 
 n ≠ 4389 ∧ in_set_C n ∧ 
 in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
 (n = a * b ∧ n = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

end exists_two_factorizations_in_C_another_number_with_property_l993_99350


namespace percentage_students_left_in_classroom_l993_99367

def total_students : ℕ := 250
def fraction_painting : ℚ := 3 / 10
def fraction_field : ℚ := 2 / 10
def fraction_science : ℚ := 1 / 5

theorem percentage_students_left_in_classroom :
  let gone_painting := total_students * fraction_painting
  let gone_field := total_students * fraction_field
  let gone_science := total_students * fraction_science
  let students_gone := gone_painting + gone_field + gone_science
  let students_left := total_students - students_gone
  (students_left / total_students) * 100 = 30 :=
by sorry

end percentage_students_left_in_classroom_l993_99367


namespace common_difference_is_one_l993_99338

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions given in the problem
axiom h1 : a 1 ^ 2 + a 10 ^ 2 = 101
axiom h2 : a 5 + a 6 = 11
axiom h3 : ∀ n m, n < m → a n < a m
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n+1) = a n + d

-- Theorem stating the common difference d is 1
theorem common_difference_is_one : is_arithmetic_sequence a d → d = 1 := 
by
  sorry

end common_difference_is_one_l993_99338


namespace groom_age_proof_l993_99302

theorem groom_age_proof (G B : ℕ) (h1 : B = G + 19) (h2 : G + B = 185) : G = 83 :=
by
  sorry

end groom_age_proof_l993_99302


namespace find_value_l993_99311

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic : ∀ x : ℝ, f (x + Real.pi) = f x
axiom value_at_neg_pi_third : f (-Real.pi / 3) = 1 / 2

theorem find_value : f (2017 * Real.pi / 3) = 1 / 2 :=
by
  sorry

end find_value_l993_99311


namespace range_of_k_l993_99353

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x < 3 → x - k < 2 * k) → 1 ≤ k :=
by
  sorry

end range_of_k_l993_99353


namespace ratio_john_to_jenna_l993_99301

theorem ratio_john_to_jenna (J : ℕ) 
  (h1 : 100 - J - 40 = 35) : 
  J = 25 ∧ (J / 100 = 1 / 4) := 
by
  sorry

end ratio_john_to_jenna_l993_99301


namespace jemma_total_grasshoppers_l993_99397

def number_of_grasshoppers_on_plant : Nat := 7
def number_of_dozen_baby_grasshoppers : Nat := 2
def number_in_a_dozen : Nat := 12

theorem jemma_total_grasshoppers :
  number_of_grasshoppers_on_plant + number_of_dozen_baby_grasshoppers * number_in_a_dozen = 31 := by
  sorry

end jemma_total_grasshoppers_l993_99397


namespace total_eyes_in_family_l993_99377

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end total_eyes_in_family_l993_99377


namespace mango_coconut_ratio_l993_99373

open Function

theorem mango_coconut_ratio
  (mango_trees : ℕ)
  (coconut_trees : ℕ)
  (total_trees : ℕ)
  (R : ℚ)
  (H1 : mango_trees = 60)
  (H2 : coconut_trees = R * 60 - 5)
  (H3 : total_trees = 85)
  (H4 : total_trees = mango_trees + coconut_trees) :
  R = 1/2 :=
by
  sorry

end mango_coconut_ratio_l993_99373


namespace largest_sample_number_l993_99388

theorem largest_sample_number (n : ℕ) (start interval total : ℕ) (h1 : start = 7) (h2 : interval = 25) (h3 : total = 500) (h4 : n = total / interval) : 
(start + interval * (n - 1) = 482) :=
sorry

end largest_sample_number_l993_99388


namespace confectioner_customers_l993_99324

theorem confectioner_customers (x : ℕ) (h : 0 < x) :
  (49 * (392 / x - 6) = 392) → x = 28 :=
by
sorry

end confectioner_customers_l993_99324


namespace meaningful_square_root_l993_99333

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l993_99333


namespace factorize_expression_l993_99382

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l993_99382


namespace exist_alpha_beta_l993_99300

variables {a b : ℝ} {f : ℝ → ℝ}

-- Assume that f has the Intermediate Value Property (for simplicity, define it as a predicate)
def intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ k ∈ Set.Icc (min (f a) (f b)) (max (f a) (f b)),
    ∃ c ∈ Set.Ioo a b, f c = k

-- Assume the conditions from the problem
variables (h_ivp : intermediate_value_property f a b) (h_sign_change : f a * f b < 0)

-- The theorem we need to prove
theorem exist_alpha_beta (hivp : intermediate_value_property f a b) (hsign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end exist_alpha_beta_l993_99300


namespace solution_correct_l993_99391

variable (a b c d : ℝ)

theorem solution_correct (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end solution_correct_l993_99391


namespace is_equilateral_l993_99380

open Complex

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

-- Assume the conditions of the problem
axiom z1_distinct_z2 : z1 ≠ z2
axiom z2_distinct_z3 : z2 ≠ z3
axiom z3_distinct_z1 : z3 ≠ z1
axiom z1_unit_circle : abs z1 = 1
axiom z2_unit_circle : abs z2 = 1
axiom z3_unit_circle : abs z3 = 1
axiom condition : (1 / (2 + abs (z1 + z2)) + 1 / (2 + abs (z2 + z3)) + 1 / (2 + abs (z3 + z1))) = 1
axiom acute_angled_triangle : sorry

theorem is_equilateral (A B C : ℂ) (hA : A = z1) (hB : B = z2) (hC : C = z3) : 
  (sorry : Prop) := sorry

end is_equilateral_l993_99380


namespace weeks_in_semester_l993_99376

-- Define the conditions and the question as a hypothesis
def annie_club_hours : Nat := 13

theorem weeks_in_semester (w : Nat) (h : 13 * (w - 2) = 52) : w = 6 := by
  sorry

end weeks_in_semester_l993_99376


namespace work_days_of_a_l993_99334

variable (da wa wb wc : ℕ)
variable (hcp : 3 * wc = 5 * wa)
variable (hbw : 4 * wc = 5 * wb)
variable (hwc : wc = 100)
variable (hear : 60 * da + 9 * 80 + 4 * 100 = 1480)

theorem work_days_of_a : da = 6 :=
by
  sorry

end work_days_of_a_l993_99334


namespace train_speed_168_l993_99381

noncomputable def speed_of_train (L : ℕ) (V_man : ℕ) (T : ℕ) : ℚ :=
  let V_man_mps := (V_man * 5) / 18
  let relative_speed := L / T
  let V_train_mps := relative_speed - V_man_mps
  V_train_mps * (18 / 5)

theorem train_speed_168 :
  speed_of_train 500 12 10 = 168 :=
by
  sorry

end train_speed_168_l993_99381


namespace probability_of_getting_specific_clothing_combination_l993_99392

def total_articles := 21

def ways_to_choose_4_articles : ℕ := Nat.choose total_articles 4

def ways_to_choose_2_shirts_from_6 : ℕ := Nat.choose 6 2

def ways_to_choose_1_pair_of_shorts_from_7 : ℕ := Nat.choose 7 1

def ways_to_choose_1_pair_of_socks_from_8 : ℕ := Nat.choose 8 1

def favorable_outcomes := 
  ways_to_choose_2_shirts_from_6 * 
  ways_to_choose_1_pair_of_shorts_from_7 * 
  ways_to_choose_1_pair_of_socks_from_8

def probability := (favorable_outcomes : ℚ) / (ways_to_choose_4_articles : ℚ)

theorem probability_of_getting_specific_clothing_combination : 
  probability = 56 / 399 := by
  sorry

end probability_of_getting_specific_clothing_combination_l993_99392


namespace evaluate_expression_l993_99395

theorem evaluate_expression (x : ℝ) (h1 : x^4 + 2 * x + 2 ≠ 0)
    (h2 : x^4 - 2 * x + 2 ≠ 0) :
    ( ( ( (x + 2) ^ 3 * (x^3 - 2 * x + 2) ^ 3 ) / ( ( x^4 + 2 * x + 2) ) ^ 3 ) ^ 3 * 
      ( ( (x - 2) ^ 3 * ( x^3 + 2 * x + 2 ) ^ 3 ) / ( ( x^4 - 2 * x + 2 ) ) ^ 3 ) ^ 3 ) = 1 :=
by
  sorry

end evaluate_expression_l993_99395


namespace lifespan_histogram_l993_99339

theorem lifespan_histogram :
  (class_interval = 20) →
  (height_vertical_axis_60_80 = 0.03) →
  (total_people = 1000) →
  (number_of_people_60_80 = 600) :=
by
  intro class_interval height_vertical_axis_60_80 total_people
  -- Perform necessary calculations (omitting actual proof as per instructions)
  sorry

end lifespan_histogram_l993_99339


namespace min_value_ineq_l993_99321

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 3 * y = 1) :
  (1 / x) + (1 / (3 * y)) ≥ 4 :=
  sorry

end min_value_ineq_l993_99321


namespace find_m_l993_99390

variables {a1 a2 b1 b2 c1 c2 : ℝ} {m : ℝ}
def vectorA := (3, -2 * m)
def vectorB := (m - 1, 2)
def vectorC := (-2, 1)
def vectorAC := (5, -2 * m - 1)

theorem find_m (h : (5 * (m - 1) + (-2 * m - 1) * 2) = 0) : 
  m = 7 := 
  sorry

end find_m_l993_99390
