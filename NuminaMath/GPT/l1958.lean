import Mathlib

namespace NUMINAMATH_GPT_find_percentage_l1958_195824

noncomputable def percentage (P : ℝ) : Prop :=
  (P / 100) * 1265 / 6 = 354.2

theorem find_percentage : ∃ (P : ℝ), percentage P ∧ P = 168 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1958_195824


namespace NUMINAMATH_GPT_strawberry_blueberry_price_difference_l1958_195879

theorem strawberry_blueberry_price_difference
  (s p t : ℕ → ℕ)
  (strawberries_sold blueberries_sold strawberries_sale_revenue blueberries_sale_revenue strawberries_loss blueberries_loss : ℕ)
  (h1 : strawberries_sold = 54)
  (h2 : strawberries_sale_revenue = 216)
  (h3 : strawberries_loss = 108)
  (h4 : blueberries_sold = 36)
  (h5 : blueberries_sale_revenue = 144)
  (h6 : blueberries_loss = 72)
  (h7 : p strawberries_sold = strawberries_sale_revenue + strawberries_loss)
  (h8 : p blueberries_sold = blueberries_sale_revenue + blueberries_loss)
  : p strawberries_sold / strawberries_sold - p blueberries_sold / blueberries_sold = 0 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_blueberry_price_difference_l1958_195879


namespace NUMINAMATH_GPT_measure_of_angle_BCD_l1958_195804

-- Define angles and sides as given in the problem
variables (α β : ℝ)

-- Conditions: angles and side equalities
axiom angle_ABD_eq_BDC : α = β
axiom angle_DAB_eq_80 : α = 80
axiom side_AB_eq_AD : ∀ AB AD : ℝ, AB = AD
axiom side_DB_eq_DC : ∀ DB DC : ℝ, DB = DC

-- Prove that the measure of angle BCD is 65 degrees
theorem measure_of_angle_BCD : β = 65 :=
sorry

end NUMINAMATH_GPT_measure_of_angle_BCD_l1958_195804


namespace NUMINAMATH_GPT_final_cost_correct_l1958_195805

def dozen_cost : ℝ := 18
def num_dozen : ℝ := 2.5
def discount_rate : ℝ := 0.15

def cost_before_discount : ℝ := num_dozen * dozen_cost
def discount_amount : ℝ := discount_rate * cost_before_discount

def final_cost : ℝ := cost_before_discount - discount_amount

theorem final_cost_correct : final_cost = 38.25 := by
  -- The proof would go here, but we just provide the statement.
  sorry

end NUMINAMATH_GPT_final_cost_correct_l1958_195805


namespace NUMINAMATH_GPT_proof_problem_l1958_195802

open Real

theorem proof_problem :
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 4) →
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 16/4) →
  (∀ x : ℤ, abs x = 4 → abs (-4) = 4) →
  (∀ x : ℤ, x^2 = 16 → (-4)^2 = 16) →
  (- sqrt 16 = -4) := 
by 
  simp
  sorry

end NUMINAMATH_GPT_proof_problem_l1958_195802


namespace NUMINAMATH_GPT_find_cost_price_l1958_195817

theorem find_cost_price 
  (C : ℝ)
  (h1 : 1.10 * C + 110 = 1.15 * C)
  : C = 2200 :=
sorry

end NUMINAMATH_GPT_find_cost_price_l1958_195817


namespace NUMINAMATH_GPT_cubes_with_two_or_three_blue_faces_l1958_195874

theorem cubes_with_two_or_three_blue_faces 
  (four_inch_cube : ℝ)
  (painted_blue_faces : ℝ)
  (one_inch_cubes : ℝ) :
  (four_inch_cube = 4) →
  (painted_blue_faces = 6) →
  (one_inch_cubes = 64) →
  (num_cubes_with_two_or_three_blue_faces = 32) :=
sorry

end NUMINAMATH_GPT_cubes_with_two_or_three_blue_faces_l1958_195874


namespace NUMINAMATH_GPT_tan_alpha_equals_one_l1958_195896

theorem tan_alpha_equals_one (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β))
  : Real.tan α = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_equals_one_l1958_195896


namespace NUMINAMATH_GPT_caleb_ice_cream_l1958_195871

theorem caleb_ice_cream (x : ℕ) (hx1 : ∃ x, x ≥ 0) (hx2 : 4 * x - 36 = 4) : x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_caleb_ice_cream_l1958_195871


namespace NUMINAMATH_GPT_truck_and_trailer_total_weight_l1958_195851

def truck_weight : ℝ := 4800
def trailer_weight (truck_weight : ℝ) : ℝ := 0.5 * truck_weight - 200
def total_weight (truck_weight trailer_weight : ℝ) : ℝ := truck_weight + trailer_weight 

theorem truck_and_trailer_total_weight : 
  total_weight truck_weight (trailer_weight truck_weight) = 7000 :=
by 
  sorry

end NUMINAMATH_GPT_truck_and_trailer_total_weight_l1958_195851


namespace NUMINAMATH_GPT_total_apples_correct_l1958_195826

def craig_initial := 20.5
def judy_initial := 11.25
def dwayne_initial := 17.85
def eugene_to_craig := 7.15
def craig_to_dwayne := 3.5 / 2
def judy_to_sally := judy_initial / 2

def craig_final := craig_initial + eugene_to_craig - craig_to_dwayne
def dwayne_final := dwayne_initial + craig_to_dwayne
def judy_final := judy_initial - judy_to_sally
def sally_final := judy_to_sally

def total_apples := craig_final + judy_final + dwayne_final + sally_final

theorem total_apples_correct : total_apples = 56.75 := by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_total_apples_correct_l1958_195826


namespace NUMINAMATH_GPT_factorize_expression_l1958_195898

theorem factorize_expression (R : Type*) [CommRing R] (m n : R) : 
  m^2 * n - n = n * (m + 1) * (m - 1) := 
sorry

end NUMINAMATH_GPT_factorize_expression_l1958_195898


namespace NUMINAMATH_GPT_polynomial_evaluation_l1958_195840

theorem polynomial_evaluation :
  ∀ x : ℤ, x = -2 → (x^3 + x^2 + x + 1 = -5) :=
by
  intros x hx
  rw [hx]
  norm_num

end NUMINAMATH_GPT_polynomial_evaluation_l1958_195840


namespace NUMINAMATH_GPT_workshopA_more_stable_than_B_l1958_195881

-- Given data sets for workshops A and B
def workshopA_data := [102, 101, 99, 98, 103, 98, 99]
def workshopB_data := [110, 115, 90, 85, 75, 115, 110]

-- Define stability of a product in terms of the standard deviation or similar metric
def is_more_stable (dataA dataB : List ℕ) : Prop :=
  sorry -- Replace with a definition comparing stability based on a chosen metric, e.g., standard deviation

-- Prove that Workshop A's product is more stable than Workshop B's product
theorem workshopA_more_stable_than_B : is_more_stable workshopA_data workshopB_data :=
  sorry

end NUMINAMATH_GPT_workshopA_more_stable_than_B_l1958_195881


namespace NUMINAMATH_GPT_max_value_is_one_l1958_195866

noncomputable def max_value_fraction (x : ℝ) : ℝ :=
  (1 + Real.cos x) / (Real.sin x + Real.cos x + 2)

theorem max_value_is_one : ∃ x : ℝ, max_value_fraction x = 1 := by
  sorry

end NUMINAMATH_GPT_max_value_is_one_l1958_195866


namespace NUMINAMATH_GPT_square_side_length_theorem_l1958_195892

-- Define the properties of the geometric configurations
def is_tangent_to_extension_segments (circle_radius : ℝ) (segment_length : ℝ) : Prop :=
  segment_length = circle_radius

def angle_between_tangents_from_point (angle : ℝ) : Prop :=
  angle = 60 

def square_side_length (side : ℝ) : Prop :=
  side = 4 * (Real.sqrt 2 - 1)

-- Main theorem
theorem square_side_length_theorem (circle_radius : ℝ) (segment_length : ℝ) (angle : ℝ) (side : ℝ)
  (h1 : is_tangent_to_extension_segments circle_radius segment_length)
  (h2 : angle_between_tangents_from_point angle) :
  square_side_length side :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_theorem_l1958_195892


namespace NUMINAMATH_GPT_swan_percentage_not_ducks_l1958_195868

theorem swan_percentage_not_ducks (total_birds geese swans herons ducks : ℝ)
  (h_total : total_birds = 100)
  (h_geese : geese = 0.30 * total_birds)
  (h_swans : swans = 0.20 * total_birds)
  (h_herons : herons = 0.20 * total_birds)
  (h_ducks : ducks = 0.30 * total_birds) :
  (swans / (total_birds - ducks) * 100) = 28.57 :=
by
  sorry

end NUMINAMATH_GPT_swan_percentage_not_ducks_l1958_195868


namespace NUMINAMATH_GPT_third_vs_second_plant_relationship_l1958_195808

-- Define the constants based on the conditions
def first_plant_tomatoes := 24
def second_plant_tomatoes := 12 + 5  -- Half of 24 plus 5
def total_tomatoes := 60

-- Define the production of the third plant based on the total number of tomatoes
def third_plant_tomatoes := total_tomatoes - (first_plant_tomatoes + second_plant_tomatoes)

-- Define the relationship to be proved
theorem third_vs_second_plant_relationship : 
  third_plant_tomatoes = second_plant_tomatoes + 2 :=
by
  -- Proof not provided, adding sorry to skip
  sorry

end NUMINAMATH_GPT_third_vs_second_plant_relationship_l1958_195808


namespace NUMINAMATH_GPT_liz_three_pointers_l1958_195838

-- Define the points scored by Liz's team in the final quarter.
def points_scored_by_liz (free_throws jump_shots three_pointers : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2 + three_pointers * 3

-- Define the points needed to tie the game.
def points_needed_to_tie (initial_deficit points_lost other_team_points : ℕ) : ℕ :=
  points_lost + (initial_deficit - points_lost) + other_team_points

-- The total points scored by Liz from free throws and jump shots.
def liz_regular_points (free_throws jump_shots : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2

theorem liz_three_pointers :
  ∀ (free_throws jump_shots liz_team_deficit_final quarter_deficit other_team_points liz_team_deficit_end final_deficit : ℕ),
    liz_team_deficit_final = 20 →
    free_throws = 5 →
    jump_shots = 4 →
    other_team_points = 10 →
    liz_team_deficit_end = 8 →
    final_deficit = liz_team_deficit_final - liz_team_deficit_end →
    (free_throws * 1 + jump_shots * 2 + 3 * final_deficit) = 
      points_needed_to_tie 20 other_team_points 8 →
    (3 * final_deficit) = 9 →
    final_deficit = 3 →
    final_deficit = 3 :=
by
  intros 
  try sorry

end NUMINAMATH_GPT_liz_three_pointers_l1958_195838


namespace NUMINAMATH_GPT_quadrants_I_and_II_l1958_195822

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y > 3 * x
def condition2 (x y : ℝ) : Prop := y > 6 - x^2

-- Prove that any point satisfying the conditions lies in Quadrant I or II
theorem quadrants_I_and_II (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 6 - x^2) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_quadrants_I_and_II_l1958_195822


namespace NUMINAMATH_GPT_last_ball_probability_l1958_195882

variables (p q : ℕ)

def probability_white_last_ball (p : ℕ) : ℝ :=
  if p % 2 = 0 then 0 else 1

theorem last_ball_probability :
  ∀ {p q : ℕ},
    probability_white_last_ball p = if p % 2 = 0 then 0 else 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_last_ball_probability_l1958_195882


namespace NUMINAMATH_GPT_expression_for_3_diamond_2_l1958_195813

variable {a b : ℝ}

def diamond (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem expression_for_3_diamond_2 (a : ℝ) :
  3 * diamond a 2 = 12 * a - 18 :=
by
  sorry

end NUMINAMATH_GPT_expression_for_3_diamond_2_l1958_195813


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_l1958_195844

theorem smallest_n_for_divisibility (n : ℕ) (h : 2 ∣ 3^(2*n) - 1) (k : ℕ) : n = 2^(2007) := by
  sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_l1958_195844


namespace NUMINAMATH_GPT_range_of_x_for_fx1_positive_l1958_195870

-- Define the conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x
def f_at_2_eq_zero (f : ℝ → ℝ) := f 2 = 0

-- Define the problem statement that needs to be proven
theorem range_of_x_for_fx1_positive (f : ℝ → ℝ) :
  is_even f →
  is_monotonic_decreasing_on_nonneg f →
  f_at_2_eq_zero f →
  ∀ x, f (x - 1) > 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_range_of_x_for_fx1_positive_l1958_195870


namespace NUMINAMATH_GPT_even_number_less_than_its_square_l1958_195857

theorem even_number_less_than_its_square (m : ℕ) (h1 : 2 ∣ m) (h2 : m > 1) : m < m^2 :=
by
sorry

end NUMINAMATH_GPT_even_number_less_than_its_square_l1958_195857


namespace NUMINAMATH_GPT_ellipse_line_intersection_l1958_195800

-- Definitions of the conditions in the Lean 4 language
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

def midpoint_eq (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2 = 1) ∧ (y1 + y2 = -2)

-- The problem statement
theorem ellipse_line_intersection :
  (∃ (l : ℝ → ℝ → Prop),
  (∀ x1 y1 x2 y2 : ℝ, ellipse_eq x1 y1 → ellipse_eq x2 y2 → midpoint_eq x1 y1 x2 y2 →
     l x1 y1 ∧ l x2 y2) ∧
  (∀ x y : ℝ, l x y → (x - 4 * y - 9 / 2 = 0))) :=
sorry

end NUMINAMATH_GPT_ellipse_line_intersection_l1958_195800


namespace NUMINAMATH_GPT_total_spent_on_birthday_presents_l1958_195883

noncomputable def leonards_total_before_discount :=
  (3 * 35.50) + (2 * 120.75) + 44.25

noncomputable def leonards_total_after_discount :=
  leonards_total_before_discount - (0.10 * leonards_total_before_discount)

noncomputable def michaels_total_before_discount :=
  89.50 + (3 * 54.50) + 24.75

noncomputable def michaels_total_after_discount :=
  michaels_total_before_discount - (0.15 * michaels_total_before_discount)

noncomputable def emilys_total_before_tax :=
  (2 * 69.25) + (4 * 14.80)

noncomputable def emilys_total_after_tax :=
  emilys_total_before_tax + (0.08 * emilys_total_before_tax)

noncomputable def total_amount_spent :=
  leonards_total_after_discount + michaels_total_after_discount + emilys_total_after_tax

theorem total_spent_on_birthday_presents :
  total_amount_spent = 802.64 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_on_birthday_presents_l1958_195883


namespace NUMINAMATH_GPT_find_x_of_series_eq_15_l1958_195818

noncomputable def infinite_series (x : ℝ) : ℝ :=
  5 + (5 + x) / 3 + (5 + 2 * x) / 3^2 + (5 + 3 * x) / 3^3 + ∑' n, (5 + (n + 1) * x) / 3 ^ (n + 1)

theorem find_x_of_series_eq_15 (x : ℝ) (h : infinite_series x = 15) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_x_of_series_eq_15_l1958_195818


namespace NUMINAMATH_GPT_speed_ratio_l1958_195890

def distance_to_work := 28
def speed_back := 14
def total_time := 6

theorem speed_ratio 
  (d : ℕ := distance_to_work) 
  (v_2 : ℕ := speed_back) 
  (t : ℕ := total_time) : 
  ∃ v_1 : ℕ, (d / v_1 + d / v_2 = t) ∧ (v_2 / v_1 = 2) :=
by 
  sorry

end NUMINAMATH_GPT_speed_ratio_l1958_195890


namespace NUMINAMATH_GPT_brandon_businesses_l1958_195848

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end NUMINAMATH_GPT_brandon_businesses_l1958_195848


namespace NUMINAMATH_GPT_find_number_l1958_195862

theorem find_number 
  (x : ℝ)
  (h : (258 / 100 * x) / 6 = 543.95) :
  x = 1265 :=
sorry

end NUMINAMATH_GPT_find_number_l1958_195862


namespace NUMINAMATH_GPT_simplify_expression_l1958_195899

theorem simplify_expression (x : ℝ) : (2 * x)^3 + (3 * x) * (x^2) = 11 * x^3 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1958_195899


namespace NUMINAMATH_GPT_no_integer_right_triangle_side_x_l1958_195812

theorem no_integer_right_triangle_side_x :
  ∀ (x : ℤ), (12 + 30 > x ∧ 12 + x > 30 ∧ 30 + x > 12) →
             (12^2 + 30^2 = x^2 ∨ 12^2 + x^2 = 30^2 ∨ 30^2 + x^2 = 12^2) →
             (¬ (∃ x : ℤ, 18 < x ∧ x < 42)) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_right_triangle_side_x_l1958_195812


namespace NUMINAMATH_GPT_find_time_period_l1958_195835

theorem find_time_period (P r CI : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (hP : P = 10000)
  (hr : r = 0.15)
  (hCI : CI = 3886.25)
  (hn : n = 1)
  (hA : A = P + CI)
  (h_formula : A = P * (1 + r / n) ^ (n * t)) : 
  t = 2 := 
  sorry

end NUMINAMATH_GPT_find_time_period_l1958_195835


namespace NUMINAMATH_GPT_farmer_apples_after_giving_l1958_195815

-- Define the initial number of apples and the number of apples given to the neighbor
def initial_apples : ℕ := 127
def given_apples : ℕ := 88

-- Define the expected number of apples after giving some away
def remaining_apples : ℕ := 39

-- Formulate the proof problem
theorem farmer_apples_after_giving : initial_apples - given_apples = remaining_apples := by
  sorry

end NUMINAMATH_GPT_farmer_apples_after_giving_l1958_195815


namespace NUMINAMATH_GPT_employee_total_correct_l1958_195828

variable (total_employees : ℝ)
variable (percentage_female : ℝ)
variable (percentage_male_literate : ℝ)
variable (percentage_total_literate : ℝ)
variable (number_female_literate : ℝ)
variable (percentage_male : ℝ := 1 - percentage_female)

variables (E : ℝ) (CF : ℝ) (M : ℝ) (total_literate : ℝ)

theorem employee_total_correct :
  percentage_female = 0.60 ∧
  percentage_male_literate = 0.50 ∧
  percentage_total_literate = 0.62 ∧
  number_female_literate = 546 ∧
  (total_employees = 1300) :=
by
  -- Change these variables according to the context or find a way to prove this
  let total_employees := 1300
  have Cf := number_female_literate / (percentage_female * total_employees)
  have total_male := percentage_male * total_employees
  have male_literate := percentage_male_literate * total_male
  have total_literate := percentage_total_literate * total_employees

  -- We replace "proof statements" with sorry here
  sorry

end NUMINAMATH_GPT_employee_total_correct_l1958_195828


namespace NUMINAMATH_GPT_find_weight_of_a_l1958_195809

theorem find_weight_of_a (A B C D E : ℕ) 
  (h1 : A + B + C = 3 * 84)
  (h2 : A + B + C + D = 4 * 80)
  (h3 : E = D + 3)
  (h4 : B + C + D + E = 4 * 79) : 
  A = 75 := by
  sorry

end NUMINAMATH_GPT_find_weight_of_a_l1958_195809


namespace NUMINAMATH_GPT_irrational_product_rational_l1958_195806

-- Definitions of irrational and rational for clarity
def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q
def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement of the problem in Lean 4
theorem irrational_product_rational (a b : ℕ) (ha : irrational (Real.sqrt a)) (hb : irrational (Real.sqrt b)) :
  rational ((Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b)) :=
by
  sorry

end NUMINAMATH_GPT_irrational_product_rational_l1958_195806


namespace NUMINAMATH_GPT_find_k_l1958_195801

-- Define the set A using a condition on the quadratic equation
def A (k : ℝ) : Set ℝ := {x | k * x ^ 2 + 4 * x + 4 = 0}

-- Define the condition for the set A to have exactly one element
def has_exactly_one_element (k : ℝ) : Prop :=
  ∃ x : ℝ, A k = {x}

-- The problem statement is to find the value of k for which A has exactly one element
theorem find_k : ∃ k : ℝ, has_exactly_one_element k ∧ k = 1 :=
by
  simp [has_exactly_one_element, A]
  sorry

end NUMINAMATH_GPT_find_k_l1958_195801


namespace NUMINAMATH_GPT_total_tiles_covering_floor_l1958_195827

-- Let n be the width of the rectangle (in tiles)
-- The length would then be 2n (in tiles)
-- The total number of tiles that lie on both diagonals is given as 39

theorem total_tiles_covering_floor (n : ℕ) (H : 2 * n + 1 = 39) : 2 * n^2 = 722 :=
by sorry

end NUMINAMATH_GPT_total_tiles_covering_floor_l1958_195827


namespace NUMINAMATH_GPT_polygon_with_120_degree_interior_angle_has_6_sides_l1958_195816

theorem polygon_with_120_degree_interior_angle_has_6_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (sum_interior_angles : ℕ) = (n-2) * 180 / n ∧ (each_angle : ℕ) = 120) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_120_degree_interior_angle_has_6_sides_l1958_195816


namespace NUMINAMATH_GPT_greatest_value_of_a_plus_b_l1958_195830

-- Definition of the problem conditions
def is_pos_int (n : ℕ) := n > 0

-- Lean statement to prove the greatest possible value of a + b
theorem greatest_value_of_a_plus_b :
  ∃ a b : ℕ, is_pos_int a ∧ is_pos_int b ∧ (1 / (a : ℝ) + 1 / (b : ℝ) = 1 / 9) ∧ a + b = 100 :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_greatest_value_of_a_plus_b_l1958_195830


namespace NUMINAMATH_GPT_beast_of_war_running_time_correct_l1958_195891

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end NUMINAMATH_GPT_beast_of_war_running_time_correct_l1958_195891


namespace NUMINAMATH_GPT_intersection_line_circle_diameter_l1958_195853

noncomputable def length_of_AB : ℝ := 2

theorem intersection_line_circle_diameter 
  (x y : ℝ)
  (h_line : x - 2*y - 1 = 0)
  (h_circle : (x - 1)^2 + y^2 = 1) :
  |(length_of_AB)| = 2 := 
sorry

end NUMINAMATH_GPT_intersection_line_circle_diameter_l1958_195853


namespace NUMINAMATH_GPT_f_g_of_2_eq_4_l1958_195877

def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (x : ℝ) : ℝ := 2*x - 5

theorem f_g_of_2_eq_4 : f (g 2) = 4 := by
  sorry

end NUMINAMATH_GPT_f_g_of_2_eq_4_l1958_195877


namespace NUMINAMATH_GPT_angle_QPR_l1958_195807

theorem angle_QPR (PQ QR PR RS : Real) (angle_PQR angle_PRS : Real) 
  (h1 : PQ = QR) (h2 : PR = RS) (h3 : angle_PQR = 50) (h4 : angle_PRS = 100) : 
  ∃ angle_QPR : Real, angle_QPR = 25 :=
by
  -- We are proving that angle_QPR is 25 given the conditions.
  sorry

end NUMINAMATH_GPT_angle_QPR_l1958_195807


namespace NUMINAMATH_GPT_number_condition_l1958_195845

theorem number_condition (x : ℤ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end NUMINAMATH_GPT_number_condition_l1958_195845


namespace NUMINAMATH_GPT_sum_of_fractions_l1958_195875

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end NUMINAMATH_GPT_sum_of_fractions_l1958_195875


namespace NUMINAMATH_GPT_distinct_diagonals_nonagon_l1958_195865

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end NUMINAMATH_GPT_distinct_diagonals_nonagon_l1958_195865


namespace NUMINAMATH_GPT_valid_decomposition_2009_l1958_195821

/-- A definition to determine whether a number can be decomposed
    into sums of distinct numbers with repeated digits representation. -/
def decomposable_2009 (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  a = 1111 ∧ b = 777 ∧ c = 66 ∧ d = 55 ∧ a + b + c + d = n

theorem valid_decomposition_2009 :
  decomposable_2009 2009 :=
sorry

end NUMINAMATH_GPT_valid_decomposition_2009_l1958_195821


namespace NUMINAMATH_GPT_problem_l1958_195814

noncomputable def d : ℝ := -8.63

theorem problem :
  let floor_d := ⌊d⌋
  let frac_d := d - floor_d
  (3 * floor_d^2 + 20 * floor_d - 67 = 0) ∧
  (4 * frac_d^2 - 15 * frac_d + 5 = 0) → 
  d = -8.63 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l1958_195814


namespace NUMINAMATH_GPT_factorize_expression_l1958_195825

theorem factorize_expression (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) :=
  sorry

end NUMINAMATH_GPT_factorize_expression_l1958_195825


namespace NUMINAMATH_GPT_square_side_length_l1958_195886

theorem square_side_length (s : ℝ) (h : s^2 = 12 * s) : s = 12 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l1958_195886


namespace NUMINAMATH_GPT_canned_boxes_equation_l1958_195819

theorem canned_boxes_equation (x : ℕ) (h₁: x ≤ 300) :
  2 * 14 * x = 32 * (300 - x) :=
by
sorry

end NUMINAMATH_GPT_canned_boxes_equation_l1958_195819


namespace NUMINAMATH_GPT_suzanna_textbooks_page_total_l1958_195884

theorem suzanna_textbooks_page_total :
  let H := 160
  let G := H + 70
  let M := (H + G) / 2
  let S := 2 * H
  let L := (H + G) - 30
  let E := M + L + 25
  H + G + M + S + L + E = 1845 := by
  sorry

end NUMINAMATH_GPT_suzanna_textbooks_page_total_l1958_195884


namespace NUMINAMATH_GPT_determine_d_l1958_195867

theorem determine_d (f g : ℝ → ℝ) (c d : ℝ) (h1 : ∀ x, f x = 5 * x + c) (h2 : ∀ x, g x = c * x + 3) (h3 : ∀ x, f (g x) = 15 * x + d) : d = 18 := 
  sorry

end NUMINAMATH_GPT_determine_d_l1958_195867


namespace NUMINAMATH_GPT_new_number_of_groups_l1958_195846

-- Define the number of students
def total_students : ℕ := 2808

-- Define the initial and new number of groups
def initial_groups (n : ℕ) : ℕ := n + 4
def new_groups (n : ℕ) : ℕ := n

-- Condition: Fewer than 30 students per new group
def fewer_than_30_students_per_group (n : ℕ) : Prop :=
  total_students / n < 30

-- Condition: n and n + 4 must be divisors of total_students
def is_divisor (d : ℕ) (a : ℕ) : Prop :=
  a % d = 0

def valid_group_numbers (n : ℕ) : Prop :=
  is_divisor n total_students ∧ is_divisor (n + 4) total_students ∧ n > 93

-- The main theorem
theorem new_number_of_groups : ∃ n : ℕ, valid_group_numbers n ∧ fewer_than_30_students_per_group n ∧ n = 104 :=
by
  sorry

end NUMINAMATH_GPT_new_number_of_groups_l1958_195846


namespace NUMINAMATH_GPT_find_number_l1958_195872

theorem find_number (x : ℝ) (h : x + (2/3) * x + 1 = 10) : x = 27/5 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1958_195872


namespace NUMINAMATH_GPT_determine_b_l1958_195837

theorem determine_b (b : ℚ) (x y : ℚ) (h1 : x = -3) (h2 : y = 4) (h3 : 2 * b * x + (b + 2) * y = b + 6) :
  b = 2 / 3 := 
sorry

end NUMINAMATH_GPT_determine_b_l1958_195837


namespace NUMINAMATH_GPT_d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l1958_195834

variable (c d : ℕ)

-- Conditions: c is a multiple of 4 and d is a multiple of 8
def is_multiple_of_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k
def is_multiple_of_8 (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k

-- Statements to prove:

-- A. d is a multiple of 4
theorem d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 d :=
sorry

-- B. c - d is a multiple of 4
theorem c_minus_d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 (c - d) :=
sorry

-- D. c - d is a multiple of 2
theorem c_minus_d_is_multiple_of_2 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : ∃ k : ℕ, c - d = 2 * k :=
sorry

end NUMINAMATH_GPT_d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l1958_195834


namespace NUMINAMATH_GPT_sin_alpha_sol_cos_2alpha_pi4_sol_l1958_195810

open Real

-- Define the main problem conditions
def cond1 (α : ℝ) := sin (α + π / 3) + sin α = 9 * sqrt 7 / 14
def range (α : ℝ) := 0 < α ∧ α < π / 3

-- Define the statement for the first problem
theorem sin_alpha_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) : sin α = 2 * sqrt 7 / 7 := 
sorry

-- Define the statement for the second problem
theorem cos_2alpha_pi4_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) (h3 : sin α = 2 * sqrt 7 / 7) : 
  cos (2 * α - π / 4) = (4 * sqrt 6 - sqrt 2) / 14 := 
sorry

end NUMINAMATH_GPT_sin_alpha_sol_cos_2alpha_pi4_sol_l1958_195810


namespace NUMINAMATH_GPT_parabola_directrix_l1958_195876

theorem parabola_directrix (y : ℝ) : 
  x = -((1:ℝ)/4)*y^2 → x = 1 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1958_195876


namespace NUMINAMATH_GPT_angle_sum_proof_l1958_195854

theorem angle_sum_proof (x α β : ℝ) (h1 : 3 * x + 4 * x + α = 180)
 (h2 : α + 5 * x + β = 180)
 (h3 : 2 * x + 2 * x + 6 * x = 180) :
  x = 18 := by
  sorry

end NUMINAMATH_GPT_angle_sum_proof_l1958_195854


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l1958_195803

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l1958_195803


namespace NUMINAMATH_GPT_simplify_fraction_l1958_195873

theorem simplify_fraction :
  ( (5^2010)^2 - (5^2008)^2 ) / ( (5^2009)^2 - (5^2007)^2 ) = 25 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1958_195873


namespace NUMINAMATH_GPT_time_for_A_alone_l1958_195836

variable {W : ℝ}
variable {x : ℝ}

theorem time_for_A_alone (h1 : (W / x) + (W / 24) = W / 12) : x = 24 := 
sorry

end NUMINAMATH_GPT_time_for_A_alone_l1958_195836


namespace NUMINAMATH_GPT_number_of_students_in_third_grade_l1958_195878

theorem number_of_students_in_third_grade
    (total_students : ℕ)
    (sample_size : ℕ)
    (students_first_grade : ℕ)
    (students_second_grade : ℕ)
    (sample_first_and_second : ℕ)
    (students_in_third_grade : ℕ)
    (h1 : total_students = 2000)
    (h2 : sample_size = 100)
    (h3 : sample_first_and_second = students_first_grade + students_second_grade)
    (h4 : students_first_grade = 30)
    (h5 : students_second_grade = 30)
    (h6 : sample_first_and_second = 60)
    (h7 : sample_size - sample_first_and_second = students_in_third_grade)
    (h8 : students_in_third_grade * total_students = 40 * total_students / 100) :
  students_in_third_grade = 800 :=
sorry

end NUMINAMATH_GPT_number_of_students_in_third_grade_l1958_195878


namespace NUMINAMATH_GPT_theta_solutions_count_l1958_195885

theorem theta_solutions_count :
  (∃ (count : ℕ), count = 4 ∧ ∀ θ, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ) = 0 ↔ count = 4) :=
sorry

end NUMINAMATH_GPT_theta_solutions_count_l1958_195885


namespace NUMINAMATH_GPT_square_area_parabola_inscribed_l1958_195858

theorem square_area_parabola_inscribed (s : ℝ) (x y : ℝ) :
  (y = x^2 - 6 * x + 8) ∧
  (s = -2 + 2 * Real.sqrt 5) ∧
  (x = 3 - s / 2 ∨ x = 3 + s / 2) →
  s ^ 2 = 24 - 8 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_square_area_parabola_inscribed_l1958_195858


namespace NUMINAMATH_GPT_find_f_log2_3_l1958_195887

noncomputable def f : ℝ → ℝ := sorry

axiom f_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_condition : ∀ x : ℝ, f (f x + 2 / (2^x + 1)) = (1 / 3)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_f_log2_3_l1958_195887


namespace NUMINAMATH_GPT_three_divides_n_of_invertible_diff_l1958_195832

theorem three_divides_n_of_invertible_diff
  (n : ℕ)
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A * A + B * B = A * B)
  (h2 : Invertible (B * A - A * B)) :
  3 ∣ n :=
sorry

end NUMINAMATH_GPT_three_divides_n_of_invertible_diff_l1958_195832


namespace NUMINAMATH_GPT_age_sum_in_5_years_l1958_195831

variable (MikeAge MomAge : ℕ)
variable (h1 : MikeAge = MomAge - 30)
variable (h2 : MikeAge + MomAge = 70)

theorem age_sum_in_5_years (h1 : MikeAge = MomAge - 30) (h2 : MikeAge + MomAge = 70) :
  (MikeAge + 5) + (MomAge + 5) = 80 := by
  sorry

end NUMINAMATH_GPT_age_sum_in_5_years_l1958_195831


namespace NUMINAMATH_GPT_find_n_that_satisfies_l1958_195849

theorem find_n_that_satisfies :
  ∃ (n : ℕ), (1 / (n + 2 : ℕ) + 2 / (n + 2) + (n + 1) / (n + 2) = 2) ∧ (n = 0) :=
by 
  existsi (0 : ℕ)
  sorry

end NUMINAMATH_GPT_find_n_that_satisfies_l1958_195849


namespace NUMINAMATH_GPT_marble_arrangement_mod_l1958_195893

def num_ways_arrange_marbles (m : ℕ) : ℕ := Nat.choose (m + 3) 3

theorem marble_arrangement_mod (N : ℕ) (m : ℕ) (h1: m = 11) (h2: N = num_ways_arrange_marbles m): 
  N % 1000 = 35 := by
  sorry

end NUMINAMATH_GPT_marble_arrangement_mod_l1958_195893


namespace NUMINAMATH_GPT_fraction_upgraded_l1958_195859

theorem fraction_upgraded :
  ∀ (N U : ℕ), 24 * N = 6 * U → (U : ℚ) / (24 * N + U) = 1 / 7 :=
by
  intros N U h_eq
  sorry

end NUMINAMATH_GPT_fraction_upgraded_l1958_195859


namespace NUMINAMATH_GPT_find_p_l1958_195842

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_find_p_l1958_195842


namespace NUMINAMATH_GPT_Kelly_egg_price_l1958_195856

/-- Kelly has 8 chickens, and each chicken lays 3 eggs per day.
Kelly makes $280 in 4 weeks by selling all the eggs.
We want to prove that Kelly sells a dozen eggs for $5. -/
theorem Kelly_egg_price (chickens : ℕ) (eggs_per_day_per_chicken : ℕ) (earnings_in_4_weeks : ℕ)
  (days_in_4_weeks : ℕ) (eggs_per_dozen : ℕ) (price_per_dozen : ℕ) :
  chickens = 8 →
  eggs_per_day_per_chicken = 3 →
  earnings_in_4_weeks = 280 →
  days_in_4_weeks = 28 →
  eggs_per_dozen = 12 →
  price_per_dozen = earnings_in_4_weeks / ((chickens * eggs_per_day_per_chicken * days_in_4_weeks) / eggs_per_dozen) →
  price_per_dozen = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_Kelly_egg_price_l1958_195856


namespace NUMINAMATH_GPT_complement_union_l1958_195829

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1958_195829


namespace NUMINAMATH_GPT_shorter_piece_length_l1958_195855

def wireLength := 150
def ratioLongerToShorter := 5 / 8

theorem shorter_piece_length : ∃ x : ℤ, x + (5 / 8) * x = wireLength ∧ x = 92 := by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l1958_195855


namespace NUMINAMATH_GPT_johns_final_push_time_l1958_195833

theorem johns_final_push_time :
  ∃ t : ℝ, t = 17 / 4.2 := 
by
  sorry

end NUMINAMATH_GPT_johns_final_push_time_l1958_195833


namespace NUMINAMATH_GPT_coeff_x3_in_expansion_l1958_195860

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion :
  (2 : ℚ)^(4 - 2) * binomial_coeff 4 2 = 24 := by 
  sorry

end NUMINAMATH_GPT_coeff_x3_in_expansion_l1958_195860


namespace NUMINAMATH_GPT_selling_price_of_article_l1958_195864

theorem selling_price_of_article (cost_price gain_percent : ℝ) (h1 : cost_price = 100) (h2 : gain_percent = 30) : 
  cost_price + (gain_percent / 100) * cost_price = 130 := 
by 
  sorry

end NUMINAMATH_GPT_selling_price_of_article_l1958_195864


namespace NUMINAMATH_GPT_perimeter_shaded_region_l1958_195897

theorem perimeter_shaded_region (r: ℝ) (circumference: ℝ) (h1: circumference = 36) (h2: {x // x = 3 * (circumference / 6)}) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_shaded_region_l1958_195897


namespace NUMINAMATH_GPT_polygon_not_hexagon_if_quadrilateral_after_cut_off_l1958_195850

-- Definition of polygonal shape and quadrilateral condition
def is_quadrilateral (sides : Nat) : Prop := sides = 4

-- Definition of polygonal shape with general condition of cutting off one angle
def after_cut_off (original_sides : Nat) (remaining_sides : Nat) : Prop :=
  original_sides > remaining_sides ∧ remaining_sides + 1 = original_sides

-- Problem statement: If a polygon's one angle cut-off results in a quadrilateral, then it is not a hexagon
theorem polygon_not_hexagon_if_quadrilateral_after_cut_off
  (original_sides : Nat) (remaining_sides : Nat) :
  after_cut_off original_sides remaining_sides → is_quadrilateral remaining_sides → original_sides ≠ 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_not_hexagon_if_quadrilateral_after_cut_off_l1958_195850


namespace NUMINAMATH_GPT_curvilinear_quadrilateral_area_l1958_195895

-- Conditions: Define radius R, and plane angles of the tetrahedral angle.
noncomputable def radius (R : Real) : Prop :=
  R > 0

noncomputable def angle (theta : Real) : Prop :=
  theta = 60

-- Establishing the final goal based on the given conditions and solution's correct answer.
theorem curvilinear_quadrilateral_area
  (R : Real)     -- given radius of the sphere
  (hR : radius R) -- the radius of the sphere touching all edges
  (theta : Real)  -- given angle in degrees
  (hθ : angle theta) -- the plane angle of 60 degrees
  :
  ∃ A : Real, 
    A = π * R^2 * (16/3 * (Real.sqrt (2/3)) - 2) := 
  sorry

end NUMINAMATH_GPT_curvilinear_quadrilateral_area_l1958_195895


namespace NUMINAMATH_GPT_scientific_notation_256000_l1958_195839

theorem scientific_notation_256000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 256000 = a * 10^n ∧ a = 2.56 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_256000_l1958_195839


namespace NUMINAMATH_GPT_pairs_count_l1958_195811

theorem pairs_count (A B : Set ℕ) (h1 : A ∪ B = {1, 2, 3, 4, 5}) (h2 : 3 ∈ A ∩ B) : 
  Nat.card {p : Set ℕ × Set ℕ | p.1 ∪ p.2 = {1, 2, 3, 4, 5} ∧ 3 ∈ p.1 ∩ p.2} = 81 := by
  sorry

end NUMINAMATH_GPT_pairs_count_l1958_195811


namespace NUMINAMATH_GPT_milk_production_days_l1958_195852

theorem milk_production_days (x : ℕ) (h : x > 0) :
  let daily_production_per_cow := (x + 1) / (x * (x + 2))
  let total_daily_production := (x + 4) * daily_production_per_cow
  ((x + 7) / total_daily_production) = (x * (x + 2) * (x + 7)) / ((x + 1) * (x + 4)) := 
by
  sorry

end NUMINAMATH_GPT_milk_production_days_l1958_195852


namespace NUMINAMATH_GPT_first_place_points_is_eleven_l1958_195820

/-
Conditions:
1. Points are awarded as follows: first place = x points, second place = 7 points, third place = 5 points, fourth place = 2 points.
2. John participated 7 times in the competition.
3. John finished in each of the top four positions at least once.
4. The product of all the points John received was 38500.
Theorem: The first place winner receives 11 points.
-/

noncomputable def archery_first_place_points (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), -- number of times John finished first, second, third, fourth respectively
    a + b + c + d = 7 ∧ -- condition 2, John participated 7 times
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ -- condition 3, John finished each position at least once
    x ^ a * 7 ^ b * 5 ^ c * 2 ^ d = 38500 -- condition 4, product of all points John received

theorem first_place_points_is_eleven : archery_first_place_points 11 :=
  sorry

end NUMINAMATH_GPT_first_place_points_is_eleven_l1958_195820


namespace NUMINAMATH_GPT_xyz_distinct_real_squares_l1958_195880

theorem xyz_distinct_real_squares (x y z : ℝ) 
  (h1 : x^2 = 2 + y)
  (h2 : y^2 = 2 + z)
  (h3 : z^2 = 2 + x) 
  (h4 : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_xyz_distinct_real_squares_l1958_195880


namespace NUMINAMATH_GPT_solve_system_of_equations_l1958_195889

theorem solve_system_of_equations (x1 x2 x3 x4 x5 y : ℝ) :
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x5 →
  (y = 2 ∧ x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∨
  (y ≠ 2 ∧ (y^2 + y - 1 ≠ 0 ∧ x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (y^2 + y - 1 = 0 ∧ y = (1 / 2) * (-1 + Real.sqrt 5) ∨ y = (1 / 2) * (-1 - Real.sqrt 5) ∧
    ∃ a b : ℝ, x1 = a ∧ x2 = b ∧ x3 = y * b - a ∧ x4 = - y * (a + b) ∧ x5 = y * a - b))
:=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1958_195889


namespace NUMINAMATH_GPT_megan_broke_3_eggs_l1958_195843

variables (total_eggs B C P : ℕ)

theorem megan_broke_3_eggs (h1 : total_eggs = 24) (h2 : C = 2 * B) (h3 : P = 24 - (B + C)) (h4 : P - C = 9) : B = 3 := by
  sorry

end NUMINAMATH_GPT_megan_broke_3_eggs_l1958_195843


namespace NUMINAMATH_GPT_foci_distance_l1958_195823

open Real

-- Defining parameters and conditions
variables (a : ℝ) (b : ℝ) (c : ℝ)
  (F1 F2 A B : ℝ × ℝ) -- Foci and points A, B
  (hyp_cavity : c ^ 2 = a ^ 2 + b ^ 2)
  (perimeters_eq : dist A B = 3 * a ∧ dist A F1 + dist B F1 = dist B F1 + dist B F2 + dist F1 F2)
  (distance_property : dist A F2 - dist A F1 = 2 * a)
  (c_value : c = 2 * a) -- Derived from hyperbolic definition
  
-- Main theorem to prove the distance between foci
theorem foci_distance : dist F1 F2 = 4 * a :=
  sorry

end NUMINAMATH_GPT_foci_distance_l1958_195823


namespace NUMINAMATH_GPT_unique_shell_arrangements_l1958_195863

theorem unique_shell_arrangements : 
  let shells := 12
  let symmetry_ops := 12
  let total_arrangements := Nat.factorial shells
  let distinct_arrangements := total_arrangements / symmetry_ops
  distinct_arrangements = 39916800 := by
  sorry

end NUMINAMATH_GPT_unique_shell_arrangements_l1958_195863


namespace NUMINAMATH_GPT_radius_of_circle_proof_l1958_195888

noncomputable def radius_of_circle (x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : ℝ :=
  r

theorem radius_of_circle_proof (r x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : r = 10 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_proof_l1958_195888


namespace NUMINAMATH_GPT_student_opinion_change_l1958_195841

theorem student_opinion_change (init_enjoy : ℕ) (init_not_enjoy : ℕ)
                               (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  init_enjoy = 40 ∧ init_not_enjoy = 60 ∧ final_enjoy = 75 ∧ final_not_enjoy = 25 →
  ∃ y_min y_max : ℕ, 
    y_min = 35 ∧ y_max = 75 ∧ (y_max - y_min = 40) :=
by
  sorry

end NUMINAMATH_GPT_student_opinion_change_l1958_195841


namespace NUMINAMATH_GPT_quadratic_minimum_value_l1958_195894

theorem quadratic_minimum_value :
  ∀ (x : ℝ), (x - 1)^2 + 2 ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_minimum_value_l1958_195894


namespace NUMINAMATH_GPT_six_digit_numbers_with_zero_l1958_195861

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_numbers_with_zero_l1958_195861


namespace NUMINAMATH_GPT_remainder_of_86_l1958_195847

theorem remainder_of_86 {m : ℕ} (h1 : m ≠ 1) 
  (h2 : 69 % m = 90 % m) (h3 : 90 % m = 125 % m) : 86 % m = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_86_l1958_195847


namespace NUMINAMATH_GPT_probability_of_karnataka_student_l1958_195869

-- Defining the conditions

-- Number of students from each region
def total_students : ℕ := 10
def maharashtra_students : ℕ := 4
def karnataka_students : ℕ := 3
def goa_students : ℕ := 3

-- Number of students to be selected
def students_to_select : ℕ := 4

-- Total ways to choose 4 students out of 10
def C_total : ℕ := Nat.choose total_students students_to_select

-- Ways to select 4 students from the 7 students not from Karnataka
def non_karnataka_students : ℕ := maharashtra_students + goa_students
def C_non_karnataka : ℕ := Nat.choose non_karnataka_students students_to_select

-- Probability calculations
def P_no_karnataka : ℚ := C_non_karnataka / C_total
def P_at_least_one_karnataka : ℚ := 1 - P_no_karnataka

-- The statement to be proved
theorem probability_of_karnataka_student :
  P_at_least_one_karnataka = 5 / 6 :=
sorry

end NUMINAMATH_GPT_probability_of_karnataka_student_l1958_195869
