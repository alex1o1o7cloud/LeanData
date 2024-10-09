import Mathlib

namespace contractor_fired_two_people_l1308_130846

theorem contractor_fired_two_people
  (total_days : ℕ) (initial_people : ℕ) (days_worked : ℕ) (fraction_completed : ℚ)
  (remaining_days : ℕ) (people_fired : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_people = 10)
  (h3 : days_worked = 20)
  (h4 : fraction_completed = 1/4)
  (h5 : remaining_days = 75)
  (h6 : remaining_days + days_worked = total_days)
  (h7 : people_fired = initial_people - 8) :
  people_fired = 2 :=
  sorry

end contractor_fired_two_people_l1308_130846


namespace abel_arrival_earlier_l1308_130853

variable (distance : ℕ) (speed_abel : ℕ) (speed_alice : ℕ) (start_delay_alice : ℕ)

theorem abel_arrival_earlier (h_dist : distance = 1000) 
                             (h_speed_abel : speed_abel = 50) 
                             (h_speed_alice : speed_alice = 40) 
                             (h_start_delay : start_delay_alice = 1) : 
                             (start_delay_alice + distance / speed_alice) * 60 - (distance / speed_abel) * 60 = 360 :=
by
  sorry

end abel_arrival_earlier_l1308_130853


namespace perpendicular_lines_a_value_l1308_130805

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ m1 m2 : ℝ, (m1 = -a / 2 ∧ m2 = -1 / (a * (a + 1)) ∧ m1 * m2 = -1) ∨
   (a = 0 ∧ ax + 2 * y + 6 = 0 ∧ x + a * (a + 1) * y + (a^2 - 1) = 0)) →
  (a = -3 / 2 ∨ a = 0) :=
by
  sorry

end perpendicular_lines_a_value_l1308_130805


namespace factorize_expression_l1308_130863

theorem factorize_expression (m n : ℤ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end factorize_expression_l1308_130863


namespace ratio_sub_div_a_l1308_130844

theorem ratio_sub_div_a (a b : ℝ) (h : a / b = 5 / 8) : (b - a) / a = 3 / 5 :=
sorry

end ratio_sub_div_a_l1308_130844


namespace log_diff_l1308_130803

theorem log_diff : (Real.log (12:ℝ) / Real.log (2:ℝ)) - (Real.log (3:ℝ) / Real.log (2:ℝ)) = 2 := 
by
  sorry

end log_diff_l1308_130803


namespace find_f_neg_a_l1308_130807

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

-- Given condition
axiom h_fa : f a = 11

-- Statement to prove
theorem find_f_neg_a : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l1308_130807


namespace gcd_g50_g52_l1308_130892

def g (x : ℕ) : ℕ := x^2 - 2 * x + 2021

theorem gcd_g50_g52 : Nat.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g50_g52_l1308_130892


namespace prime_sol_is_7_l1308_130856

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end prime_sol_is_7_l1308_130856


namespace dots_not_visible_l1308_130841

def total_dots (n_dice : ℕ) : ℕ := n_dice * 21

def sum_visible_dots (visible : List ℕ) : ℕ := visible.foldl (· + ·) 0

theorem dots_not_visible (visible : List ℕ) (h : visible = [1, 1, 2, 3, 4, 5, 5, 6]) :
  total_dots 4 - sum_visible_dots visible = 57 :=
by
  rw [total_dots, sum_visible_dots]
  simp
  sorry

end dots_not_visible_l1308_130841


namespace find_sum_A_B_l1308_130809

-- Definitions based on conditions
def A : ℤ := -3 - (-5)
def B : ℤ := 2 + (-2)

-- Theorem statement matching the problem
theorem find_sum_A_B : A + B = 2 :=
sorry

end find_sum_A_B_l1308_130809


namespace marble_ratio_l1308_130800

theorem marble_ratio (total_marbles red_marbles dark_blue_marbles : ℕ) (h_total : total_marbles = 63) (h_red : red_marbles = 38) (h_blue : dark_blue_marbles = 6) :
  (total_marbles - red_marbles - dark_blue_marbles) / red_marbles = 1 / 2 := by
  sorry

end marble_ratio_l1308_130800


namespace fraction_exists_l1308_130897

theorem fraction_exists (n d k : ℕ) (h₁ : n = k * d) (h₂ : d > 0) (h₃ : k > 0) : 
  ∃ (i j : ℕ), i < n ∧ j < n ∧ i + j = n ∧ i/j = d-1 :=
by
  sorry

end fraction_exists_l1308_130897


namespace compare_magnitudes_proof_l1308_130875

noncomputable def compare_magnitudes (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) : Prop :=
  b > c ∧ c > a ∧ b > a

theorem compare_magnitudes_proof (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) :
  compare_magnitudes a b c ha hbc heq :=
sorry

end compare_magnitudes_proof_l1308_130875


namespace scientific_notation_of_taichulight_performance_l1308_130890

noncomputable def trillion := 10^12

def convert_to_scientific_notation (x : ℝ) (n : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ x * 10^n = 12.5 * trillion

theorem scientific_notation_of_taichulight_performance :
  ∃ (x : ℝ) (n : ℤ), convert_to_scientific_notation x n ∧ x = 1.25 ∧ n = 13 :=
by
  unfold convert_to_scientific_notation
  use 1.25
  use 13
  sorry

end scientific_notation_of_taichulight_performance_l1308_130890


namespace number_of_dimes_l1308_130801

theorem number_of_dimes (k : ℕ) (dimes quarters : ℕ) (value : ℕ)
  (h1 : 3 * k = dimes)
  (h2 : 2 * k = quarters)
  (h3 : value = (10 * dimes) + (25 * quarters))
  (h4 : value = 400) :
  dimes = 15 :=
by {
  sorry
}

end number_of_dimes_l1308_130801


namespace greatest_ribbon_length_l1308_130839

-- Define lengths of ribbons
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

-- Condition ensures gcd and prime check
def gcd_is_prime (n : ℕ) : Prop :=
  ∃ d : ℕ, (∀ l ∈ ribbon_lengths, d ∣ l) ∧ Prime d ∧ n = d

-- Prove the greatest length that can make the ribbon pieces, with no ribbon left over, is 2
theorem greatest_ribbon_length : ∃ d, gcd_is_prime d ∧ ∀ m, gcd_is_prime m → m ≤ 2 := 
sorry

end greatest_ribbon_length_l1308_130839


namespace possible_values_of_m_l1308_130826

theorem possible_values_of_m (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end possible_values_of_m_l1308_130826


namespace classroom_students_count_l1308_130891

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end classroom_students_count_l1308_130891


namespace rate_per_kg_grapes_is_70_l1308_130819

-- Let G be the rate per kg for the grapes
def rate_per_kg_grapes (G : ℕ) := G

-- Bruce purchased 8 kg of grapes at rate G per kg
def grapes_cost (G : ℕ) := 8 * G

-- Bruce purchased 11 kg of mangoes at the rate of 55 per kg
def mangoes_cost := 11 * 55

-- Bruce paid a total of 1165 to the shopkeeper
def total_paid := 1165

-- The problem: Prove that the rate per kg for the grapes is 70
theorem rate_per_kg_grapes_is_70 : rate_per_kg_grapes 70 = 70 ∧ grapes_cost 70 + mangoes_cost = total_paid := by
  sorry

end rate_per_kg_grapes_is_70_l1308_130819


namespace combined_value_of_silver_and_gold_l1308_130894

noncomputable def silver_cube_side : ℝ := 3
def silver_weight_per_cubic_inch : ℝ := 6
def silver_price_per_ounce : ℝ := 25
def gold_layer_fraction : ℝ := 0.5
def gold_weight_per_square_inch : ℝ := 0.1
def gold_price_per_ounce : ℝ := 1800
def markup_percentage : ℝ := 1.10

def calculate_combined_value (side weight_per_cubic_inch silver_price layer_fraction weight_per_square_inch gold_price markup : ℝ) : ℝ :=
  let volume := side^3
  let weight_silver := volume * weight_per_cubic_inch
  let value_silver := weight_silver * silver_price
  let surface_area := 6 * side^2
  let area_gold := surface_area * layer_fraction
  let weight_gold := area_gold * weight_per_square_inch
  let value_gold := weight_gold * gold_price
  let total_value_before_markup := value_silver + value_gold
  let selling_price := total_value_before_markup * (1 + markup)
  selling_price

theorem combined_value_of_silver_and_gold :
  calculate_combined_value silver_cube_side silver_weight_per_cubic_inch silver_price_per_ounce gold_layer_fraction gold_weight_per_square_inch gold_price_per_ounce markup_percentage = 18711 :=
by
  sorry

end combined_value_of_silver_and_gold_l1308_130894


namespace integer_coordinates_point_exists_l1308_130899

theorem integer_coordinates_point_exists (p q : ℤ) (h : p^2 - 4 * q = 0) :
  ∃ a b : ℤ, b = a^2 + p * a + q ∧ (a = -p ∧ b = q) ∧ (a ≠ -p → (a = p ∧ b = q) → (p^2 - 4 * b = 0)) :=
by
  sorry

end integer_coordinates_point_exists_l1308_130899


namespace ad_minus_bc_divisible_by_2017_l1308_130857

theorem ad_minus_bc_divisible_by_2017 
  (a b c d n : ℕ) 
  (h1 : (a * n + b) % 2017 = 0) 
  (h2 : (c * n + d) % 2017 = 0) : 
  (a * d - b * c) % 2017 = 0 :=
sorry

end ad_minus_bc_divisible_by_2017_l1308_130857


namespace north_pond_ducks_l1308_130893

-- Definitions based on the conditions
def ducks_lake_michigan : ℕ := 100
def twice_ducks_lake_michigan : ℕ := 2 * ducks_lake_michigan
def additional_ducks : ℕ := 6
def ducks_north_pond : ℕ := twice_ducks_lake_michigan + additional_ducks

-- Theorem to prove the answer
theorem north_pond_ducks : ducks_north_pond = 206 :=
by
  sorry

end north_pond_ducks_l1308_130893


namespace find_a4_l1308_130845

variable (a : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

theorem find_a4 (h₁ : S 5 = 25) (h₂ : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l1308_130845


namespace box_one_contains_at_least_one_ball_l1308_130883

-- Define the conditions
def boxes : List ℕ := [1, 2, 3, 4]
def balls : List ℕ := [1, 2, 3]

-- Define the problem
def count_ways_box_one_contains_ball :=
  let total_ways := (boxes.length)^(balls.length)
  let ways_box_one_empty := (boxes.length - 1)^(balls.length)
  total_ways - ways_box_one_empty

-- The proof problem statement
theorem box_one_contains_at_least_one_ball : count_ways_box_one_contains_ball = 37 := by
  sorry

end box_one_contains_at_least_one_ball_l1308_130883


namespace total_animals_for_sale_l1308_130813

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end total_animals_for_sale_l1308_130813


namespace light_flashes_in_three_quarters_hour_l1308_130821

theorem light_flashes_in_three_quarters_hour (flash_interval seconds_in_three_quarters_hour : ℕ) 
  (h1 : flash_interval = 15) (h2 : seconds_in_three_quarters_hour = 2700) : 
  (seconds_in_three_quarters_hour / flash_interval = 180) :=
by
  sorry

end light_flashes_in_three_quarters_hour_l1308_130821


namespace proof_problem_l1308_130840

open Real

-- Definitions of curves and transformations
def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def C2 := { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 }

-- Parametric equation of C2
def parametric_C2 := ∃ α : ℝ, (0 ≤ α ∧ α ≤ 2*π) ∧
  (C2 = { p : ℝ × ℝ | p.1 = 2 * cos α ∧ p.2 = (1/2) * sin α })

-- Equation of line l1 maximizing the perimeter of ABCD
def line_l1 (p : ℝ × ℝ): Prop :=
  p.2 = (1/4) * p.1

theorem proof_problem : parametric_C2 ∧
  ∀ (A B C D : ℝ × ℝ),
    (A ∈ C2 ∧ B ∈ C2 ∧ C ∈ C2 ∧ D ∈ C2) →
    (line_l1 A ∧ line_l1 B) → 
    (line_l1 A ∧ line_l1 B) ∧
    (line_l1 C ∧ line_l1 D) →
    y = (1 / 4) * x :=
sorry

end proof_problem_l1308_130840


namespace greatest_a_no_integral_solution_l1308_130828

theorem greatest_a_no_integral_solution (a : ℤ) :
  (∀ x : ℤ, |x + 1| ≥ a - 3 / 2) → a = 1 :=
by
  sorry

end greatest_a_no_integral_solution_l1308_130828


namespace books_in_library_final_l1308_130850

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end books_in_library_final_l1308_130850


namespace range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l1308_130836

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def setB : Set ℝ := {x : ℝ | x < -1 ∨ x > 2}

-- Question (1): Proof statement for A ∩ B = ∅ implying 0 ≤ a ≤ 1
theorem range_of_a_if_intersection_empty (a : ℝ) :
  (setA a ∩ setB = ∅) → (0 ≤ a ∧ a ≤ 1) := 
sorry

-- Question (2): Proof statement for A ∪ B = B implying a ≤ -2 or a ≥ 3
theorem range_of_a_if_union_equal_B (a : ℝ) :
  (setA a ∪ setB = setB) → (a ≤ -2 ∨ 3 ≤ a) := 
sorry

end range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l1308_130836


namespace fraction_comparison_l1308_130896

theorem fraction_comparison (a b c d : ℝ) (h1 : a / b < c / d) (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < 1/2 * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l1308_130896


namespace inequality_solution_l1308_130835

theorem inequality_solution (x : ℝ) : 
  x^3 - 10 * x^2 + 28 * x > 0 ↔ (0 < x ∧ x < 4) ∨ (6 < x)
:= sorry

end inequality_solution_l1308_130835


namespace monica_problem_l1308_130852

open Real

noncomputable def completingSquare : Prop :=
  ∃ (b c : ℤ), (∀ x : ℝ, (x - 4) ^ 2 = x^2 - 8 * x + 16) ∧ b = -4 ∧ c = 8 ∧ (b + c = 4)

theorem monica_problem : completingSquare := by
  sorry

end monica_problem_l1308_130852


namespace find_a_l1308_130877

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (x - a)^2 + y^2 = (x^2 + (y-1)^2)) ∧ (¬ ∃ x y : ℝ, y = x + 1) → a = 1 :=
by
  sorry

end find_a_l1308_130877


namespace continuity_at_4_l1308_130815

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end continuity_at_4_l1308_130815


namespace perpendicular_lines_m_value_l1308_130858

def is_perpendicular (m : ℝ) : Prop :=
    let slope1 := 1 / 2
    let slope2 := -2 / m
    slope1 * slope2 = -1

theorem perpendicular_lines_m_value (m : ℝ) (h : is_perpendicular m) : m = 1 := by
    sorry

end perpendicular_lines_m_value_l1308_130858


namespace water_content_in_boxes_l1308_130886

noncomputable def totalWaterInBoxes (num_boxes : ℕ) (bottles_per_box : ℕ) (capacity_per_bottle : ℚ) (fill_fraction : ℚ) : ℚ :=
  num_boxes * bottles_per_box * capacity_per_bottle * fill_fraction

theorem water_content_in_boxes :
  totalWaterInBoxes 10 50 12 (3 / 4) = 4500 := 
by
  sorry

end water_content_in_boxes_l1308_130886


namespace possible_values_f_one_l1308_130822

noncomputable def f (x : ℝ) : ℝ := sorry

variables (a b : ℝ)
axiom f_equation : ∀ x y : ℝ, 
  f ((x - y) ^ 2) = a * (f x)^2 - 2 * x * f y + b * y^2

theorem possible_values_f_one : f 1 = 1 ∨ f 1 = 2 :=
sorry

end possible_values_f_one_l1308_130822


namespace leonardo_needs_more_money_l1308_130832

-- Defining the problem
def cost_of_chocolate : ℕ := 500 -- 5 dollars in cents
def leonardo_own_money : ℕ := 400 -- 4 dollars in cents
def borrowed_money : ℕ := 59 -- borrowed cents

-- Prove that Leonardo needs 41 more cents
theorem leonardo_needs_more_money : (cost_of_chocolate - (leonardo_own_money + borrowed_money) = 41) :=
by
  sorry

end leonardo_needs_more_money_l1308_130832


namespace tree_planting_problem_l1308_130888

noncomputable def total_trees_needed (length width tree_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let intervals := perimeter / tree_distance
  intervals

theorem tree_planting_problem : total_trees_needed 150 60 10 = 42 :=
by
  sorry

end tree_planting_problem_l1308_130888


namespace cos_of_theta_cos_double_of_theta_l1308_130881

noncomputable def theta : ℝ := sorry -- Placeholder for theta within the interval (0, π/2)
axiom theta_in_range : 0 < theta ∧ theta < Real.pi / 2
axiom sin_theta_eq : Real.sin theta = 1/3

theorem cos_of_theta : Real.cos theta = 2 * Real.sqrt 2 / 3 := by
  sorry

theorem cos_double_of_theta : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_of_theta_cos_double_of_theta_l1308_130881


namespace find_triangle_with_properties_l1308_130820

-- Define the angles forming an arithmetic progression
def angles_arithmetic_progression (α β γ : ℝ) : Prop :=
  β - α = γ - β

-- Define the sides forming an arithmetic progression
def sides_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the sides forming a geometric progression
def sides_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Define the sum of angles in a triangle
def sum_of_angles (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- The problem statement:
theorem find_triangle_with_properties 
    (α β γ a b c : ℝ)
    (h1 : angles_arithmetic_progression α β γ)
    (h2 : sum_of_angles α β γ)
    (h3 : sides_arithmetic_progression a b c ∨ sides_geometric_progression a b c) :
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by 
  sorry

end find_triangle_with_properties_l1308_130820


namespace inequality_solution_set_l1308_130869

   theorem inequality_solution_set : 
     {x : ℝ | (4 * x - 5)^2 + (3 * x - 2)^2 < (x - 3)^2} = {x : ℝ | (2 / 3 : ℝ) < x ∧ x < (5 / 4 : ℝ)} :=
   by
     sorry
   
end inequality_solution_set_l1308_130869


namespace opposite_of_neg2023_l1308_130830

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l1308_130830


namespace parabola_focus_coordinates_l1308_130872

theorem parabola_focus_coordinates :
  ∃ h k : ℝ, (y = -1/8 * x^2 + 2 * x - 1) ∧ (h = 8 ∧ k = 5) :=
sorry

end parabola_focus_coordinates_l1308_130872


namespace sum_on_simple_interest_is_1750_l1308_130812

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem sum_on_simple_interest_is_1750 :
  let P_ci := 4000
  let r_ci := 0.10
  let t_ci := 2
  let r_si := 0.08
  let t_si := 3
  let CI := compound_interest P_ci r_ci t_ci
  let SI := CI / 2
  let P_si := SI / (r_si * t_si)
  P_si = 1750 :=
by
  sorry

end sum_on_simple_interest_is_1750_l1308_130812


namespace find_point_A_equidistant_l1308_130837

theorem find_point_A_equidistant :
  ∃ (x : ℝ), (∃ A : ℝ × ℝ × ℝ, A = (x, 0, 0)) ∧
              (∃ B : ℝ × ℝ × ℝ, B = (4, 0, 5)) ∧
              (∃ C : ℝ × ℝ × ℝ, C = (5, 4, 2)) ∧
              (dist (x, 0, 0) (4, 0, 5) = dist (x, 0, 0) (5, 4, 2)) ∧ 
              (x = 2) :=
by
  sorry

end find_point_A_equidistant_l1308_130837


namespace emptying_rate_l1308_130802

theorem emptying_rate (fill_time1 : ℝ) (total_fill_time : ℝ) (T : ℝ) 
  (h1 : fill_time1 = 4) 
  (h2 : total_fill_time = 20) 
  (h3 : 1 / fill_time1 - 1 / T = 1 / total_fill_time) :
  T = 5 :=
by
  sorry

end emptying_rate_l1308_130802


namespace medial_triangle_AB_AC_BC_l1308_130843

theorem medial_triangle_AB_AC_BC
  (l m n : ℝ)
  (A B C : Type)
  (midpoint_BC := (l, 0, 0))
  (midpoint_AC := (0, m, 0))
  (midpoint_AB := (0, 0, n)) :
  (AB^2 + AC^2 + BC^2) / (l^2 + m^2 + n^2) = 8 :=
by
  sorry

end medial_triangle_AB_AC_BC_l1308_130843


namespace trig_identity_l1308_130851

theorem trig_identity (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = 1 / 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := 
sorry

end trig_identity_l1308_130851


namespace sum_of_areas_l1308_130842

theorem sum_of_areas (r s t : ℝ)
  (h1 : r + s = 13)
  (h2 : s + t = 5)
  (h3 : r + t = 12)
  (h4 : t = r / 2) : 
  π * (r ^ 2 + s ^ 2 + t ^ 2) = 105 * π := 
by
  sorry

end sum_of_areas_l1308_130842


namespace factor_exp_l1308_130804

theorem factor_exp (k : ℕ) : 3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 :=
by
  intro h
  sorry

end factor_exp_l1308_130804


namespace return_trip_time_l1308_130864

theorem return_trip_time (d p w : ℝ) (h1 : d = 84 * (p - w)) (h2 : d / (p + w) = d / p - 9) :
  (d / (p + w) = 63) ∨ (d / (p + w) = 12) :=
by
  sorry

end return_trip_time_l1308_130864


namespace expression_value_zero_l1308_130816

theorem expression_value_zero (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) : 
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end expression_value_zero_l1308_130816


namespace max_m_sq_plus_n_sq_l1308_130873

theorem max_m_sq_plus_n_sq (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m*n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_sq_plus_n_sq_l1308_130873


namespace problem_I_problem_II_l1308_130849

-- Definitions
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0
def q (m : ℝ) (x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Problem (I)
theorem problem_I (m : ℝ) : m > 0 → (∀ x : ℝ, q m x → p x) → 0 < m ∧ m ≤ 2 := by
  sorry

-- Problem (II)
theorem problem_II (x : ℝ) : 7 > 0 → 
  (p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x) → 
  (-6 ≤ x ∧ x < -2) ∨ (3 < x ∧ x ≤ 8) := by
  sorry

end problem_I_problem_II_l1308_130849


namespace john_has_leftover_bulbs_l1308_130884

-- Definitions of the problem statements
def initial_bulbs : ℕ := 40
def used_bulbs : ℕ := 16
def remaining_bulbs_after_use : ℕ := initial_bulbs - used_bulbs
def given_to_friend : ℕ := remaining_bulbs_after_use / 2

-- Statement to prove
theorem john_has_leftover_bulbs :
  remaining_bulbs_after_use - given_to_friend = 12 :=
by
  sorry

end john_has_leftover_bulbs_l1308_130884


namespace vector_dot_product_l1308_130887

-- Definitions based on the given conditions
variables (A B C M : ℝ)  -- points in 2D or 3D space can be generalized as real numbers for simplicity
variables (BA BC BM : ℝ) -- vector magnitudes
variables (AC : ℝ) -- magnitude of AC

-- Hypotheses from the problem conditions
variable (hM : 2 * BM = BA + BC)  -- M is the midpoint of AC
variable (hAC : AC = 4)
variable (hBM : BM = 3)

-- Theorem statement asserting the desired result
theorem vector_dot_product :
  BA * BC = 5 :=
by {
  sorry
}

end vector_dot_product_l1308_130887


namespace pandas_bamboo_consumption_l1308_130895

def small_pandas : ℕ := 4
def big_pandas : ℕ := 5
def daily_bamboo_small : ℕ := 25
def daily_bamboo_big : ℕ := 40
def days_in_week : ℕ := 7

theorem pandas_bamboo_consumption : 
  (small_pandas * daily_bamboo_small + big_pandas * daily_bamboo_big) * days_in_week = 2100 := by
  sorry

end pandas_bamboo_consumption_l1308_130895


namespace no_real_solution_for_inequality_l1308_130878

theorem no_real_solution_for_inequality :
  ∀ x : ℝ, ¬(3 * x^2 - x + 2 < 0) :=
by
  sorry

end no_real_solution_for_inequality_l1308_130878


namespace positive_integer_solutions_l1308_130824

theorem positive_integer_solutions
  (x : ℤ) :
  (5 + 3 * x < 13) ∧ ((x + 2) / 3 - (x - 1) / 2 <= 2) →
  (x = 1 ∨ x = 2) :=
by
  sorry

end positive_integer_solutions_l1308_130824


namespace xinxin_nights_at_seaside_l1308_130823

-- Definitions from conditions
def arrival_day : ℕ := 30
def may_days : ℕ := 31
def departure_day : ℕ := 4
def nights_spent : ℕ := (departure_day + (may_days - arrival_day))

-- Theorem to prove the number of nights spent
theorem xinxin_nights_at_seaside : nights_spent = 5 := 
by
  -- Include proof steps here in actual Lean proof
  sorry

end xinxin_nights_at_seaside_l1308_130823


namespace find_constant_k_l1308_130838

theorem find_constant_k (k : ℝ) :
    -x^2 - (k + 9) * x - 8 = -(x - 2) * (x - 4) → k = -15 := by
  sorry

end find_constant_k_l1308_130838


namespace angle_relationship_l1308_130817

-- Define the angles and the relationship
def larger_angle : ℝ := 99
def smaller_angle : ℝ := 81

-- State the problem as a theorem
theorem angle_relationship : larger_angle - smaller_angle = 18 := 
by
  -- The proof would be here
  sorry

end angle_relationship_l1308_130817


namespace rectangle_area_l1308_130862

open Real

theorem rectangle_area (A : ℝ) (s l w : ℝ) (h1 : A = 9 * sqrt 3) (h2 : A = (sqrt 3 / 4) * s^2)
  (h3 : w = s) (h4 : l = 3 * w) : w * l = 108 :=
by
  sorry

end rectangle_area_l1308_130862


namespace trigonometric_identity_proof_l1308_130825

variable (α : ℝ)

theorem trigonometric_identity_proof :
  3 + 4 * (Real.sin (4 * α + (3 / 2) * Real.pi)) +
  Real.sin (8 * α + (5 / 2) * Real.pi) = 
  8 * (Real.sin (2 * α))^4 :=
sorry

end trigonometric_identity_proof_l1308_130825


namespace exam_questions_count_l1308_130859

theorem exam_questions_count (Q S : ℕ) 
    (hS : S = (4 * Q) / 5)
    (sergio_correct : Q - 4 = S + 6) : 
    Q = 50 :=
by 
  sorry

end exam_questions_count_l1308_130859


namespace line_intersects_axes_l1308_130810

theorem line_intersects_axes (a b : ℝ) (x1 y1 x2 y2 : ℝ) (h_points : (x1, y1) = (8, 2) ∧ (x2, y2) = (4, 6)) :
  (∃ x_intercept : ℝ, (x_intercept, 0) = (10, 0)) ∧ (∃ y_intercept : ℝ, (0, y_intercept) = (0, 10)) :=
by
  sorry

end line_intersects_axes_l1308_130810


namespace value_of_expression_l1308_130860

theorem value_of_expression :
  (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 :=
by {
  sorry
}

end value_of_expression_l1308_130860


namespace total_sum_of_grid_is_745_l1308_130866

theorem total_sum_of_grid_is_745 :
  let top_row := [12, 13, 15, 17, 19]
  let left_column := [12, 14, 16, 18]
  let total_sum := 360 + 375 + 10
  total_sum = 745 :=
by
  -- The theorem establishes the total sum calculation.
  sorry

end total_sum_of_grid_is_745_l1308_130866


namespace total_cost_l1308_130814

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end total_cost_l1308_130814


namespace unit_digit_seven_power_500_l1308_130834

def unit_digit (x : ℕ) : ℕ := x % 10

theorem unit_digit_seven_power_500 :
  unit_digit (7 ^ 500) = 1 := 
by
  sorry

end unit_digit_seven_power_500_l1308_130834


namespace exponent_fraction_simplification_l1308_130829

theorem exponent_fraction_simplification :
  (2 ^ 2020 + 2 ^ 2016) / (2 ^ 2020 - 2 ^ 2016) = 17 / 15 :=
by
  sorry

end exponent_fraction_simplification_l1308_130829


namespace solution_set_inequality_l1308_130811

theorem solution_set_inequality (x : ℝ) : (x + 1) * (2 - x) < 0 ↔ x > 2 ∨ x < -1 :=
sorry

end solution_set_inequality_l1308_130811


namespace evaluate_expression_l1308_130889

theorem evaluate_expression : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end evaluate_expression_l1308_130889


namespace Kelly_weight_is_M_l1308_130831

variable (M : ℝ) -- Megan's weight
variable (K : ℝ) -- Kelly's weight
variable (Mike : ℝ) -- Mike's weight

-- Conditions based on the problem statement
def Kelly_less_than_Megan (M K : ℝ) : Prop := K = 0.85 * M
def Mike_greater_than_Megan (M Mike : ℝ) : Prop := Mike = M + 5
def Total_weight_exceeds_bridge (total_weight : ℝ) : Prop := total_weight = 100 + 19
def Total_weight_of_children (M K Mike total_weight : ℝ) : Prop := total_weight = M + K + Mike

theorem Kelly_weight_is_M : (M = 40) → (Total_weight_exceeds_bridge 119) → (Kelly_less_than_Megan M K) → (Mike_greater_than_Megan M Mike) → K = 34 :=
by
  -- Insert proof here
  sorry

end Kelly_weight_is_M_l1308_130831


namespace apples_for_juice_is_correct_l1308_130806

noncomputable def apples_per_year : ℝ := 8 -- 8 million tons
noncomputable def percentage_mixed : ℝ := 0.30 -- 30%
noncomputable def remaining_apples := apples_per_year * (1 - percentage_mixed) -- Apples after mixed
noncomputable def percentage_for_juice : ℝ := 0.60 -- 60%
noncomputable def apples_for_juice := remaining_apples * percentage_for_juice -- Apples for juice

theorem apples_for_juice_is_correct :
  apples_for_juice = 3.36 :=
by
  sorry

end apples_for_juice_is_correct_l1308_130806


namespace derivative_of_y_l1308_130848

variable (a b c x : ℝ)

def y : ℝ := (x - a) * (x - b) * (x - c)

theorem derivative_of_y :
  deriv (fun x:ℝ => (x - a) * (x - b) * (x - c)) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by
  sorry

end derivative_of_y_l1308_130848


namespace intersection_A_B_l1308_130847

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | x^2 - 2 * x < 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  -- We are going to skip the proof for now
  sorry

end intersection_A_B_l1308_130847


namespace sum_first_5n_l1308_130861

theorem sum_first_5n (n : ℕ) (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210) : 
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_l1308_130861


namespace magnitude_correct_l1308_130818

open Real

noncomputable def magnitude_of_vector_addition
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) : ℝ :=
  ‖3 • a + b‖

theorem magnitude_correct 
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) :
  magnitude_of_vector_addition a b theta ha hb h_angle = sqrt 34 :=
sorry

end magnitude_correct_l1308_130818


namespace road_length_kopatych_to_losyash_l1308_130876

variable (T Krosh_dist Yozhik_dist : ℕ)
variable (d_k d_y r_k r_y : ℕ)

theorem road_length_kopatych_to_losyash : 
    (d_k = 20) → (d_y = 16) → (r_k = 30) → (r_y = 60) → 
    (Krosh_dist = 5 * T / 9) → (Yozhik_dist = 4 * T / 9) → 
    (T = Krosh_dist + r_k) →
    (T = Yozhik_dist + r_y) → 
    (T = 180) :=
by
  intros
  sorry

end road_length_kopatych_to_losyash_l1308_130876


namespace ring_area_l1308_130882

theorem ring_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 5) : 
  (π * r1^2) - (π * r2^2) = 119 * π := 
by simp [h1, h2]; sorry

end ring_area_l1308_130882


namespace figure_50_unit_squares_l1308_130833

-- Definitions reflecting the conditions from step A
def f (n : ℕ) := (1/2 : ℚ) * n^3 + (7/2 : ℚ) * n + 1

theorem figure_50_unit_squares : f 50 = 62676 := by
  sorry

end figure_50_unit_squares_l1308_130833


namespace first_guinea_pig_food_l1308_130827

theorem first_guinea_pig_food (x : ℕ) (h1 : ∃ x : ℕ, R = x + 2 * x + (2 * x + 3)) (hp : 13 = x + 2 * x + (2 * x + 3)) : x = 2 :=
by
  sorry

end first_guinea_pig_food_l1308_130827


namespace minimum_value_l1308_130808

theorem minimum_value :
  ∀ (m n : ℝ), m > 0 → n > 0 → (3 * m + n = 1) → (3 / m + 1 / n) ≥ 16 :=
by
  intros m n hm hn hline
  sorry

end minimum_value_l1308_130808


namespace martha_total_clothes_l1308_130867

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l1308_130867


namespace isosceles_trapezoid_area_l1308_130870

theorem isosceles_trapezoid_area (m h : ℝ) (hg : h = 3) (mg : m = 15) : 
  (m * h = 45) :=
by
  simp [hg, mg]
  sorry

end isosceles_trapezoid_area_l1308_130870


namespace determine_placemat_length_l1308_130855

theorem determine_placemat_length :
  ∃ (y : ℝ), ∀ (r : ℝ), r = 5 →
  (∀ (n : ℕ), n = 8 →
  (∀ (w : ℝ), w = 1 →
  y = 10 * Real.sin (5 * Real.pi / 16))) :=
by
  sorry

end determine_placemat_length_l1308_130855


namespace dividend_correct_l1308_130871

-- Given constants for the problem
def divisor := 19
def quotient := 7
def remainder := 6

-- Dividend formula
def dividend := (divisor * quotient) + remainder

-- The proof problem statement
theorem dividend_correct : dividend = 139 := by
  sorry

end dividend_correct_l1308_130871


namespace roots_quadratic_identity_l1308_130874

theorem roots_quadratic_identity :
  ∀ (r s : ℝ), (r^2 - 5 * r + 3 = 0) ∧ (s^2 - 5 * s + 3 = 0) → r^2 + s^2 = 19 :=
by
  intros r s h
  sorry

end roots_quadratic_identity_l1308_130874


namespace range_of_a_l1308_130879

theorem range_of_a {
  a : ℝ
} :
  (∀ x ∈ Set.Ici (2 : ℝ), (x^2 + (2 - a) * x + 4 - 2 * a) > 0) ↔ a < 3 :=
by
  sorry

end range_of_a_l1308_130879


namespace inequality_x_y_z_squares_l1308_130868

theorem inequality_x_y_z_squares (x y z m : ℝ) (h : x + y + z = m) : x^2 + y^2 + z^2 ≥ (m^2) / 3 := by
  sorry

end inequality_x_y_z_squares_l1308_130868


namespace dot_product_property_l1308_130885

-- Definitions based on conditions
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- Required property
theorem dot_product_property : dot_product (vec_add (scalar_mult 2 vec_a) vec_b) vec_a = 6 :=
by sorry

end dot_product_property_l1308_130885


namespace polygon_sides_eq_eleven_l1308_130880

theorem polygon_sides_eq_eleven (n : ℕ) (D : ℕ)
(h1 : D = n + 33)
(h2 : D = n * (n - 3) / 2) :
  n = 11 :=
by {
  sorry
}

end polygon_sides_eq_eleven_l1308_130880


namespace doll_cost_l1308_130865

theorem doll_cost (D : ℝ) (h : 4 * D = 60) : D = 15 :=
by {
  sorry
}

end doll_cost_l1308_130865


namespace single_elimination_games_l1308_130854

theorem single_elimination_games (n : ℕ) (h : n = 128) : (n - 1) = 127 :=
by
  sorry

end single_elimination_games_l1308_130854


namespace number_of_people_in_group_l1308_130898

/-- The number of people in the group N is such that when one of the people weighing 65 kg is replaced
by a new person weighing 100 kg, the average weight of the group increases by 3.5 kg. -/
theorem number_of_people_in_group (N : ℕ) (W : ℝ) 
  (h1 : (W + 35) / N = W / N + 3.5) 
  (h2 : W + 35 = W - 65 + 100) : 
  N = 10 :=
sorry

end number_of_people_in_group_l1308_130898
