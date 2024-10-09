import Mathlib

namespace car_speed_conversion_l932_93243

noncomputable def miles_to_yards : ℕ :=
  1760

theorem car_speed_conversion (speed_mph : ℕ) (time_sec : ℝ) (distance_yards : ℕ) :
  speed_mph = 90 →
  time_sec = 0.5 →
  distance_yards = 22 →
  (1 : ℕ) * miles_to_yards = 1760 := by
  intros h1 h2 h3
  sorry

end car_speed_conversion_l932_93243


namespace total_annual_interest_l932_93229

theorem total_annual_interest 
    (principal1 principal2 : ℝ)
    (rate1 rate2 : ℝ)
    (time : ℝ)
    (h1 : principal1 = 26000)
    (h2 : rate1 = 0.08)
    (h3 : principal2 = 24000)
    (h4 : rate2 = 0.085)
    (h5 : time = 1) :
    principal1 * rate1 * time + principal2 * rate2 * time = 4120 := 
sorry

end total_annual_interest_l932_93229


namespace find_side_length_of_cut_out_square_l932_93204

noncomputable def cardboard_box (x : ℝ) : Prop :=
  let length_initial := 80
  let width_initial := 60
  let area_base := 1500
  let length_final := length_initial - 2 * x
  let width_final := width_initial - 2 * x
  length_final * width_final = area_base

theorem find_side_length_of_cut_out_square : ∃ x : ℝ, cardboard_box x ∧ 0 ≤ x ∧ (80 - 2 * x) > 0 ∧ (60 - 2 * x) > 0 ∧ x = 15 :=
by
  sorry

end find_side_length_of_cut_out_square_l932_93204


namespace probability_correct_l932_93222

variable (new_balls old_balls total_balls : ℕ)

-- Define initial conditions
def initial_conditions (new_balls old_balls : ℕ) : Prop :=
  new_balls = 4 ∧ old_balls = 2

-- Define total number of balls in the box
def total_balls_condition (new_balls old_balls total_balls : ℕ) : Prop :=
  total_balls = new_balls + old_balls ∧ total_balls = 6

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of picking one new ball and one old ball
def probability_one_new_one_old (new_balls old_balls total_balls : ℕ) : ℚ :=
  (combination new_balls 1 * combination old_balls 1) / (combination total_balls 2)

-- The theorem to prove the probability
theorem probability_correct (new_balls old_balls total_balls : ℕ)
  (h_initial : initial_conditions new_balls old_balls)
  (h_total : total_balls_condition new_balls old_balls total_balls) :
  probability_one_new_one_old new_balls old_balls total_balls = 8 / 15 := by
  sorry

end probability_correct_l932_93222


namespace curve_self_intersection_l932_93219

def curve_crosses_itself_at_point (x y : ℝ) : Prop :=
∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ (t₁^2 - 4 = x) ∧ (t₁^3 - 6 * t₁ + 7 = y) ∧ (t₂^2 - 4 = x) ∧ (t₂^3 - 6 * t₂ + 7 = y)

theorem curve_self_intersection : curve_crosses_itself_at_point 2 7 :=
sorry

end curve_self_intersection_l932_93219


namespace negation_of_proposition_l932_93255

theorem negation_of_proposition (x : ℝ) (h : 2 * x + 1 ≤ 0) : ¬ (2 * x + 1 ≤ 0) ↔ 2 * x + 1 > 0 := 
by
  sorry

end negation_of_proposition_l932_93255


namespace right_triangle_area_l932_93233

theorem right_triangle_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  (1 / 2 : ℝ) * a * b = 24 := by
  sorry

end right_triangle_area_l932_93233


namespace simplify_trig_expression_l932_93241

open Real

theorem simplify_trig_expression (A : ℝ) (h1 : cos A ≠ 0) (h2 : sin A ≠ 0) :
  (1 - (cos A) / (sin A) + 1 / (sin A)) * (1 + (sin A) / (cos A) - 1 / (cos A)) = -2 * (cos (2 * A) / sin (2 * A)) :=
by
  sorry

end simplify_trig_expression_l932_93241


namespace luke_piles_of_quarters_l932_93258

theorem luke_piles_of_quarters (Q D : ℕ) 
  (h1 : Q = D) -- number of piles of quarters equals number of piles of dimes
  (h2 : 3 * Q + 3 * D = 30) -- total number of coins is 30
  : Q = 5 :=
by
  sorry

end luke_piles_of_quarters_l932_93258


namespace age_ratio_l932_93291

theorem age_ratio (s a : ℕ) (h1 : s - 3 = 2 * (a - 3)) (h2 : s - 7 = 3 * (a - 7)) :
  ∃ x : ℕ, (x = 23) ∧ (s + x) / (a + x) = 3 / 2 :=
by
  sorry

end age_ratio_l932_93291


namespace line_transformation_l932_93249

theorem line_transformation (a b : ℝ)
  (h1 : ∀ x y : ℝ, a * x + y - 7 = 0)
  (A : Matrix (Fin 2) (Fin 2) ℝ) (hA : A = ![![3, 0], ![-1, b]])
  (h2 : ∀ x' y' : ℝ, 9 * x' + y' - 91 = 0) :
  (a = 2) ∧ (b = 13) :=
by
  sorry

end line_transformation_l932_93249


namespace julia_age_after_10_years_l932_93245

-- Define the conditions
def Justin_age : Nat := 26
def Jessica_older_by : Nat := 6
def James_older_by : Nat := 7
def Julia_younger_by : Nat := 8
def years_after : Nat := 10

-- Define the ages now
def Jessica_age_now : Nat := Justin_age + Jessica_older_by
def James_age_now : Nat := Jessica_age_now + James_older_by
def Julia_age_now : Nat := Justin_age - Julia_younger_by

-- Prove that Julia's age after 10 years is 28
theorem julia_age_after_10_years : Julia_age_now + years_after = 28 := by
  sorry

end julia_age_after_10_years_l932_93245


namespace letters_posting_ways_l932_93297

theorem letters_posting_ways :
  let mailboxes := 4
  let letters := 3
  (mailboxes ^ letters) = 64 :=
by
  let mailboxes := 4
  let letters := 3
  show (mailboxes ^ letters) = 64
  sorry

end letters_posting_ways_l932_93297


namespace positive_integer_solutions_l932_93208

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end positive_integer_solutions_l932_93208


namespace min_value_x_y_l932_93205

theorem min_value_x_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) : x + y ≥ 2 :=
sorry

end min_value_x_y_l932_93205


namespace range_of_2x_plus_y_l932_93262

-- Given that positive numbers x and y satisfy this equation:
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x + y + 4 * x * y = 15 / 2

-- Define the range for 2x + y
def range_2x_plus_y (x y : ℝ) : ℝ :=
  2 * x + y

-- State the theorem.
theorem range_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : satisfies_equation x y) :
  3 ≤ range_2x_plus_y x y :=
by
  sorry

end range_of_2x_plus_y_l932_93262


namespace triangle_classification_l932_93215

theorem triangle_classification (a b c : ℕ) (h : a + b + c = 12) :
((
  (a = b ∨ b = c ∨ a = c)  -- Isosceles
  ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)  -- Right-angled
  ∨ (a = b ∧ b = c)  -- Equilateral
)) :=
sorry

end triangle_classification_l932_93215


namespace find_abc_l932_93294

theorem find_abc (a b c : ℝ) (ha : a + 1 / b = 5)
                             (hb : b + 1 / c = 2)
                             (hc : c + 1 / a = 3) :
    a * b * c = 10 + 3 * Real.sqrt 11 :=
sorry

end find_abc_l932_93294


namespace henry_final_money_l932_93269

def initial_money : ℝ := 11.75
def received_from_relatives : ℝ := 18.50
def found_in_card : ℝ := 5.25
def spent_on_game : ℝ := 10.60
def donated_to_charity : ℝ := 3.15

theorem henry_final_money :
  initial_money + received_from_relatives + found_in_card - spent_on_game - donated_to_charity = 21.75 :=
by
  -- proof goes here
  sorry

end henry_final_money_l932_93269


namespace polynomial_operations_l932_93283

-- Define the given options for M, N, and P
def A (x : ℝ) : ℝ := 2 * x - 6
def B (x : ℝ) : ℝ := 3 * x + 5
def C (x : ℝ) : ℝ := -5 * x - 21

-- Define the original expression and its simplified form
def original_expr (M N : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * M x - 3 * N x

-- Define the simplified target expression
def simplified_expr (x : ℝ) : ℝ := -5 * x - 21

theorem polynomial_operations :
  ∀ (M N P : ℝ → ℝ),
  (original_expr M N = simplified_expr) →
  (M = A ∨ N = B ∨ P = C)
:= by
  intros M N P H
  sorry

end polynomial_operations_l932_93283


namespace infinite_series_sum_l932_93217

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) * (n + 1) + 2 * (n + 1) + 1) / ((n + 1) * (n + 2) * (n + 3) * (n + 4))) 
  = 7 / 6 := 
by
  sorry

end infinite_series_sum_l932_93217


namespace mimi_spending_adidas_l932_93238

theorem mimi_spending_adidas
  (total_spending : ℤ)
  (nike_to_adidas_ratio : ℤ)
  (adidas_to_skechers_ratio : ℤ)
  (clothes_spending : ℤ)
  (eq1 : total_spending = 8000)
  (eq2 : nike_to_adidas_ratio = 3)
  (eq3 : adidas_to_skechers_ratio = 5)
  (eq4 : clothes_spending = 2600) :
  ∃ A : ℤ, A + nike_to_adidas_ratio * A + adidas_to_skechers_ratio * A + clothes_spending = total_spending ∧ A = 600 := by
  sorry

end mimi_spending_adidas_l932_93238


namespace can_combine_fig1_can_combine_fig2_l932_93248

-- Given areas for rectangle partitions
variables (S1 S2 S3 S4 : ℝ)
-- Condition: total area of black rectangles equals total area of white rectangles
variable (h1 : S1 + S2 = S3 + S4)

-- Proof problem for Figure 1
theorem can_combine_fig1 : ∃ A : ℝ, S1 + S2 = A ∧ S3 + S4 = A := by
  sorry

-- Proof problem for Figure 2
theorem can_combine_fig2 : ∃ B : ℝ, S1 + S2 = B ∧ S3 + S4 = B := by
  sorry

end can_combine_fig1_can_combine_fig2_l932_93248


namespace net_change_over_week_l932_93273

-- Definitions of initial quantities on Day 1
def baking_powder_day1 : ℝ := 4
def flour_day1 : ℝ := 12
def sugar_day1 : ℝ := 10
def chocolate_chips_day1 : ℝ := 6

-- Definitions of final quantities on Day 7
def baking_powder_day7 : ℝ := 2.5
def flour_day7 : ℝ := 7
def sugar_day7 : ℝ := 6.5
def chocolate_chips_day7 : ℝ := 3.7

-- Definitions of changes in quantities
def change_baking_powder : ℝ := baking_powder_day1 - baking_powder_day7
def change_flour : ℝ := flour_day1 - flour_day7
def change_sugar : ℝ := sugar_day1 - sugar_day7
def change_chocolate_chips : ℝ := chocolate_chips_day1 - chocolate_chips_day7

-- Statement to prove
theorem net_change_over_week : change_baking_powder + change_flour + change_sugar + change_chocolate_chips = 12.3 :=
by
  -- (Proof omitted)
  sorry

end net_change_over_week_l932_93273


namespace find_a_l932_93232

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 3 else 4 / x

theorem find_a (a : ℝ) (h : f a = 2) : a = -1 ∨ a = 2 :=
sorry

end find_a_l932_93232


namespace area_of_square_plot_l932_93211

-- Defining the given conditions and question in Lean 4
theorem area_of_square_plot 
  (cost_per_foot : ℕ := 58)
  (total_cost : ℕ := 2784) :
  ∃ (s : ℕ), (4 * s * cost_per_foot = total_cost) ∧ (s * s = 144) :=
by
  sorry

end area_of_square_plot_l932_93211


namespace find_x_eq_neg15_l932_93250

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l932_93250


namespace dad_contribution_is_correct_l932_93213

noncomputable def carl_savings_weekly : ℕ := 25
noncomputable def savings_duration_weeks : ℕ := 6
noncomputable def coat_cost : ℕ := 170

-- Total savings after 6 weeks
noncomputable def total_savings : ℕ := carl_savings_weekly * savings_duration_weeks

-- Amount used to pay bills in the seventh week
noncomputable def bills_payment : ℕ := total_savings / 3

-- Money left after paying bills
noncomputable def remaining_savings : ℕ := total_savings - bills_payment

-- Amount needed from Dad
noncomputable def dad_contribution : ℕ := coat_cost - remaining_savings

theorem dad_contribution_is_correct : dad_contribution = 70 := by
  sorry

end dad_contribution_is_correct_l932_93213


namespace find_units_digit_l932_93228

def units_digit (n : ℕ) : ℕ := n % 10

theorem find_units_digit :
  units_digit (3 * 19 * 1933 - 3^4) = 0 :=
by
  sorry

end find_units_digit_l932_93228


namespace oranges_picked_l932_93256

theorem oranges_picked (total_oranges second_tree third_tree : ℕ) 
    (h1 : total_oranges = 260) 
    (h2 : second_tree = 60) 
    (h3 : third_tree = 120) : 
    total_oranges - (second_tree + third_tree) = 80 := by 
  sorry

end oranges_picked_l932_93256


namespace prod_ineq_min_value_l932_93298

theorem prod_ineq_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 := by
  sorry

end prod_ineq_min_value_l932_93298


namespace min_max_area_of_CDM_l932_93267

theorem min_max_area_of_CDM (x y z : ℕ) (h1 : 2 * x + y = 4) (h2 : 2 * y + z = 8) :
  z = 4 :=
by
  sorry

end min_max_area_of_CDM_l932_93267


namespace miles_collection_height_l932_93265

-- Definitions based on conditions
def pages_per_inch_miles : ℕ := 5
def pages_per_inch_daphne : ℕ := 50
def daphne_height_inches : ℕ := 25
def longest_collection_pages : ℕ := 1250

-- Theorem to prove the height of Miles's book collection.
theorem miles_collection_height :
  (longest_collection_pages / pages_per_inch_miles) = 250 := by sorry

end miles_collection_height_l932_93265


namespace police_officer_can_catch_gangster_l932_93292

theorem police_officer_can_catch_gangster
  (a : ℝ) -- length of the side of the square
  (v_police : ℝ) -- maximum speed of the police officer
  (v_gangster : ℝ) -- maximum speed of the gangster
  (h_gangster_speed : v_gangster = 2.9 * v_police) :
  ∃ (t : ℝ), t ≥ 0 ∧ (a / (2 * v_police)) = t := sorry

end police_officer_can_catch_gangster_l932_93292


namespace smallest_base_b_l932_93295

theorem smallest_base_b (k : ℕ) (hk : k = 7) : ∃ (b : ℕ), b = 64 ∧ b^k > 4^20 := by
  sorry

end smallest_base_b_l932_93295


namespace jake_reaches_ground_later_by_2_seconds_l932_93268

noncomputable def start_floor : ℕ := 12
noncomputable def steps_per_floor : ℕ := 25
noncomputable def jake_steps_per_second : ℕ := 3
noncomputable def elevator_B_time : ℕ := 90

noncomputable def total_steps_jake := (start_floor - 1) * steps_per_floor
noncomputable def time_jake := (total_steps_jake + jake_steps_per_second - 1) / jake_steps_per_second
noncomputable def time_difference := time_jake - elevator_B_time

theorem jake_reaches_ground_later_by_2_seconds :
  time_difference = 2 := by
  sorry

end jake_reaches_ground_later_by_2_seconds_l932_93268


namespace no_solution_inequality_l932_93272

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_inequality_l932_93272


namespace area_of_triangle_ABC_l932_93275

/--
Given a triangle ABC where BC is 12 cm and the height from A
perpendicular to BC is 15 cm, prove that the area of the triangle is 90 cm^2.
-/
theorem area_of_triangle_ABC (BC : ℝ) (hA : ℝ) (h_BC : BC = 12) (h_hA : hA = 15) : 
  1/2 * BC * hA = 90 := 
sorry

end area_of_triangle_ABC_l932_93275


namespace percentage_increase_l932_93230

theorem percentage_increase (L : ℕ) (h : L + 60 = 240) : 
  ((60:ℝ) / (L:ℝ)) * 100 = 33.33 := 
by
  sorry

end percentage_increase_l932_93230


namespace bobs_improvement_percentage_l932_93252

-- Define the conditions
def bobs_time_minutes := 10
def bobs_time_seconds := 40
def sisters_time_minutes := 10
def sisters_time_seconds := 8

-- Convert minutes and seconds to total seconds
def bobs_total_time_seconds := bobs_time_minutes * 60 + bobs_time_seconds
def sisters_total_time_seconds := sisters_time_minutes * 60 + sisters_time_seconds

-- Define the improvement needed and calculate the percentage improvement
def improvement_needed := bobs_total_time_seconds - sisters_total_time_seconds
def percentage_improvement := (improvement_needed / bobs_total_time_seconds) * 100

-- The lean statement to prove
theorem bobs_improvement_percentage : percentage_improvement = 5 := by
  sorry

end bobs_improvement_percentage_l932_93252


namespace each_tree_takes_one_square_foot_l932_93200

theorem each_tree_takes_one_square_foot (total_length : ℝ) (num_trees : ℕ) (gap_length : ℝ)
    (total_length_eq : total_length = 166) (num_trees_eq : num_trees = 16) (gap_length_eq : gap_length = 10) :
    (total_length - (((num_trees - 1) : ℝ) * gap_length)) / (num_trees : ℝ) = 1 :=
by
  rw [total_length_eq, num_trees_eq, gap_length_eq]
  sorry

end each_tree_takes_one_square_foot_l932_93200


namespace no_such_integers_exists_l932_93299

theorem no_such_integers_exists 
  (a b c d : ℤ) 
  (h1 : a * 19^3 + b * 19^2 + c * 19 + d = 1) 
  (h2 : a * 62^3 + b * 62^2 + c * 62 + d = 2) : 
  false :=
by
  sorry

end no_such_integers_exists_l932_93299


namespace solve_inequality_l932_93231

theorem solve_inequality (x : ℝ) (h : x < 4) : (x - 2) / (x - 4) ≥ 3 := sorry

end solve_inequality_l932_93231


namespace total_ants_found_l932_93212

-- Definitions for the number of ants each child finds
def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2

-- Statement that needs to be proven
theorem total_ants_found : abe_ants + beth_ants + cece_ants + duke_ants = 20 :=
by sorry

end total_ants_found_l932_93212


namespace martin_less_than_43_l932_93254

variable (C K M : ℕ)

-- Conditions
def campbell_correct := C = 35
def kelsey_correct := K = C + 8
def martin_fewer := M < K

-- Conclusion we want to prove
theorem martin_less_than_43 (h1 : campbell_correct C) (h2 : kelsey_correct C K) (h3 : martin_fewer K M) : M < 43 := 
by {
  sorry
}

end martin_less_than_43_l932_93254


namespace both_true_sufficient_but_not_necessary_for_either_l932_93286

variable (p q : Prop)

theorem both_true_sufficient_but_not_necessary_for_either:
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end both_true_sufficient_but_not_necessary_for_either_l932_93286


namespace simplify_exponents_l932_93202

theorem simplify_exponents : (10^0.5) * (10^0.3) * (10^0.2) * (10^0.1) * (10^0.9) = 100 := 
by 
  sorry

end simplify_exponents_l932_93202


namespace trig_identity_l932_93225

theorem trig_identity :
  let s60 := Real.sin (60 * Real.pi / 180)
  let c1 := Real.cos (1 * Real.pi / 180)
  let c20 := Real.cos (20 * Real.pi / 180)
  let s10 := Real.sin (10 * Real.pi / 180)
  s60 * c1 * c20 - s10 = Real.sqrt 3 / 2 - s10 :=
by
  sorry

end trig_identity_l932_93225


namespace one_is_sum_of_others_l932_93288

theorem one_is_sum_of_others {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : |a - b| ≥ c) (h2 : |b - c| ≥ a) (h3 : |c - a| ≥ b) :
    a = b + c ∨ b = a + c ∨ c = a + b :=
sorry

end one_is_sum_of_others_l932_93288


namespace range_of_a_l932_93290

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + 2 * x - 3

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f a x ≤ f a y) ↔ (- (1/4:ℝ) ≤ a ∧ a ≤ 0) := by
  sorry

end range_of_a_l932_93290


namespace dana_jellybeans_l932_93279

noncomputable def jellybeans_in_dana_box (alex_capacity : ℝ) (mul_factor : ℝ) : ℝ :=
  let alex_volume := 1 * 1 * 1.5
  let dana_volume := mul_factor * mul_factor * (mul_factor * 1.5)
  let volume_ratio := dana_volume / alex_volume
  volume_ratio * alex_capacity

theorem dana_jellybeans
  (alex_capacity : ℝ := 150)
  (mul_factor : ℝ := 3) :
  jellybeans_in_dana_box alex_capacity mul_factor = 4050 :=
by
  rw [jellybeans_in_dana_box]
  simp
  sorry

end dana_jellybeans_l932_93279


namespace avg_amount_lost_per_loot_box_l932_93276

-- Define the conditions
def cost_per_loot_box : ℝ := 5
def avg_value_of_items : ℝ := 3.5
def total_amount_spent : ℝ := 40

-- Define the goal
theorem avg_amount_lost_per_loot_box : 
  (total_amount_spent / cost_per_loot_box) * (cost_per_loot_box - avg_value_of_items) / (total_amount_spent / cost_per_loot_box) = 1.5 := 
by 
  sorry

end avg_amount_lost_per_loot_box_l932_93276


namespace only_n_divides_2_n_minus_1_l932_93264

theorem only_n_divides_2_n_minus_1 :
  ∀ n : ℕ, n ≥ 1 → (n ∣ (2^n - 1)) → n = 1 :=
by
  sorry

end only_n_divides_2_n_minus_1_l932_93264


namespace min_value_f_l932_93280

noncomputable def f (x : ℝ) : ℝ := (8^x + 5) / (2^x + 1)

theorem min_value_f : ∃ x : ℝ, f x = 3 :=
sorry

end min_value_f_l932_93280


namespace floral_shop_bouquets_l932_93251

theorem floral_shop_bouquets (T : ℕ) 
  (h1 : 12 + T + T / 3 = 60) 
  (hT : T = 36) : T / 12 = 3 :=
by
  -- Proof steps go here
  sorry

end floral_shop_bouquets_l932_93251


namespace a_n_is_perfect_square_l932_93209

def seqs (a b : ℕ → ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 0 ∧ ∀ n, a (n + 1) = 7 * a n + 6 * b n - 3 ∧ b (n + 1) = 8 * a n + 7 * b n - 4

theorem a_n_is_perfect_square (a b : ℕ → ℤ) (h : seqs a b) :
  ∀ n, ∃ k : ℤ, a n = k^2 :=
by
  sorry

end a_n_is_perfect_square_l932_93209


namespace car_owners_without_motorcycles_l932_93253

theorem car_owners_without_motorcycles (total_adults cars motorcycles no_vehicle : ℕ) 
  (h1 : total_adults = 560) (h2 : cars = 520) (h3 : motorcycles = 80) (h4 : no_vehicle = 10) : 
  cars - (total_adults - no_vehicle - cars - motorcycles) = 470 := 
by
  sorry

end car_owners_without_motorcycles_l932_93253


namespace son_age_is_15_l932_93237

theorem son_age_is_15 (S F : ℕ) (h1 : 2 * S + F = 70) (h2 : 2 * F + S = 95) (h3 : F = 40) :
  S = 15 :=
by {
  sorry
}

end son_age_is_15_l932_93237


namespace incorrect_correlation_statement_l932_93287

/--
  The correlation coefficient measures the degree of linear correlation between two variables. 
  The linear correlation coefficient is a quantity whose absolute value is less than 1. 
  Furthermore, the larger its absolute value, the greater the degree of correlation.

  Let r be the sample correlation coefficient.

  We want to prove that the statement "D: |r| ≥ 1, and the closer |r| is to 1, the greater the degree of correlation" 
  is incorrect.
-/
theorem incorrect_correlation_statement (r : ℝ) (h1 : |r| ≤ 1) : ¬ (|r| ≥ 1) :=
by
  -- Proof steps go here
  sorry

end incorrect_correlation_statement_l932_93287


namespace ratio_of_ages_l932_93266

variable (T N : ℕ)
variable (sum_ages : T = T) -- This is tautological based on the given condition; we can consider it a given sum
variable (age_condition : T - N = 3 * (T - 3 * N))

theorem ratio_of_ages (T N : ℕ) (sum_ages : T = T) (age_condition : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end ratio_of_ages_l932_93266


namespace books_on_desk_none_useful_l932_93221

theorem books_on_desk_none_useful :
  ∃ (answer : String), answer = "none" ∧ 
  (answer = "nothing" ∨ answer = "no one" ∨ answer = "neither" ∨ answer = "none")
  → answer = "none"
:= by
  sorry

end books_on_desk_none_useful_l932_93221


namespace range_a_satisfies_l932_93220

theorem range_a_satisfies (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, f x = x^3) (h₂ : f 2 = 8) :
  (f (a - 3) > f (1 - a)) ↔ a > 2 :=
by
  sorry

end range_a_satisfies_l932_93220


namespace min_value_frac_l932_93289

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) :
  (1 / x + 1 / (3 * y)) = 4 :=
by
  sorry

end min_value_frac_l932_93289


namespace solve_porters_transportation_l932_93261

variable (x : ℝ)

def porters_transportation_equation : Prop :=
  (5000 / x = 8000 / (x + 600))

theorem solve_porters_transportation (x : ℝ) (h₁ : 600 > 0) (h₂ : x > 0):
  porters_transportation_equation x :=
sorry

end solve_porters_transportation_l932_93261


namespace problem1_problem2_l932_93227

theorem problem1 : -20 - (-8) + (-4) = -16 := by
  sorry

theorem problem2 : -1^3 * (-2)^2 / (4 / 3 : ℚ) + |5 - 8| = 0 := by
  sorry

end problem1_problem2_l932_93227


namespace different_picture_size_is_correct_l932_93278

-- Define constants and conditions
def memory_card_picture_capacity := 3000
def single_picture_size := 8
def different_picture_capacity := 4000

-- Total memory card capacity in megabytes
def total_capacity := memory_card_picture_capacity * single_picture_size

-- The size of each different picture
def different_picture_size := total_capacity / different_picture_capacity

-- The theorem to prove
theorem different_picture_size_is_correct :
  different_picture_size = 6 := 
by
  -- We include 'sorry' here to bypass actual proof
  sorry

end different_picture_size_is_correct_l932_93278


namespace min_large_trucks_needed_l932_93216

-- Define the parameters for the problem
def total_fruit : ℕ := 134
def load_large_truck : ℕ := 15
def load_small_truck : ℕ := 7

-- Define the main theorem to be proved
theorem min_large_trucks_needed :
  ∃ (n : ℕ), n = 8 ∧ (total_fruit = n * load_large_truck + 2 * load_small_truck) :=
by sorry

end min_large_trucks_needed_l932_93216


namespace cube_inequality_l932_93207

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l932_93207


namespace tan_315_eq_neg_one_l932_93271

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l932_93271


namespace min_value_x1_x2_frac1_x1x2_l932_93203

theorem min_value_x1_x2_frac1_x1x2 (a x1 x2 : ℝ) (ha : a > 2) (h_sum : x1 + x2 = a) (h_prod : x1 * x2 = a - 2) :
  x1 + x2 + 1 / (x1 * x2) ≥ 4 :=
sorry

end min_value_x1_x2_frac1_x1x2_l932_93203


namespace opposite_points_l932_93270

theorem opposite_points (A B : ℝ) (h1 : A = -B) (h2 : A < B) (h3 : abs (A - B) = 6.4) : A = -3.2 ∧ B = 3.2 :=
by
  sorry

end opposite_points_l932_93270


namespace computation_of_sqrt_expr_l932_93274

theorem computation_of_sqrt_expr : 
  (Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549) := 
by
  sorry

end computation_of_sqrt_expr_l932_93274


namespace Eldora_total_cost_l932_93285

-- Conditions
def paper_clip_cost : ℝ := 1.85
def index_card_cost : ℝ := 3.95 -- from Finn's purchase calculation
def total_cost (clips : ℝ) (cards : ℝ) (clip_price : ℝ) (card_price : ℝ) : ℝ :=
  (clips * clip_price) + (cards * card_price)

theorem Eldora_total_cost :
  total_cost 15 7 paper_clip_cost index_card_cost = 55.40 :=
by
  sorry

end Eldora_total_cost_l932_93285


namespace remainder_ab_cd_l932_93218

theorem remainder_ab_cd (n : ℕ) (hn: n > 0) (a b c d : ℤ) 
  (hac : a * c ≡ 1 [ZMOD n]) (hbd : b * d ≡ 1 [ZMOD n]) : 
  (a * b + c * d) % n = 2 :=
by
  sorry

end remainder_ab_cd_l932_93218


namespace initial_flour_amount_l932_93259

theorem initial_flour_amount (initial_flour : ℕ) (additional_flour : ℕ) (total_flour : ℕ) 
  (h1 : additional_flour = 4) (h2 : total_flour = 16) (h3 : initial_flour + additional_flour = total_flour) :
  initial_flour = 12 := 
by 
  sorry

end initial_flour_amount_l932_93259


namespace dan_picked_l932_93223

-- Definitions:
def benny_picked : Nat := 2
def total_picked : Nat := 11

-- Problem statement:
theorem dan_picked (b : Nat) (t : Nat) (d : Nat) (h1 : b = benny_picked) (h2 : t = total_picked) (h3 : t = b + d) : d = 9 := by
  sorry

end dan_picked_l932_93223


namespace sufficient_but_not_necessary_l932_93224

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1 / a^2) : a^2 > 1 / a ∧ ¬ ∀ a, a^2 > 1 / a → a > 1 / a^2 :=
by
  sorry

end sufficient_but_not_necessary_l932_93224


namespace estimated_total_fish_l932_93293

-- Let's define the conditions first
def total_fish_marked := 100
def second_catch_total := 200
def marked_in_second_catch := 5

-- The variable representing the total number of fish in the pond
variable (x : ℕ)

-- The theorem stating that given the conditions, the total number of fish is 4000
theorem estimated_total_fish
  (h1 : total_fish_marked = 100)
  (h2 : second_catch_total = 200)
  (h3 : marked_in_second_catch = 5)
  (h4 : (marked_in_second_catch : ℝ) / second_catch_total = (total_fish_marked : ℝ) / x) :
  x = 4000 := 
sorry

end estimated_total_fish_l932_93293


namespace jessica_balloons_l932_93226

-- Defining the number of blue balloons Joan, Sally, and the total number.
def balloons_joan : ℕ := 9
def balloons_sally : ℕ := 5
def balloons_total : ℕ := 16

-- The statement to prove that Jessica has 2 blue balloons
theorem jessica_balloons : balloons_total - (balloons_joan + balloons_sally) = 2 :=
by
  -- Using the given information and arithmetic, we can show the main statement
  sorry

end jessica_balloons_l932_93226


namespace Sara_has_8_balloons_l932_93210

theorem Sara_has_8_balloons (Tom_balloons Sara_balloons total_balloons : ℕ)
  (htom : Tom_balloons = 9)
  (htotal : Tom_balloons + Sara_balloons = 17) :
  Sara_balloons = 8 :=
by
  sorry

end Sara_has_8_balloons_l932_93210


namespace six_people_with_A_not_on_ends_l932_93246

-- Define the conditions and the problem statement
def standing_arrangements (n : ℕ) (A : Type) :=
  {l : List A // l.length = n}

theorem six_people_with_A_not_on_ends : 
  (arr : standing_arrangements 6 ℕ) → 
  (∀ a ∈ arr.val, a ≠ 0 ∧ a ≠ 5) → 
  ∃! (total_arrangements : ℕ), total_arrangements = 480 :=
  by
    sorry

end six_people_with_A_not_on_ends_l932_93246


namespace common_rational_root_l932_93235

-- Definitions for the given conditions
def polynomial1 (a b c : ℤ) (x : ℚ) := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16 = 0
def polynomial2 (d e f g : ℤ) (x : ℚ) := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50 = 0

-- The proof problem statement: Given the conditions, proving that -1/2 is a common rational root
theorem common_rational_root (a b c d e f g : ℤ) (k : ℚ) 
  (h1 : polynomial1 a b c k)
  (h2 : polynomial2 d e f g k) 
  (h3 : ∃ m n : ℤ, k = -((m : ℚ) / n) ∧ Int.gcd m n = 1) :
  k = -1/2 :=
sorry

end common_rational_root_l932_93235


namespace fg_equals_seven_l932_93247

def g (x : ℤ) : ℤ := x * x
def f (x : ℤ) : ℤ := 2 * x - 1

theorem fg_equals_seven : f (g 2) = 7 := by
  sorry

end fg_equals_seven_l932_93247


namespace fraction_value_l932_93239

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end fraction_value_l932_93239


namespace sum_of_brothers_ages_l932_93214

theorem sum_of_brothers_ages (Bill Eric: ℕ) 
  (h1: 4 = Bill - Eric) 
  (h2: Bill = 16) : 
  Bill + Eric = 28 := 
by 
  sorry

end sum_of_brothers_ages_l932_93214


namespace max_value_of_perfect_sequence_l932_93240

def isPerfectSequence (c : ℕ → ℕ) : Prop := ∀ n m : ℕ, 1 ≤ m ∧ m ≤ (Finset.range (n + 1)).sum (fun k => c k) → 
  ∃ (a : ℕ → ℕ), m = (Finset.range (n + 1)).sum (fun k => c k / a k)

theorem max_value_of_perfect_sequence (n : ℕ) : 
  ∃ c : ℕ → ℕ, isPerfectSequence c ∧
    (∀ i, i ≤ n → c i ≤ if i = 1 then 2 else 4 * 3^(i - 2)) ∧
    c n = if n = 1 then 2 else 4 * 3^(n - 2) :=
by
  sorry

end max_value_of_perfect_sequence_l932_93240


namespace find_3a_plus_4b_l932_93257

noncomputable def g (x : ℝ) := 3 * x - 6

noncomputable def f_inverse (x : ℝ) := (3 * x - 2) / 2

noncomputable def f (x : ℝ) (a b : ℝ) := a * x + b

theorem find_3a_plus_4b (a b : ℝ) (h1 : ∀ x, g x = 2 * f_inverse x - 4) (h2 : ∀ x, f_inverse (f x a b) = x) :
  3 * a + 4 * b = 14 / 3 :=
sorry

end find_3a_plus_4b_l932_93257


namespace student_correct_answers_l932_93201

variable (C I : ℕ)

theorem student_correct_answers :
  C + I = 100 ∧ C - 2 * I = 76 → C = 92 :=
by
  intros h
  sorry

end student_correct_answers_l932_93201


namespace train_speed_180_kmph_l932_93260

def train_speed_in_kmph (length_meters : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_m_per_s := length_meters / time_seconds
  let speed_km_per_h := speed_m_per_s * 36 / 10
  speed_km_per_h

theorem train_speed_180_kmph:
  train_speed_in_kmph 400 8 = 180 := by
  sorry

end train_speed_180_kmph_l932_93260


namespace cos_5_theta_l932_93244

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end cos_5_theta_l932_93244


namespace selling_price_eq_120_l932_93236

-- Definitions based on the conditions
def cost_price : ℝ := 96
def profit_percentage : ℝ := 0.25

-- The proof statement
theorem selling_price_eq_120 (cost_price : ℝ) (profit_percentage : ℝ) : cost_price = 96 → profit_percentage = 0.25 → (cost_price + cost_price * profit_percentage) = 120 :=
by
  intros hcost hprofit
  rw [hcost, hprofit]
  sorry

end selling_price_eq_120_l932_93236


namespace word_value_at_l932_93296

def letter_value (c : Char) : ℕ :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1 else 0

def word_value (s : String) : ℕ :=
  let sum_values := s.toList.map letter_value |>.sum
  sum_values * s.length

theorem word_value_at : word_value "at" = 42 := by
  sorry

end word_value_at_l932_93296


namespace find_m_l932_93284

theorem find_m (x y m : ℝ) (opp_sign: y = -x) 
  (h1 : 4 * x + 2 * y = 3 * m) 
  (h2 : 3 * x + y = m + 2) : 
  m = 1 :=
by 
  -- Placeholder for the steps to prove the theorem
  sorry

end find_m_l932_93284


namespace probability_is_correct_l932_93277

def num_red : ℕ := 7
def num_green : ℕ := 9
def num_yellow : ℕ := 10
def num_blue : ℕ := 5
def num_purple : ℕ := 3

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue + num_purple

def num_blue_or_purple : ℕ := num_blue + num_purple

-- Probability of selecting a blue or purple jelly bean
def probability_blue_or_purple : ℚ := num_blue_or_purple / total_jelly_beans

theorem probability_is_correct :
  probability_blue_or_purple = 4 / 17 := sorry

end probability_is_correct_l932_93277


namespace discount_difference_l932_93281

theorem discount_difference (x : ℝ) (h1 : x = 8000) : 
  (x * 0.7) - ((x * 0.8) * 0.9) = 160 :=
by
  rw [h1]
  sorry

end discount_difference_l932_93281


namespace range_of_a_l932_93234

theorem range_of_a (a : ℝ) :  (5 - a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
by
  intro h
  sorry

end range_of_a_l932_93234


namespace quadratic_root_exists_l932_93282

theorem quadratic_root_exists (a b c : ℝ) (ha : a ≠ 0)
  (h1 : a * (0.6 : ℝ)^2 + b * 0.6 + c = -0.04)
  (h2 : a * (0.7 : ℝ)^2 + b * 0.7 + c = 0.19) :
  ∃ x : ℝ, 0.6 < x ∧ x < 0.7 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_exists_l932_93282


namespace total_distance_karl_drove_l932_93263

theorem total_distance_karl_drove :
  ∀ (consumption_rate miles_per_gallon : ℕ) 
    (tank_capacity : ℕ) 
    (initial_gas : ℕ) 
    (distance_leg1 : ℕ) 
    (purchased_gas : ℕ) 
    (remaining_gas : ℕ)
    (final_gas : ℕ),
  consumption_rate = 25 → 
  tank_capacity = 18 →
  initial_gas = 12 →
  distance_leg1 = 250 →
  purchased_gas = 10 →
  remaining_gas = initial_gas - distance_leg1 / consumption_rate + purchased_gas →
  final_gas = remaining_gas - distance_leg2 / consumption_rate →
  remaining_gas - distance_leg2 / consumption_rate = final_gas →
  distance_leg2 = (initial_gas - remaining_gas + purchased_gas - final_gas) * miles_per_gallon →
  miles_per_gallon = 25 →
  distance_leg2 + distance_leg1 = 475 :=
sorry

end total_distance_karl_drove_l932_93263


namespace geometric_sequence_sixth_term_l932_93206

theorem geometric_sequence_sixth_term (a b : ℚ) (h : a = 3 ∧ b = -1/2) : 
  (a * (b / a) ^ 5) = -1/2592 :=
by
  sorry

end geometric_sequence_sixth_term_l932_93206


namespace candies_left_after_carlos_ate_l932_93242

def num_red_candies : ℕ := 50
def num_yellow_candies : ℕ := 3 * num_red_candies - 35
def num_blue_candies : ℕ := (2 * num_yellow_candies) / 3
def num_green_candies : ℕ := 20
def num_purple_candies : ℕ := num_green_candies / 2
def num_silver_candies : ℕ := 10
def num_candies_eaten_by_carlos : ℕ := num_yellow_candies + num_green_candies / 2

def total_candies : ℕ := num_red_candies + num_yellow_candies + num_blue_candies + num_green_candies + num_purple_candies + num_silver_candies
def candies_remaining : ℕ := total_candies - num_candies_eaten_by_carlos

theorem candies_left_after_carlos_ate : candies_remaining = 156 := by
  sorry

end candies_left_after_carlos_ate_l932_93242
