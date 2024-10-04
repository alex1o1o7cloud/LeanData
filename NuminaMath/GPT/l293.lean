import Mathlib

namespace solve_inequality_l293_293365

theorem solve_inequality :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 8) ∨ (10 < x)).

end solve_inequality_l293_293365


namespace union_of_A_and_B_l293_293669

def setA : Set ℝ := {x | 2 * x - 1 > 0}
def setB : Set ℝ := {x | abs x < 1}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | x > -1} := 
by {
  sorry
}

end union_of_A_and_B_l293_293669


namespace profit_calculation_l293_293282

variable (price : ℕ) (cost : ℕ) (exchange_rate : ℕ) (profit_per_bottle : ℚ)

-- Conditions
def conditions := price = 2 ∧ cost = 1 ∧ exchange_rate = 5

-- Profit per bottle is 0.66 yuan considering the exchange policy
theorem profit_calculation (h : conditions price cost exchange_rate) : profit_per_bottle = 0.66 := sorry

end profit_calculation_l293_293282


namespace quadratic_properties_l293_293655

noncomputable def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ)
  (root_neg1 : quadratic a b c (-1) = 0)
  (ineq_condition : ∀ x : ℝ, (quadratic a b c x - x) * (quadratic a b c x - (x^2 + 1) / 2) ≤ 0) :
  quadratic a b c 1 = 1 ∧ ∀ x : ℝ, quadratic a b c x = (1 / 4) * x^2 + (1 / 2) * x + (1 / 4) :=
by
  sorry

end quadratic_properties_l293_293655


namespace example_theorem_l293_293062

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l293_293062


namespace computer_price_after_six_years_l293_293896

def price_decrease (p_0 : ℕ) (rate : ℚ) (t : ℕ) : ℚ :=
  p_0 * rate ^ (t / 2)

theorem computer_price_after_six_years :
  price_decrease 8100 (2 / 3) 6 = 2400 := by
  sorry

end computer_price_after_six_years_l293_293896


namespace electricity_usage_l293_293389

theorem electricity_usage 
  (total_usage : ℕ) (saved_cost : ℝ) (initial_cost : ℝ) (peak_cost : ℝ) (off_peak_cost : ℝ) 
  (usage_peak : ℕ) (usage_off_peak : ℕ) :
  total_usage = 100 →
  saved_cost = 3 →
  initial_cost = 0.55 →
  peak_cost = 0.6 →
  off_peak_cost = 0.4 →
  usage_peak + usage_off_peak = total_usage →
  (total_usage * initial_cost - (peak_cost * usage_peak + off_peak_cost * usage_off_peak) = saved_cost) →
  usage_peak = 60 ∧ usage_off_peak = 40 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end electricity_usage_l293_293389


namespace pentagon_PTRSQ_area_proof_l293_293511

-- Define the geometric setup and properties
def quadrilateral_PQRS_is_square (P Q R S T : Type) : Prop :=
  -- Here, we will skip the precise geometric construction and assume the properties directly.
  sorry

def segment_PT_perpendicular_to_TR (P T R : Type) : Prop :=
  sorry

def PT_eq_5 (PT : ℝ) : Prop :=
  PT = 5

def TR_eq_12 (TR : ℝ) : Prop :=
  TR = 12

def area_PTRSQ (area : ℝ) : Prop :=
  area = 139

theorem pentagon_PTRSQ_area_proof
  (P Q R S T : Type)
  (PQRS_is_square : quadrilateral_PQRS_is_square P Q R S T)
  (PT_perpendicular_TR : segment_PT_perpendicular_to_TR P T R)
  (PT_length : PT_eq_5 5)
  (TR_length : TR_eq_12 12)
  : area_PTRSQ 139 :=
  sorry

end pentagon_PTRSQ_area_proof_l293_293511


namespace partition_of_X_l293_293076

noncomputable theory

-- Define the set X for a given positive integer n
def X (n : ℕ) (hn : n ≥ 3) : Set ℕ := {x | x ∈ Finset.range (n^2 - n) ∧ x > 0}

-- Define the partition problem
theorem partition_of_X (n : ℕ) (hn : n ≥ 3) :
  ∃ (S T : Set ℕ), 
    S ∪ T = X n hn ∧
    S ∩ T = ∅ ∧
    (∀ a1 a2 a3 ... an ∈ S, a1 < a2 < ... < an → ∃ k ∈ Finset.range (n - 2), a_k > (a_{k-1} + a_{k+1}) / 2) ∧
    (∀ a1 a2 a3 ... an ∈ T, a1 < a2 < ... < an → ∃ k ∈ Finset.range (n - 2), a_k > (a_{k-1} + a_{k+1}) / 2) :=
sorry

end partition_of_X_l293_293076


namespace determine_x_l293_293952

variable {x y : ℝ}

theorem determine_x (h : (x - 1) / x = (y^3 + 3 * y^2 - 4) / (y^3 + 3 * y^2 - 5)) : 
  x = y^3 + 3 * y^2 - 5 := 
sorry

end determine_x_l293_293952


namespace mean_books_read_l293_293173

theorem mean_books_read :
  let readers1 := 4
  let books1 := 3
  let readers2 := 5
  let books2 := 5
  let readers3 := 2
  let books3 := 7
  let readers4 := 1
  let books4 := 10
  let total_readers := readers1 + readers2 + readers3 + readers4
  let total_books := (readers1 * books1) + (readers2 * books2) + (readers3 * books3) + (readers4 * books4)
  let mean_books := total_books / total_readers
  mean_books = 5.0833 :=
by
  sorry

end mean_books_read_l293_293173


namespace reciprocal_sum_l293_293321

variable {x y z a b c : ℝ}

-- The function statement where we want to show the equivalence.
theorem reciprocal_sum (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxy : (x * y) / (x - y) = a)
  (hxz : (x * z) / (x - z) = b)
  (hyz : (y * z) / (y - z) = c) :
  (1/x + 1/y + 1/z) = ((1/a + 1/b + 1/c) / 2) :=
sorry

end reciprocal_sum_l293_293321


namespace arrangement_count_l293_293293

-- Given conditions
def num_basketballs : ℕ := 5
def num_volleyballs : ℕ := 3
def num_footballs : ℕ := 2
def total_balls : ℕ := num_basketballs + num_volleyballs + num_footballs

-- Way to calculate the permutations of multiset
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Proof statement
theorem arrangement_count : 
  factorial total_balls / (factorial num_basketballs * factorial num_volleyballs * factorial num_footballs) = 2520 :=
by
  sorry

end arrangement_count_l293_293293


namespace cos_180_eq_neg1_l293_293617

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l293_293617


namespace sum_of_integers_from_1_to_10_l293_293208

theorem sum_of_integers_from_1_to_10 :
  (Finset.range 11).sum id = 55 :=
sorry

end sum_of_integers_from_1_to_10_l293_293208


namespace complement_union_in_set_l293_293087

open Set

theorem complement_union_in_set {U A B : Set ℕ} 
  (hU : U = {1, 3, 5, 9}) 
  (hA : A = {1, 3, 9}) 
  (hB : B = {1, 9}) : 
  (U \ (A ∪ B)) = {5} := 
  by sorry

end complement_union_in_set_l293_293087


namespace right_triangle_area_l293_293755

theorem right_triangle_area (h b : ℝ) (hypotenuse : h = 5) (base : b = 3) :
  ∃ a : ℝ, a = 1 / 2 * b * (Real.sqrt (h^2 - b^2)) ∧ a = 6 := 
by
  sorry

end right_triangle_area_l293_293755


namespace football_team_throwers_l293_293508

theorem football_team_throwers
    (total_players : ℕ)
    (right_handed_players : ℕ)
    (one_third : ℚ)
    (number_throwers : ℕ)
    (number_non_throwers : ℕ)
    (right_handed_non_throwers : ℕ)
    (left_handed_non_throwers : ℕ)
    (h1 : total_players = 70)
    (h2 : right_handed_players = 63)
    (h3 : one_third = 1 / 3)
    (h4 : number_non_throwers = total_players - number_throwers)
    (h5 : right_handed_non_throwers = right_handed_players - number_throwers)
    (h6 : left_handed_non_throwers = one_third * number_non_throwers)
    (h7 : 2 * left_handed_non_throwers = right_handed_non_throwers)
    : number_throwers = 49 := 
by
  sorry

end football_team_throwers_l293_293508


namespace Mary_regular_hourly_rate_l293_293689

theorem Mary_regular_hourly_rate (R : ℝ) (h1 : ∃ max_hours : ℝ, max_hours = 70)
  (h2 : ∀ hours: ℝ, hours ≤ 70 → (hours ≤ 20 → earnings = hours * R) ∧ (hours > 20 → earnings = 20 * R + (hours - 20) * 1.25 * R))
  (h3 : ∀ max_earning: ℝ, max_earning = 660)
  : R = 8 := 
sorry

end Mary_regular_hourly_rate_l293_293689


namespace cos_180_eq_neg_one_l293_293599

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l293_293599


namespace children_difference_l293_293279

theorem children_difference (initial_count : ℕ) (remaining_count : ℕ) (difference : ℕ) 
  (h1 : initial_count = 41) (h2 : remaining_count = 18) :
  difference = initial_count - remaining_count := 
by
  sorry

end children_difference_l293_293279


namespace point_A_in_second_quadrant_l293_293837

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l293_293837


namespace solve_for_x_l293_293468

theorem solve_for_x (x y z : ℝ) 
  (h1 : x * y + 3 * x + 2 * y = 12) 
  (h2 : y * z + 5 * y + 3 * z = 15) 
  (h3 : x * z + 5 * x + 4 * z = 40) :
  x = 4 :=
by
  sorry

end solve_for_x_l293_293468


namespace regular_polygons_cover_plane_l293_293550

theorem regular_polygons_cover_plane (n : ℕ) (h_n_ge_3 : 3 ≤ n)
    (h_angle_eq : ∀ n, (180 * (1 - (2 / n)) : ℝ) = (internal_angle : ℝ))
    (h_summation_eq : ∃ k : ℕ, k * internal_angle = 360) :
    n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_polygons_cover_plane_l293_293550


namespace combined_savings_after_four_weeks_l293_293357

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l293_293357


namespace museum_paintings_discarded_l293_293765

def initial_paintings : ℕ := 2500
def percentage_to_discard : ℝ := 0.35
def paintings_discarded : ℝ := initial_paintings * percentage_to_discard

theorem museum_paintings_discarded : paintings_discarded = 875 :=
by
  -- Lean automatically simplifies this using basic arithmetic rules
  sorry

end museum_paintings_discarded_l293_293765


namespace evaluate_polynomial_at_5_l293_293268

def polynomial (x : ℕ) : ℕ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem evaluate_polynomial_at_5 : polynomial 5 = 7548 := by
  sorry

end evaluate_polynomial_at_5_l293_293268


namespace num_divisible_by_10_l293_293875

theorem num_divisible_by_10 (a b d : ℕ) (h1 : 100 ≤ a) (h2 : a ≤ 500) (h3 : 100 ≤ b) (h4 : b ≤ 500) (h5 : Nat.gcd d 10 = 10) :
  (b - a) / d + 1 = 41 := by
  sorry

end num_divisible_by_10_l293_293875


namespace solve_system_equations_l293_293687

variable (x y z : ℝ)

theorem solve_system_equations (h1 : 3 * x = 20 + (20 - x))
    (h2 : y = 2 * x - 5)
    (h3 : z = Real.sqrt (x + 4)) :
  x = 10 ∧ y = 15 ∧ z = Real.sqrt 14 :=
by
  sorry

end solve_system_equations_l293_293687


namespace parabola_distance_to_focus_l293_293650

theorem parabola_distance_to_focus (P : ℝ × ℝ) (y_axis_dist : ℝ) (hx : P.1 = 4) (hy : P.2 ^ 2 = 32) :
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 36 :=
by {
  sorry
}

end parabola_distance_to_focus_l293_293650


namespace circle_through_points_eq_l293_293678

noncomputable def circle_eqn (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_through_points_eq {h k r : ℝ} :
  circle_eqn h k r (-1) 0 ∧
  circle_eqn h k r 0 2 ∧
  circle_eqn h k r 2 0 → 
  (h = 2 / 3 ∧ k = 2 / 3 ∧ r^2 = 29 / 9) :=
sorry

end circle_through_points_eq_l293_293678


namespace emily_furniture_assembly_time_l293_293416

-- Definitions based on conditions
def chairs := 4
def tables := 2
def time_per_piece := 8

-- Proof statement
theorem emily_furniture_assembly_time : (chairs + tables) * time_per_piece = 48 :=
by
  sorry

end emily_furniture_assembly_time_l293_293416


namespace rocco_total_usd_l293_293127

noncomputable def total_usd_quarters : ℝ := 40 * 0.25
noncomputable def total_usd_nickels : ℝ := 90 * 0.05

noncomputable def cad_to_usd : ℝ := 0.8
noncomputable def eur_to_usd : ℝ := 1.18
noncomputable def gbp_to_usd : ℝ := 1.4

noncomputable def total_cad_dimes : ℝ := 60 * 0.10 * 0.8
noncomputable def total_eur_cents : ℝ := 50 * 0.01 * 1.18
noncomputable def total_gbp_pence : ℝ := 30 * 0.01 * 1.4

noncomputable def total_usd : ℝ :=
  total_usd_quarters + total_usd_nickels + total_cad_dimes +
  total_eur_cents + total_gbp_pence

theorem rocco_total_usd : total_usd = 20.31 := sorry

end rocco_total_usd_l293_293127


namespace total_pages_is_905_l293_293374

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l293_293374


namespace aquarium_counts_l293_293588

theorem aquarium_counts :
  ∃ (O S L : ℕ), O + S = 7 ∧ L + S = 6 ∧ O + L = 5 ∧ (O ≤ S ∧ O ≤ L) ∧ O = 5 ∧ S = 7 ∧ L = 6 :=
by
  sorry

end aquarium_counts_l293_293588


namespace smallest_integer_solution_l293_293156

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end smallest_integer_solution_l293_293156


namespace limsup_subset_l293_293115

variable {Ω : Type*} -- assuming a universal sample space Ω for the events A_n and B_n

def limsup (A : ℕ → Set Ω) : Set Ω := 
  ⋂ k, ⋃ n ≥ k, A n

theorem limsup_subset {A B : ℕ → Set Ω} (h : ∀ n, A n ⊆ B n) : 
  limsup A ⊆ limsup B :=
by
  -- here goes the proof
  sorry

end limsup_subset_l293_293115


namespace fraction_power_equiv_l293_293193

theorem fraction_power_equiv : (75000^4) / (25000^4) = 81 := by
  sorry

end fraction_power_equiv_l293_293193


namespace sole_mart_meals_l293_293944

theorem sole_mart_meals (c_c_meals : ℕ) (meals_given_away : ℕ) (meals_left : ℕ)
  (h1 : c_c_meals = 113) (h2 : meals_givenAway = 85) (h3 : meals_left = 78)  :
  ∃ m : ℕ, m + c_c_meals = meals_givenAway + meals_left ∧ m = 50 := 
by
  sorry

end sole_mart_meals_l293_293944


namespace quadratic_polynomial_properties_l293_293959

theorem quadratic_polynomial_properties :
  ∃ k : ℝ, (k * (3 - (3+4*I)) = 8 ∧ 
            (∀ x : ℂ, (x = (3 + 4 * I) → polynomial.eval x (k * (X - (3+4*I)) * (X - (3-4*I))) = 0)) ∧ 
            polynomial.coeff (k * (X - (3+4*I)) * (X - (3-4*I))) 1 = 8) :=
sorry

end quadratic_polynomial_properties_l293_293959


namespace crayons_eaten_correct_l293_293679

variable (initial_crayons final_crayons : ℕ)

def crayonsEaten (initial_crayons final_crayons : ℕ) : ℕ :=
  initial_crayons - final_crayons

theorem crayons_eaten_correct : crayonsEaten 87 80 = 7 :=
  by
  sorry

end crayons_eaten_correct_l293_293679


namespace find_M_l293_293670

variable (M : ℕ)

theorem find_M (h : (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M) : M = 1003 :=
sorry

end find_M_l293_293670


namespace teresa_total_marks_l293_293718

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end teresa_total_marks_l293_293718


namespace intersects_x_axis_at_one_point_l293_293381

theorem intersects_x_axis_at_one_point (a : ℝ) :
  (∃ x, ax^2 + (a-3)*x + 1 = 0) ∧ (∀ x₁ x₂, ax^2 + (a-3)*x + 1 = 0 → x₁ = x₂) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersects_x_axis_at_one_point_l293_293381


namespace children_of_exceptions_l293_293243

theorem children_of_exceptions (x y : ℕ) (h : 6 * x + 2 * y = 58) (hx : x = 8) : y = 5 :=
by
  sorry

end children_of_exceptions_l293_293243


namespace complex_quadrant_l293_293480

theorem complex_quadrant (z : ℂ) (h : (2 - I) * z = 1 + I) : 
  0 < z.re ∧ 0 < z.im := 
by 
  -- Proof will be provided here 
  sorry

end complex_quadrant_l293_293480


namespace B_is_left_of_A_l293_293406

-- Define the coordinates of points A and B
def A_coord : ℚ := 5 / 8
def B_coord : ℚ := 8 / 13

-- The statement we want to prove: B is to the left of A
theorem B_is_left_of_A : B_coord < A_coord :=
  by {
    sorry
  }

end B_is_left_of_A_l293_293406


namespace max_children_l293_293426

/-- Total quantities -/
def total_apples : ℕ := 55
def total_cookies : ℕ := 114
def total_chocolates : ℕ := 83

/-- Leftover quantities after distribution -/
def leftover_apples : ℕ := 3
def leftover_cookies : ℕ := 10
def leftover_chocolates : ℕ := 5

/-- Distributed quantities -/
def distributed_apples : ℕ := total_apples - leftover_apples
def distributed_cookies : ℕ := total_cookies - leftover_cookies
def distributed_chocolates : ℕ := total_chocolates - leftover_chocolates

/-- The theorem states the maximum number of children -/
theorem max_children : Nat.gcd (Nat.gcd distributed_apples distributed_cookies) distributed_chocolates = 26 :=
by
  sorry

end max_children_l293_293426


namespace middle_angle_range_l293_293154

theorem middle_angle_range (α β γ : ℝ) (h₀: α + β + γ = 180) (h₁: 0 < α) (h₂: 0 < β) (h₃: 0 < γ) (h₄: α ≤ β) (h₅: β ≤ γ) : 
  0 < β ∧ β < 90 :=
by
  sorry

end middle_angle_range_l293_293154


namespace divisible_by_900_l293_293509

theorem divisible_by_900 (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) := 
by 
  sorry

end divisible_by_900_l293_293509


namespace part_one_part_two_l293_293977

noncomputable def f (x a : ℝ) : ℝ :=
  |x + a| + 2 * |x - 1|

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 2 :=
sorry

theorem part_two (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : ∀ x : ℝ, 1 ≤ x → x ≤ 2 → f x a > x^2 - b + 1) : 
  (a + 1 / 2)^2 + (b + 1 / 2)^2 > 2 :=
sorry

end part_one_part_two_l293_293977


namespace cos_180_proof_l293_293621

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l293_293621


namespace total_books_equals_45_l293_293189

-- Define the number of books bought in each category
def adventure_books : ℝ := 13.0
def mystery_books : ℝ := 17.0
def crime_books : ℝ := 15.0

-- Total number of books bought
def total_books := adventure_books + mystery_books + crime_books

-- The theorem we need to prove
theorem total_books_equals_45 : total_books = 45.0 := by
  -- placeholder for the proof
  sorry

end total_books_equals_45_l293_293189


namespace tan_sin_cos_proof_l293_293782

theorem tan_sin_cos_proof (h1 : Real.sin (Real.pi / 6) = 1 / 2)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2)
    (h3 : Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)) :
    ((Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6))^2) / ((Real.tan (Real.pi / 6))^2 * (Real.cos (Real.pi / 6))^2) = 1 / 3 := by
  sorry

end tan_sin_cos_proof_l293_293782


namespace zoo_problem_l293_293852

theorem zoo_problem (M B L : ℕ) (h1: 26 ≤ M + B + L) (h2: M + B + L ≤ 32) 
    (h3: M + L > B) (h4: B + L = 2 * M) (h5: M + B = 3 * L + 3) (h6: B = L / 2) : 
    B = 3 :=
by
  sorry

end zoo_problem_l293_293852


namespace at_least_two_equal_numbers_written_l293_293879

theorem at_least_two_equal_numbers_written (n : ℕ) (h_n : n > 3)
  (a : Fin n → ℕ)
  (h_distinct : Function.Injective a)
  (h_bound : ∀ i, a i < (n - 1)! + (n - 2)! ) :
  ∃ i j k l : Fin n, i ≠ j ∧ k ≠ l ∧ i ≠ l ∧ j ≠ k ∧ i ≠ k ∧ j ≠ l ∧
  ⌊a i / a j⌋ = ⌊a k / a l⌋ :=
by sorry

end at_least_two_equal_numbers_written_l293_293879


namespace probability_all_and_at_least_one_pass_l293_293821

-- Define conditions
def pA : ℝ := 0.8
def pB : ℝ := 0.6
def pC : ℝ := 0.5

-- Define the main theorem we aim to prove
theorem probability_all_and_at_least_one_pass :
  (pA * pB * pC = 0.24) ∧ ((1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.96) := by
  sorry

end probability_all_and_at_least_one_pass_l293_293821


namespace lucy_times_three_ago_l293_293252

  -- Defining the necessary variables and conditions
  def lucy_age_now : ℕ := 50
  def lovely_age (x : ℕ) : ℕ := 20  -- The age of Lovely when x years has passed
  
  -- Statement of the problem
  theorem lucy_times_three_ago {x : ℕ} : 
    (lucy_age_now - x = 3 * (lovely_age x - x)) → (lucy_age_now + 10 = 2 * (lovely_age x + 10)) → x = 5 := 
  by
  -- Proof is omitted
  sorry
  
end lucy_times_three_ago_l293_293252


namespace min_x_plus_4y_min_value_l293_293647

noncomputable def min_x_plus_4y (x y: ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) : ℝ :=
  x + 4 * y

theorem min_x_plus_4y_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) :
  min_x_plus_4y x y hx hy h = 3 + 2 * Real.sqrt 2 :=
sorry

end min_x_plus_4y_min_value_l293_293647


namespace average_monthly_growth_rate_equation_l293_293333

-- Definitions directly from the conditions
def JanuaryOutput : ℝ := 50
def QuarterTotalOutput : ℝ := 175
def averageMonthlyGrowthRate (x : ℝ) : ℝ :=
  JanuaryOutput + JanuaryOutput * (1 + x) + JanuaryOutput * (1 + x) ^ 2

-- The statement to prove that the derived equation is correct
theorem average_monthly_growth_rate_equation (x : ℝ) :
  averageMonthlyGrowthRate x = QuarterTotalOutput :=
sorry

end average_monthly_growth_rate_equation_l293_293333


namespace min_value_l293_293117

open Real

noncomputable def func (x y z : ℝ) : ℝ := 1 / x + 1 / y + 1 / z

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  func x y z ≥ 4.5 :=
by
  sorry

end min_value_l293_293117


namespace problem_solution_l293_293233

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := 
by
  sorry

end problem_solution_l293_293233


namespace total_amount_l293_293774

theorem total_amount (a b c : ℕ) (h1 : a * 5 = b * 3) (h2 : c * 5 = b * 9) (h3 : b = 50) :
  a + b + c = 170 := by
  sorry

end total_amount_l293_293774


namespace simplify_fraction_l293_293712

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l293_293712


namespace count_four_digit_numbers_ending_25_l293_293473

theorem count_four_digit_numbers_ending_25 : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 100]) → ∃ n : ℕ, n = 100 :=
by
  sorry

end count_four_digit_numbers_ending_25_l293_293473


namespace workshop_worker_allocation_l293_293575

theorem workshop_worker_allocation :
  ∃ (x y : ℕ), 
    x + y = 22 ∧
    6 * x = 5 * y ∧
    x = 10 ∧ y = 12 :=
by
  sorry

end workshop_worker_allocation_l293_293575


namespace slices_leftover_l293_293185

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end slices_leftover_l293_293185


namespace problem1_part1_problem1_part2_problem2_l293_293866

-- Definitions
def quadratic (a b c x : ℝ) := a * x ^ 2 + b * x + c
def has_two_real_roots (a b c : ℝ) := b ^ 2 - 4 * a * c ≥ 0 
def neighboring_root_equation (a b c : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧ |x₁ - x₂| = 1

-- Proof problem 1: Prove whether x^2 + x - 6 = 0 is a neighboring root equation
theorem problem1_part1 : ¬ neighboring_root_equation 1 1 (-6) := 
sorry

-- Proof problem 2: Prove whether 2x^2 - 2√5x + 2 = 0 is a neighboring root equation
theorem problem1_part2 : neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 := 
sorry

-- Proof problem 3: Prove that m = -1 or m = -3 for x^2 - (m-2)x - 2m = 0 to be a neighboring root equation
theorem problem2 (m : ℝ) (h : neighboring_root_equation 1 (-(m-2)) (-2*m)) : 
  m = -1 ∨ m = -3 := 
sorry

end problem1_part1_problem1_part2_problem2_l293_293866


namespace total_pages_correct_l293_293372

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l293_293372


namespace perpendicular_line_eq_l293_293725

theorem perpendicular_line_eq (x y : ℝ) : 
  (∃ m : ℝ, (m * y + 2 * x = -5 / 2) ∧ (x - 2 * y + 3 = 0)) →
  ∃ a b c : ℝ, (a * x + b * y + c = 0) ∧ (2 * a + b = 0) ∧ c = 1 := sorry

end perpendicular_line_eq_l293_293725


namespace rose_needs_more_money_l293_293517

def cost_paintbrush : ℝ := 2.40
def cost_paints : ℝ := 9.20
def cost_easel : ℝ := 6.50
def money_rose_has : ℝ := 7.10

theorem rose_needs_more_money : 
  cost_paintbrush + cost_paints + cost_easel - money_rose_has = 11.00 :=
begin
  sorry
end

end rose_needs_more_money_l293_293517


namespace pizza_slices_leftover_l293_293187

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end pizza_slices_leftover_l293_293187


namespace parallelogram_area_perimeter_impossible_l293_293286

theorem parallelogram_area_perimeter_impossible (a b h : ℕ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h)
    (A : ℕ := b * h) (P : ℕ := 2 * a + 2 * b) :
    (A + P + 6) ≠ 102 := by
  sorry

end parallelogram_area_perimeter_impossible_l293_293286


namespace percentage_of_childrens_books_l293_293176

/-- Conditions: 
- There are 160 books in total.
- 104 of them are for adults.
Prove that the percentage of books intended for children is 35%. --/
theorem percentage_of_childrens_books (total_books : ℕ) (adult_books : ℕ) 
  (h_total : total_books = 160) (h_adult : adult_books = 104) :
  (160 - 104) / 160 * 100 = 35 := 
by {
  sorry -- Proof skipped
}

end percentage_of_childrens_books_l293_293176


namespace ratio_proof_l293_293324

theorem ratio_proof (a b c d : ℝ) (h1 : b = 3 * a) (h2 : c = 4 * b) (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 :=
by sorry

end ratio_proof_l293_293324


namespace range_of_function_l293_293012

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_function : 
  (∀ x : ℝ, x ≠ -2 → f x ≠ 1) ∧
  (∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, f x = y) :=
sorry

end range_of_function_l293_293012


namespace smallest_sum_of_18_consecutive_integers_is_perfect_square_l293_293877

theorem smallest_sum_of_18_consecutive_integers_is_perfect_square 
  (n : ℕ) 
  (S : ℕ) 
  (h1 : S = 9 * (2 * n + 17)) 
  (h2 : ∃ k : ℕ, 2 * n + 17 = k^2) 
  (h3 : ∀ m : ℕ, m < 5 → 2 * n + 17 ≠ m^2) : 
  S = 225 := 
by
  sorry

end smallest_sum_of_18_consecutive_integers_is_perfect_square_l293_293877


namespace values_of_k_real_equal_roots_l293_293790

theorem values_of_k_real_equal_roots (k : ℝ) : 
  (∃ k, (3 - 2 * k)^2 - 4 * 3 * 12 = 0 ∧ (k = -9 / 2 ∨ k = 15 / 2)) :=
by
  sorry

end values_of_k_real_equal_roots_l293_293790


namespace huanhuan_initial_coins_l293_293322

theorem huanhuan_initial_coins :
  ∃ (H L n : ℕ), H = 7 * L ∧ (H + n = 6 * (L + n)) ∧ (H + 2 * n = 5 * (L + 2 * n)) ∧ H = 70 :=
by
  sorry

end huanhuan_initial_coins_l293_293322


namespace number_of_squares_centered_at_60_45_l293_293129

noncomputable def number_of_squares_centered_at (cx : ℕ) (cy : ℕ) : ℕ :=
  let aligned_with_axes := 45
  let not_aligned_with_axes := 2025
  aligned_with_axes + not_aligned_with_axes

theorem number_of_squares_centered_at_60_45 : number_of_squares_centered_at 60 45 = 2070 := 
  sorry

end number_of_squares_centered_at_60_45_l293_293129


namespace number_of_pipes_l293_293196

theorem number_of_pipes (L : ℝ) : 
  let r_small := 1
  let r_large := 3
  let len_small := L
  let len_large := 2 * L
  let volume_large := π * r_large^2 * len_large
  let volume_small := π * r_small^2 * len_small
  volume_large = 18 * volume_small :=
by
  sorry

end number_of_pipes_l293_293196


namespace probability_no_coinciding_sides_l293_293180

theorem probability_no_coinciding_sides :
  let total_triangles := Nat.choose 10 3
  let unfavorable_outcomes := 60 + 10
  let favorable_outcomes := total_triangles - unfavorable_outcomes
  favorable_outcomes / total_triangles = 5 / 12 := by
  sorry

end probability_no_coinciding_sides_l293_293180


namespace four_digit_integer_existence_l293_293722

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end four_digit_integer_existence_l293_293722


namespace yellow_more_than_green_by_l293_293113

-- Define the problem using the given conditions.
def weight_yellow_block : ℝ := 0.6
def weight_green_block  : ℝ := 0.4

-- State the theorem that the yellow block weighs 0.2 pounds more than the green block.
theorem yellow_more_than_green_by : weight_yellow_block - weight_green_block = 0.2 :=
by sorry

end yellow_more_than_green_by_l293_293113


namespace sin_3pi_over_4_minus_alpha_l293_293453

theorem sin_3pi_over_4_minus_alpha (α : ℝ) (h : sin (π / 4 + α) = 3 / 5) :
  sin (3 * π / 4 - α) = 3 / 5 :=
by
  -- since we rely on the result, we don't need to show the proof steps
  sorry

end sin_3pi_over_4_minus_alpha_l293_293453


namespace captain_co_captain_selection_l293_293042

theorem captain_co_captain_selection 
  (men women : ℕ)
  (h_men : men = 12) 
  (h_women : women = 12) : 
  (men * (men - 1) + women * (women - 1)) = 264 := 
by
  -- Since we are skipping the proof here, we use sorry.
  sorry

end captain_co_captain_selection_l293_293042


namespace farmer_plough_rate_l293_293425

-- Define the problem statement and the required proof 

theorem farmer_plough_rate :
  ∀ (x y : ℕ),
  90 * x = 3780 ∧ y * (x + 2) = 3740 → y = 85 :=
by
  sorry

end farmer_plough_rate_l293_293425


namespace smallest_angle_triangle_l293_293337

theorem smallest_angle_triangle 
  (a b c : ℝ)
  (h1 : 3 * a < b + c)
  (h2 : a + b > c)
  (h3 : a + c > b) : 
  ∠BAC < ∠ABC ∧ ∠BAC < ∠ACB :=
by 
  sorry

end smallest_angle_triangle_l293_293337


namespace prob_factor_less_than_nine_l293_293887

theorem prob_factor_less_than_nine : 
  (∃ (n : ℕ), n = 72) ∧ (∃ (total_factors : ℕ), total_factors = 12) ∧ 
  (∃ (factors_lt_9 : ℕ), factors_lt_9 = 6) → 
  (6 / 12 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end prob_factor_less_than_nine_l293_293887


namespace days_A_worked_l293_293557

theorem days_A_worked (W : ℝ) (x : ℝ) (hA : W / 15 * x = W - 6 * (W / 9))
  (hB : W = 6 * (W / 9)) : x = 5 :=
sorry

end days_A_worked_l293_293557


namespace box_prices_l293_293753

theorem box_prices (a b c : ℝ) 
  (h1 : a + b + c = 9) 
  (h2 : 3 * a + 2 * b + c = 16) : 
  c - a = 2 := 
by 
  sorry

end box_prices_l293_293753


namespace min_max_value_z_l293_293355

theorem min_max_value_z (x y z : ℝ) (h1 : x^2 ≤ y + z) (h2 : y^2 ≤ z + x) (h3 : z^2 ≤ x + y) :
  -1/4 ≤ z ∧ z ≤ 2 :=
by {
  sorry
}

end min_max_value_z_l293_293355


namespace num_arrangements_with_ab_together_l293_293395

theorem num_arrangements_with_ab_together (products : Fin 5 → Type) :
  (∃ A B : Fin 5 → Type, A ≠ B) →
  ∃ (n : ℕ), n = 48 :=
by
  sorry

end num_arrangements_with_ab_together_l293_293395


namespace probability_sum_is_prime_l293_293545

theorem probability_sum_is_prime :
  (∃ (d1 d2 d3 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧
  (d1 + d2 + d3 = 3 ∨ d1 + d2 + d3 = 5 ∨ d1 + d2 + d3 = 7 ∨ d1 + d2 + d3 = 11 ∨ d1 + d2 + d3 = 13 ∨ d1 + d2 + d3 = 17)) →
  (∃ p, p = (64/216 : ℚ) ∧ p = (8/27 : ℚ)) :=
begin
  sorry
end

end probability_sum_is_prime_l293_293545


namespace decimal_equivalent_of_one_tenth_squared_l293_293756

theorem decimal_equivalent_of_one_tenth_squared : 
  (1 / 10 : ℝ)^2 = 0.01 := by
  sorry

end decimal_equivalent_of_one_tenth_squared_l293_293756


namespace boxes_needed_to_complete_flooring_l293_293001

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l293_293001


namespace product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l293_293247

theorem product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240
 (p : ℕ) (prime_p : Prime p) (prime_p_plus_2 : Prime (p + 2)) (p_gt_7 : p > 7) :
  240 ∣ ((p - 1) * p * (p + 1)) := by
  sorry

end product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l293_293247


namespace min_value_x_squared_plus_6x_l293_293894

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l293_293894


namespace slope_parallel_to_line_l293_293401

theorem slope_parallel_to_line (x y : ℝ) (h : 3 * x - 6 * y = 15) :
  (∃ m, (∀ b, y = m * x + b) ∧ (∀ k, k ≠ m → ¬ 3 * x - 6 * (k * x + b) = 15)) →
  ∃ p, p = 1/2 :=
sorry

end slope_parallel_to_line_l293_293401


namespace intersects_x_axis_at_one_point_l293_293380

theorem intersects_x_axis_at_one_point (a : ℝ) :
  (∃ x, ax^2 + (a-3)*x + 1 = 0) ∧ (∀ x₁ x₂, ax^2 + (a-3)*x + 1 = 0 → x₁ = x₂) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersects_x_axis_at_one_point_l293_293380


namespace candle_burning_problem_l293_293943

theorem candle_burning_problem (burn_time_per_night_1h : ∀ n : ℕ, n = 8) 
                                (nightly_burn_rate : ∀ h : ℕ, h / 2 = 4) 
                                (total_nights : ℕ) 
                                (two_hour_nightly_burn : ∀ t : ℕ, t = 24) 
                                : ∃ candles : ℕ, candles = 6 := 
by {
  sorry
}

end candle_burning_problem_l293_293943


namespace paco_salty_cookies_left_l293_293694

theorem paco_salty_cookies_left (initial_salty : ℕ) (eaten_salty : ℕ) : initial_salty = 26 ∧ eaten_salty = 9 → initial_salty - eaten_salty = 17 :=
by
  intro h
  cases h
  sorry


end paco_salty_cookies_left_l293_293694


namespace ratio_A_to_B_l293_293906

theorem ratio_A_to_B (A B C : ℕ) (h1 : A + B + C = 406) (h2 : C = 232) (h3 : B = C / 2) : A / gcd A B = 1 ∧ B / gcd A B = 2 := 
by sorry

end ratio_A_to_B_l293_293906


namespace sum_of_cubes_l293_293367

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = 5) : a^3 + b^3 + c^3 = 15 :=
by
  sorry

end sum_of_cubes_l293_293367


namespace sqrt_fraction_difference_l293_293940

theorem sqrt_fraction_difference : 
  (Real.sqrt (16 / 9) - Real.sqrt (9 / 16)) = 7 / 12 :=
by
  sorry

end sqrt_fraction_difference_l293_293940


namespace required_HCl_moles_l293_293090

-- Definitions of chemical substances:
def HCl: Type := Unit
def NaHCO3: Type := Unit
def H2O: Type := Unit
def CO2: Type := Unit
def NaCl: Type := Unit

-- The reaction as a balanced chemical equation:
def balanced_eq (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl) : Prop :=
  ∃ (m: ℕ), m = 1

-- Given conditions:
def condition1: Prop := balanced_eq () () () () ()
def condition2 (moles_H2O moles_CO2 moles_NaCl: ℕ): Prop :=
  moles_H2O = moles_CO2 ∧ moles_CO2 = moles_NaCl ∧ moles_NaCl = moles_H2O

def condition3: ℕ := 3  -- moles of NaHCO3

-- The theorem statement:
theorem required_HCl_moles (moles_HCl moles_NaHCO3: ℕ)
  (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl)
  (balanced: balanced_eq hcl nahco3 h2o co2 nacl)
  (equal_moles: condition2 moles_H2O moles_CO2 moles_NaCl)
  (nahco3_eq_3: moles_NaHCO3 = condition3):
  moles_HCl = 3 :=
sorry

end required_HCl_moles_l293_293090


namespace triangle_square_ratio_l293_293181

theorem triangle_square_ratio (s_t s_s : ℝ) (h : 3 * s_t = 4 * s_s) : s_t / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l293_293181


namespace heights_on_equal_sides_are_equal_l293_293016

-- Given conditions as definitions
def is_isosceles_triangle (a b c : ℝ) := (a = b ∨ b = c ∨ c = a)
def height_on_equal_sides_equal (a b c : ℝ) := is_isosceles_triangle a b c → a = b

-- Lean theorem statement to prove
theorem heights_on_equal_sides_are_equal {a b c : ℝ} : is_isosceles_triangle a b c → height_on_equal_sides_equal a b c := 
sorry

end heights_on_equal_sides_are_equal_l293_293016


namespace min_purchase_amount_is_18_l293_293033

def burger_cost := 2 * 3.20
def fries_cost := 2 * 1.90
def milkshake_cost := 2 * 2.40
def current_total := burger_cost + fries_cost + milkshake_cost
def additional_needed := 3.00
def min_purchase_amount_for_free_delivery := current_total + additional_needed

theorem min_purchase_amount_is_18 : min_purchase_amount_for_free_delivery = 18 := by
  sorry

end min_purchase_amount_is_18_l293_293033


namespace number_of_ways_to_assign_roles_l293_293037

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 5
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let total_men := men - male_roles
  let total_women := women - female_roles
  (men.choose male_roles) * (women.choose female_roles) * (total_men + total_women).choose either_gender_roles = 14400 := by 
sorry

end number_of_ways_to_assign_roles_l293_293037


namespace min_value_expr_l293_293967

/-- Given x > y > 0 and x^2 - y^2 = 1, we need to prove that the minimum value of 2x^2 + 3y^2 - 4xy is 1. -/
theorem min_value_expr {x y : ℝ} (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  2 * x^2 + 3 * y^2 - 4 * x * y = 1 :=
sorry

end min_value_expr_l293_293967


namespace scientific_notation_of_463_4_billion_l293_293351

theorem scientific_notation_of_463_4_billion :
  (463.4 * 10^9) = (4.634 * 10^11) := by
  sorry

end scientific_notation_of_463_4_billion_l293_293351


namespace positiveDifferenceEquation_l293_293141

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l293_293141


namespace solve_for_x_l293_293327

theorem solve_for_x (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l293_293327


namespace Bethany_total_riding_hours_l293_293934

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l293_293934


namespace four_digit_number_l293_293723

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end four_digit_number_l293_293723


namespace problem_solution_l293_293938

noncomputable def negThreePower25 : Real := (-3) ^ 25
noncomputable def twoPowerExpression : Real := 2 ^ (4^2 + 5^2 - 7^2)
noncomputable def threeCubed : Real := 3^3

theorem problem_solution :
  negThreePower25 + twoPowerExpression + threeCubed = -3^25 + 27 + (1 / 256) :=
by
  -- proof omitted
  sorry

end problem_solution_l293_293938


namespace compare_abc_l293_293499

noncomputable def a := Real.exp (Real.sqrt 2)
noncomputable def b := 2 + Real.sqrt 2
noncomputable def c := Real.log (12 + 6 * Real.sqrt 2)

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l293_293499


namespace gcd_306_522_l293_293053

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := 
  by sorry

end gcd_306_522_l293_293053


namespace number_13_on_top_after_folds_l293_293419

/-
A 5x5 grid of numbers from 1 to 25 with the following sequence of folds:
1. Fold along the diagonal from bottom-left to top-right
2. Fold the left half over the right half
3. Fold the top half over the bottom half
4. Fold the bottom half over the top half
Prove that the number 13 ends up on top after all folds.
-/

def grid := (⟨ 5, 5 ⟩ : Nat × Nat)

def initial_grid : ℕ → ℕ := λ n => if 1 ≤ n ∧ n ≤ 25 then n else 0

def fold_diagonal (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 1 fold

def fold_left_over_right (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 2 fold

def fold_top_over_bottom (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 3 fold

def fold_bottom_over_top (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 4 fold

theorem number_13_on_top_after_folds : (fold_bottom_over_top (fold_top_over_bottom (fold_left_over_right (fold_diagonal initial_grid)))) 13 = 13 :=
by {
  sorry
}

end number_13_on_top_after_folds_l293_293419


namespace find_geometric_arithmetic_progressions_l293_293884

theorem find_geometric_arithmetic_progressions
    (b1 b2 b3 : ℚ)
    (h1 : b2^2 = b1 * b3)
    (h2 : b2 + 2 = (b1 + b3) / 2)
    (h3 : (b2 + 2)^2 = b1 * (b3 + 16)) :
    (b1 = 1 ∧ b2 = 3 ∧ b3 = 9) ∨ (b1 = 1/9 ∧ b2 = -5/9 ∧ b3 = 25/9) :=
  sorry

end find_geometric_arithmetic_progressions_l293_293884


namespace AH_HD_ratio_l293_293340

-- Given conditions
variables {A B C H D : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited D]
variables (BC : ℝ) (AC : ℝ) (angle_C : ℝ)
-- We assume the values provided in the problem
variables (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4)

-- Altitudes and orthocenter assumption, representing intersections at orthocenter H
variables (A D H : Type) -- Points to represent A, D, and orthocenter H

noncomputable def AH_H_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) : ℝ :=
  if BC = 6 ∧ AC = 4 * Real.sqrt 2 ∧ angle_C = Real.pi / 4 then 2 else 0

-- We need to prove the ratio AH:HD equals 2 given the conditions
theorem AH_HD_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) :
  AH_H_ratio BC AC angle_C BC_eq AC_eq angle_C_eq = 2 :=
by {
  -- the statement will be proved here
  sorry
}

end AH_HD_ratio_l293_293340


namespace contractor_total_amount_l293_293565

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l293_293565


namespace log_x_squared_y_squared_l293_293981

theorem log_x_squared_y_squared (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x^2 * y^2) = 12 / 5 := 
by
  sorry

end log_x_squared_y_squared_l293_293981


namespace toothpick_pattern_15th_stage_l293_293313

theorem toothpick_pattern_15th_stage :
  let a₁ := 5
  let d := 3
  let n := 15
  a₁ + (n - 1) * d = 47 :=
by
  sorry

end toothpick_pattern_15th_stage_l293_293313


namespace base9_to_decimal_unique_solution_l293_293097

theorem base9_to_decimal_unique_solution :
  ∃ m : ℕ, 1 * 9^4 + 6 * 9^3 + m * 9^2 + 2 * 9^1 + 7 = 11203 ∧ m = 3 :=
by
  sorry

end base9_to_decimal_unique_solution_l293_293097


namespace cos_alpha_beta_value_l293_293971

theorem cos_alpha_beta_value
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β) = Real.sqrt 3 / 3) :
  Real.cos (α + β) = (5 * Real.sqrt 3) / 9 := 
by
  sorry

end cos_alpha_beta_value_l293_293971


namespace number_of_graphic_novels_l293_293931

theorem number_of_graphic_novels (total_books novels_percent comics_percent : ℝ) 
  (h_total : total_books = 120) 
  (h_novels_percent : novels_percent = 0.65) 
  (h_comics_percent : comics_percent = 0.20) :
  total_books - (novels_percent * total_books + comics_percent * total_books) = 18 :=
by
  sorry

end number_of_graphic_novels_l293_293931


namespace volunteer_org_percentage_change_l293_293754

theorem volunteer_org_percentage_change :
  ∀ (X : ℝ), X > 0 → 
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  (X - spring_decrease) / X * 100 = 11.71 :=
by
  intro X hX
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  show (_ - _) / _ * _ = _
  sorry

end volunteer_org_percentage_change_l293_293754


namespace problem_statement_l293_293094

variable {F : Type*} [Field F]

theorem problem_statement (m : F) (h : m + 1 / m = 6) : m^2 + 1 / m^2 + 4 = 38 :=
by
  sorry

end problem_statement_l293_293094


namespace bob_pays_more_than_samantha_l293_293519

theorem bob_pays_more_than_samantha
  (total_slices : ℕ := 12)
  (cost_plain_pizza : ℝ := 12)
  (cost_olives : ℝ := 3)
  (slices_one_third_pizza : ℕ := total_slices / 3)
  (total_cost : ℝ := cost_plain_pizza + cost_olives)
  (cost_per_slice : ℝ := total_cost / total_slices)
  (bob_slices_total : ℕ := slices_one_third_pizza + 3)
  (samantha_slices_total : ℕ := total_slices - bob_slices_total)
  (bob_total_cost : ℝ := bob_slices_total * cost_per_slice)
  (samantha_total_cost : ℝ := samantha_slices_total * cost_per_slice) :
  bob_total_cost - samantha_total_cost = 2.5 :=
by
  sorry

end bob_pays_more_than_samantha_l293_293519


namespace polynomial_even_iff_exists_Q_l293_293857

open Polynomial

noncomputable def exists_polynomial_Q (P : Polynomial ℂ) : Prop :=
  ∃ Q : Polynomial ℂ, ∀ z : ℂ, P.eval z = (Q.eval z) * (Q.eval (-z))

theorem polynomial_even_iff_exists_Q (P : Polynomial ℂ) :
  (∀ z : ℂ, P.eval z = P.eval (-z)) ↔ exists_polynomial_Q P :=
by 
  sorry

end polynomial_even_iff_exists_Q_l293_293857


namespace simplify_expr_l293_293709

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l293_293709


namespace convert_512_to_base_7_l293_293437

/-- Convert 512 in base 10 to base 7 and verify the result is 1331 in base 7 -/
theorem convert_512_to_base_7 : Nat.toDigits 7 512 = [1, 3, 3, 1] := sorry

end convert_512_to_base_7_l293_293437


namespace remainder_of_expression_l293_293368

theorem remainder_of_expression (x y u v : ℕ) (h : x = u * y + v) (Hv : 0 ≤ v ∧ v < y) :
  (if v + 2 < y then (x + 3 * u * y + 2) % y = v + 2
   else (x + 3 * u * y + 2) % y = v + 2 - y) :=
by sorry

end remainder_of_expression_l293_293368


namespace quotient_is_36_l293_293830

-- Conditions
def divisor := 85
def remainder := 26
def dividend := 3086

-- The Question and Answer (proof required)
theorem quotient_is_36 (quotient : ℕ) (h : dividend = (divisor * quotient) + remainder) : quotient = 36 := by 
  sorry

end quotient_is_36_l293_293830


namespace senior_citizen_ticket_cost_l293_293571

theorem senior_citizen_ticket_cost 
  (total_tickets : ℕ)
  (regular_ticket_cost : ℕ)
  (total_sales : ℕ)
  (sold_regular_tickets : ℕ)
  (x : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : sold_regular_tickets = 41)
  (h5 : total_sales = (sold_regular_tickets * regular_ticket_cost) + ((total_tickets - sold_regular_tickets) * x)) :
  x = 10 :=
by
  sorry

end senior_citizen_ticket_cost_l293_293571


namespace val_need_33_stamps_l293_293745

def valerie_needs_total_stamps 
    (thank_you_cards : ℕ) 
    (bills_water : ℕ) 
    (bills_electric : ℕ) 
    (bills_internet : ℕ) 
    (rebate_addition : ℕ) 
    (rebate_stamps : ℕ) 
    (job_apps_multiplier : ℕ) 
    (job_app_stamps : ℕ) 
    (total_stamps : ℕ) : Prop :=
    thank_you_cards = 3 ∧
    bills_water = 1 ∧
    bills_electric = 2 ∧
    bills_internet = 3 ∧
    rebate_addition = 3 ∧
    rebate_stamps = 2 ∧
    job_apps_multiplier = 2 ∧
    job_app_stamps = 1 ∧
    total_stamps = 33

theorem val_need_33_stamps : 
  valerie_needs_total_stamps 3 1 2 3 3 2 2 1 33 :=
by 
  -- proof skipped
  sorry

end val_need_33_stamps_l293_293745


namespace find_x_value_l293_293058

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l293_293058


namespace smallest_root_of_quadratic_l293_293962

theorem smallest_root_of_quadratic (y : ℝ) (h : 4 * y^2 - 7 * y + 3 = 0) : y = 3 / 4 :=
sorry

end smallest_root_of_quadratic_l293_293962


namespace amount_spent_per_sibling_l293_293349

-- Definitions and conditions
def total_spent := 150
def amount_per_parent := 30
def num_parents := 2
def num_siblings := 3

-- Claim
theorem amount_spent_per_sibling :
  (total_spent - (amount_per_parent * num_parents)) / num_siblings = 30 :=
by
  sorry

end amount_spent_per_sibling_l293_293349


namespace curve_intersections_l293_293817

theorem curve_intersections (m : ℝ) :
  (∃ x y : ℝ, ((x-1)^2 + y^2 = 1) ∧ (y = mx + m) ∧ (y ≠ 0) ∧ (y^2 = 0)) =
  ((m > -Real.sqrt 3 / 3) ∧ (m < 0)) ∨ ((m > 0) ∧ (m < Real.sqrt 3 / 3)) := 
sorry

end curve_intersections_l293_293817


namespace max_roses_purchasable_l293_293362

theorem max_roses_purchasable 
  (price_individual : ℝ) (price_dozen : ℝ) (price_two_dozen : ℝ) (price_five_dozen : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) (total_money : ℝ) : 
  (price_individual = 4.50) →
  (price_dozen = 36) →
  (price_two_dozen = 50) →
  (price_five_dozen = 110) →
  (discount_threshold = 36) →
  (discount_rate = 0.10) →
  (total_money = 680) →
  ∃ (roses : ℕ), roses = 364 :=
by
  -- Definitions based on conditions
  intros
  -- The proof steps have been omitted for brevity
  sorry

end max_roses_purchasable_l293_293362


namespace expected_value_is_one_third_l293_293910

noncomputable def expected_value_of_winnings : ℚ :=
  let p1 := (1/6 : ℚ)
  let p2 := (1/6 : ℚ)
  let p3 := (1/6 : ℚ)
  let p4 := (1/6 : ℚ)
  let p5 := (1/6 : ℚ)
  let p6 := (1/6 : ℚ)
  let winnings1 := (5 : ℚ)
  let winnings2 := (5 : ℚ)
  let winnings3 := (0 : ℚ)
  let winnings4 := (0 : ℚ)
  let winnings5 := (-4 : ℚ)
  let winnings6 := (-4 : ℚ)
  (p1 * winnings1 + p2 * winnings2 + p3 * winnings3 + p4 * winnings4 + p5 * winnings5 + p6 * winnings6)

theorem expected_value_is_one_third : expected_value_of_winnings = 1 / 3 := by
  sorry

end expected_value_is_one_third_l293_293910


namespace sequence_sum_l293_293833

variable (P Q R S T U V : ℤ)
variable (hR : R = 7)
variable (h1 : P + Q + R = 36)
variable (h2 : Q + R + S = 36)
variable (h3 : R + S + T = 36)
variable (h4 : S + T + U = 36)
variable (h5 : T + U + V = 36)

theorem sequence_sum (P Q R S T U V : ℤ)
  (hR : R = 7)
  (h1 : P + Q + R = 36)
  (h2 : Q + R + S = 36)
  (h3 : R + S + T = 36)
  (h4 : S + T + U = 36)
  (h5 : T + U + V = 36) :
  P + V = 29 := 
sorry

end sequence_sum_l293_293833


namespace percent_neither_filler_nor_cheese_l293_293760

-- Define the given conditions as constants
def total_weight : ℕ := 200
def filler_weight : ℕ := 40
def cheese_weight : ℕ := 30

-- Definition of the remaining weight that is neither filler nor cheese
def neither_weight : ℕ := total_weight - filler_weight - cheese_weight

-- Calculation of the percentage of the burger that is neither filler nor cheese
def percentage_neither : ℚ := (neither_weight : ℚ) / (total_weight : ℚ) * 100

-- The theorem to prove
theorem percent_neither_filler_nor_cheese :
  percentage_neither = 65 := by
  sorry

end percent_neither_filler_nor_cheese_l293_293760


namespace compound_interest_rate_l293_293048

open Real

theorem compound_interest_rate
  (P : ℝ) (A : ℝ) (t : ℝ) (r : ℝ)
  (h_inv : P = 8000)
  (h_time : t = 2)
  (h_maturity : A = 8820) :
  r = 0.05 :=
by
  sorry

end compound_interest_rate_l293_293048


namespace total_hours_over_two_weeks_l293_293932

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l293_293932


namespace correct_equation_option_l293_293159

theorem correct_equation_option :
  (∀ (x : ℝ), (x = 4 → false) ∧ (x = -4 → false)) →
  (∀ (y : ℝ), (y = 12 → true) ∧ (y = -12 → false)) →
  (∀ (z : ℝ), (z = -7 → false) ∧ (z = 7 → true)) →
  (∀ (w : ℝ), (w = 2 → true)) →
  ∃ (option : ℕ), option = 4 := 
by
  sorry

end correct_equation_option_l293_293159


namespace sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l293_293584

-- Definition and proof of sqrt(9) = 3
theorem sqrt_of_9 : Real.sqrt 9 = 3 := by
  sorry

-- Definition and proof of -sqrt(0.49) = -0.7
theorem neg_sqrt_of_0_49 : -Real.sqrt 0.49 = -0.7 := by
  sorry

-- Definition and proof of ±sqrt(64/81) = ±(8/9)
theorem pm_sqrt_of_64_div_81 : (Real.sqrt (64 / 81) = 8 / 9) ∧ (Real.sqrt (64 / 81) = -8 / 9) := by
  sorry

end sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l293_293584


namespace distinct_sequences_triangle_l293_293474

open Finset

theorem distinct_sequences_triangle : 
  let available_letters := {'R', 'I', 'A', 'N', 'G', 'L'} : Finset Char,
      choose_4_letters := available_letters.choose 4,
      permute_4_letters := \u (x : Finset (Finset Char)) (x ∈ choose_4_letters), x.card.perm 4
  in ∑ val in choose_4_letters, val.card.perm (4 : ℕ) = 360 := sorry

end distinct_sequences_triangle_l293_293474


namespace floor_factorial_even_l293_293964

theorem floor_factorial_even (n : ℕ) (hn : n > 0) : 
  Nat.floor ((Nat.factorial (n - 1) : ℝ) / (n * (n + 1))) % 2 = 0 := 
sorry

end floor_factorial_even_l293_293964


namespace negation_of_p_is_false_l293_293663

def prop_p : Prop :=
  ∀ x : ℝ, 1 < x → (Real.log (x + 2) / Real.log 3 - 2 / 2^x) > 0

theorem negation_of_p_is_false : ¬(∃ x : ℝ, 1 < x ∧ (Real.log (x + 2) / Real.log 3 - 2 / 2^x) ≤ 0) :=
sorry

end negation_of_p_is_false_l293_293663


namespace arithmetic_sequence_n_equals_100_l293_293101

theorem arithmetic_sequence_n_equals_100
  (a₁ : ℕ) (d : ℕ) (a_n : ℕ)
  (h₁ : a₁ = 1)
  (h₂ : d = 3)
  (h₃ : a_n = 298) :
  ∃ n : ℕ, a_n = a₁ + (n - 1) * d ∧ n = 100 :=
by
  sorry

end arithmetic_sequence_n_equals_100_l293_293101


namespace price_diff_is_correct_l293_293386

-- Define initial conditions
def initial_price : ℝ := 30
def flat_discount : ℝ := 5
def percent_discount : ℝ := 0.25
def sales_tax : ℝ := 0.10

def price_after_flat_discount (price : ℝ) : ℝ :=
  price - flat_discount

def price_after_percent_discount (price : ℝ) : ℝ :=
  price * (1 - percent_discount)

def price_after_tax (price : ℝ) : ℝ :=
  price * (1 + sales_tax)

def final_price_method1 : ℝ :=
  price_after_tax (price_after_percent_discount (price_after_flat_discount initial_price))

def final_price_method2 : ℝ :=
  price_after_tax (price_after_flat_discount (price_after_percent_discount initial_price))

def difference_in_cents : ℝ :=
  (final_price_method1 - final_price_method2) * 100

-- Lean statement to prove the final difference in cents
theorem price_diff_is_correct : difference_in_cents = 137.5 :=
  by sorry

end price_diff_is_correct_l293_293386


namespace train_length_is_100_meters_l293_293043

-- Definitions of conditions
def speed_kmh := 40  -- speed in km/hr
def time_s := 9  -- time in seconds

-- Conversion factors
def km_to_m := 1000  -- 1 km = 1000 meters
def hr_to_s := 3600  -- 1 hour = 3600 seconds

-- Converting speed from km/hr to m/s
def speed_ms := (speed_kmh * km_to_m) / hr_to_s

-- The proof that the length of the train is 100 meters
theorem train_length_is_100_meters :
  (speed_ms * time_s) = 100 :=
by
  sorry

-- The Lean statement merely sets up the problem as asked.

end train_length_is_100_meters_l293_293043


namespace convex_quadrilaterals_count_l293_293201

-- We define the problem in Lean
theorem convex_quadrilaterals_count (n : ℕ) (h : n > 4) 
  (h_no_collinear : ∀ p1 p2 p3 : nat, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear p1 p2 p3) :
  ∃ quadrilateral_count : ℕ, quadrilateral_count ≥ (nat.choose (n-3) 2) :=
sorry

end convex_quadrilaterals_count_l293_293201


namespace problem1_problem2_l293_293585

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end problem1_problem2_l293_293585


namespace find_a6_geometric_sequence_l293_293490

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem find_a6_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h1 : geom_seq a q) (h2 : a 4 = 7) (h3 : a 8 = 63) : 
  a 6 = 21 :=
sorry

end find_a6_geometric_sequence_l293_293490


namespace arcsin_one_half_l293_293783

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end arcsin_one_half_l293_293783


namespace locus_of_vertices_l293_293155

theorem locus_of_vertices (t : ℝ) (x y : ℝ) (h : y = x^2 + t * x + 1) : y = 1 - x^2 :=
by
  sorry

end locus_of_vertices_l293_293155


namespace count_valid_permutations_l293_293257

theorem count_valid_permutations : finset.filter (λ (s : finset ℕ), s ∈ (finset.perm (finset.range 1 7)) ∧
  (∀ (i j : ℕ), i < j → (s.nth i = 1 → s.nth j = 2)) ∧ 
  (∀ (i j : ℕ), i < j → (s.nth i = 3 → s.nth j = 4))) finset.univ.card = 180 :=
by
  sorry

end count_valid_permutations_l293_293257


namespace expr_eval_l293_293190

noncomputable def expr_value : ℕ :=
  (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6)

theorem expr_eval : expr_value = 18 := by
  sorry

end expr_eval_l293_293190


namespace average_t_value_is_15_l293_293818

noncomputable def average_of_distinct_t_values (t_vals : List ℤ) : ℤ :=
t_vals.sum / t_vals.length

theorem average_t_value_is_15 :
  average_of_distinct_t_values [8, 14, 18, 20] = 15 :=
by
  sorry

end average_t_value_is_15_l293_293818


namespace smallest_area_inscribed_ngon_midpoints_l293_293249

noncomputable theory

open Real

variables {n : ℕ} (h_n : n > 3) (A : Fin n → ℝ × ℝ) (B : Fin n → ℝ × ℝ)

def is_regular_polygon (vertices : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j, dist (vertices i) (vertices ((i + 1) % n)) = dist (vertices j) (vertices ((j + 1) % n)) ∧
        ∠ (vertices i) (vertices ((i + 1) % n)) (vertices ((i + 2) % n)) = 2 * π / n

def is_midpoint (A : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)) (B : (ℝ × ℝ)) : Prop :=
  ∀ i, B i = ((A i) + (A ((i + 1) % n))) / 2

def area (vertices : Fin n → ℝ × ℝ) : ℝ := sorry  -- Implementation of area for a polygon

theorem smallest_area_inscribed_ngon_midpoints :
  is_regular_polygon h_n A →
  is_regular_polygon h_n B →
  (∀ i, ∃ c, B i = (1 - c) • A i + c • A ((i + 1) % n)) →
  (∀ i, is_midpoint A (B i)) :=
begin
  intros hA hB h_inscribed,
  sorry -- Proof would go here
end

end smallest_area_inscribed_ngon_midpoints_l293_293249


namespace rhombus_side_length_l293_293734

variables (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r)

theorem rhombus_side_length (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r) :
  ∃ s : ℝ, s = 2 * r / Real.sin α :=
sorry

end rhombus_side_length_l293_293734


namespace loss_per_metre_proof_l293_293770

-- Define the given conditions
def cost_price_per_metre : ℕ := 66
def quantity_sold : ℕ := 200
def total_selling_price : ℕ := 12000

-- Define total cost price based on cost price per metre and quantity sold
def total_cost_price : ℕ := cost_price_per_metre * quantity_sold

-- Define total loss based on total cost price and total selling price
def total_loss : ℕ := total_cost_price - total_selling_price

-- Define loss per metre
def loss_per_metre : ℕ := total_loss / quantity_sold

-- The theorem we need to prove:
theorem loss_per_metre_proof : loss_per_metre = 6 :=
  by
    sorry

end loss_per_metre_proof_l293_293770


namespace cos_180_degree_l293_293607

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l293_293607


namespace sum_of_digits_l293_293118

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 9
noncomputable def C : ℕ := 2
noncomputable def BC : ℕ := B * 10 + C
noncomputable def ABC : ℕ := A * 100 + B * 10 + C

theorem sum_of_digits (H1: A ≠ 0) (H2: B ≠ 0) (H3: C ≠ 0) (H4: BC + ABC + ABC = 876):
  A + B + C = 14 :=
sorry

end sum_of_digits_l293_293118


namespace range_rational_function_l293_293013

noncomputable def rational_function (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_rational_function :
  (Set.range rational_function) = Set.Ioo (⊥ : ℝ) 1 ∪ Set.Ioo 1 ⊤ :=
by
  sorry

end range_rational_function_l293_293013


namespace train_passes_platform_in_39_2_seconds_l293_293922

def length_of_train : ℝ := 360
def speed_in_kmh : ℝ := 45
def length_of_platform : ℝ := 130

noncomputable def speed_in_mps : ℝ := speed_in_kmh * 1000 / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_in_mps

theorem train_passes_platform_in_39_2_seconds :
  time_to_pass_platform = 39.2 := by
  sorry

end train_passes_platform_in_39_2_seconds_l293_293922


namespace arithmetic_sequence_common_diff_l293_293651

noncomputable def variance (s : List ℝ) : ℝ :=
  let mean := (s.sum) / (s.length : ℝ)
  (s.map (λ x => (x - mean) ^ 2)).sum / (s.length : ℝ)

theorem arithmetic_sequence_common_diff (a1 a2 a3 a4 a5 a6 a7 d : ℝ) 
(h_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ a5 = a1 + 4 * d ∧ a6 = a1 + 5 * d ∧ a7 = a1 + 6 * d)
(h_var : variance [a1, a2, a3, a4, a5, a6, a7] = 1) : 
d = 1 / 2 ∨ d = -1 / 2 := 
sorry

end arithmetic_sequence_common_diff_l293_293651


namespace quotient_zero_l293_293870

theorem quotient_zero (D d R Q : ℕ) (hD : D = 12) (hd : d = 17) (hR : R = 8) (h : D = d * Q + R) : Q = 0 :=
by
  sorry

end quotient_zero_l293_293870


namespace price_returns_to_initial_l293_293537

theorem price_returns_to_initial (x : ℝ) (h : 0.918 * (100 + x) = 100) : x = 9 := 
by
  sorry

end price_returns_to_initial_l293_293537


namespace a_share_is_6300_l293_293409

noncomputable def investment_split (x : ℝ) :  ℝ × ℝ × ℝ :=
  let a_share := x * 12
  let b_share := 2 * x * 6
  let c_share := 3 * x * 4
  (a_share, b_share, c_share)

noncomputable def total_gain : ℝ := 18900

noncomputable def a_share_calculation : ℝ :=
  let (a_share, b_share, c_share) := investment_split 1
  total_gain / (a_share + b_share + c_share) * a_share

theorem a_share_is_6300 : a_share_calculation = 6300 := by
  -- Here, you would provide the proof, but for now we skip it.
  sorry

end a_share_is_6300_l293_293409


namespace dima_and_serezha_meet_time_l293_293792

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end dima_and_serezha_meet_time_l293_293792


namespace dot_product_a_b_l293_293469

-- Define the given vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

-- Define the dot product function
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- State the theorem with the correct answer
theorem dot_product_a_b : dot_product a b = 1 :=
by
  sorry

end dot_product_a_b_l293_293469


namespace external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l293_293006

variables {O₁ O₂ : ℝ} {r R : ℝ}

-- External tangency implies sum of radii equals distance between centers
theorem external_tangency_sum {O₁ O₂ r R : ℝ} (h1 : O₁ ≠ O₂) (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = r + R) : 
  dist O₁ O₂ = r + R :=
sorry

-- Internal tangency implies difference of radii equals distance between centers
theorem internal_tangency_diff {O₁ O₂ r R : ℝ} 
  (h1 : O₁ ≠ O₂) 
  (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = abs (R - r)) : 
  dist O₁ O₂ = abs (R - r) :=
sorry

-- Converse for sum of radii equals distance between centers
theorem converse_sum_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = r + R) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = r + R) :=
sorry

-- Converse for difference of radii equals distance between centers
theorem converse_diff_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = abs (R - r)) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = abs (R - r)) :=
sorry

end external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l293_293006


namespace simplify_fraction_l293_293705

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l293_293705


namespace product_polynomials_l293_293779

theorem product_polynomials (x : ℝ) : 
  (1 + x^3) * (1 - 2 * x + x^4) = 1 - 2 * x + x^3 - x^4 + x^7 :=
by sorry

end product_polynomials_l293_293779


namespace perfect_square_as_sum_of_powers_of_2_l293_293056

theorem perfect_square_as_sum_of_powers_of_2 (n a b : ℕ) (h : n^2 = 2^a + 2^b) (hab : a ≥ b) :
  (∃ k : ℕ, n^2 = 4^(k + 1)) ∨ (∃ k : ℕ, n^2 = 9 * 4^k) :=
by
  sorry

end perfect_square_as_sum_of_powers_of_2_l293_293056


namespace common_solution_l293_293640

theorem common_solution (x : ℚ) : 
  (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) → x = 1 / 8 :=
by { sorry }

end common_solution_l293_293640


namespace mila_social_media_hours_l293_293307

/-- 
Mila spends 6 hours on his phone every day. 
Half of this time is spent on social media. 
Prove that Mila spends 21 hours on social media in a week.
-/
theorem mila_social_media_hours 
  (hours_per_day : ℕ)
  (phone_time_per_day : hours_per_day = 6)
  (daily_social_media_fraction : ℕ)
  (fractional_time : daily_social_media_fraction = hours_per_day / 2)
  (days_per_week : ℕ)
  (days_in_week : days_per_week = 7) :
  (daily_social_media_fraction * days_per_week = 21) :=
sorry

end mila_social_media_hours_l293_293307


namespace selling_price_before_brokerage_l293_293022

variables (CR BR SP : ℝ)
variables (hCR : CR = 120.50) (hBR : BR = 1 / 400)

theorem selling_price_before_brokerage :
  SP = (CR * 400) / (399) := 
by
  sorry

end selling_price_before_brokerage_l293_293022


namespace mustard_bottles_total_l293_293435

theorem mustard_bottles_total (b1 b2 b3 : ℝ) (h1 : b1 = 0.25) (h2 : b2 = 0.25) (h3 : b3 = 0.38) :
  b1 + b2 + b3 = 0.88 :=
by
  sorry

end mustard_bottles_total_l293_293435


namespace base8_units_digit_l293_293005

theorem base8_units_digit (n m : ℕ) (h1 : n = 348) (h2 : m = 27) : 
  (n * m % 8) = 4 := sorry

end base8_units_digit_l293_293005


namespace non_degenerate_triangles_l293_293290

theorem non_degenerate_triangles :
  let total_points := 16
  let collinear_points := 5
  let total_triangles := Nat.choose total_points 3
  let degenerate_triangles := 2 * Nat.choose collinear_points 3
  let nondegenerate_triangles := total_triangles - degenerate_triangles
  nondegenerate_triangles = 540 := 
by
  sorry

end non_degenerate_triangles_l293_293290


namespace price_per_gallon_in_NC_l293_293484

variable (P : ℝ)
variable (price_nc := P) -- price per gallon in North Carolina
variable (price_va := P + 1) -- price per gallon in Virginia
variable (gallons_nc := 10) -- gallons bought in North Carolina
variable (gallons_va := 10) -- gallons bought in Virginia
variable (total_cost := 50) -- total amount spent on gas

theorem price_per_gallon_in_NC :
  (gallons_nc * price_nc) + (gallons_va * price_va) = total_cost → price_nc = 2 :=
by
  sorry

end price_per_gallon_in_NC_l293_293484


namespace smallest_angle_in_triangle_l293_293336

open Real

theorem smallest_angle_in_triangle
  (a b c : ℝ)
  (h : a = (b + c) / 3)
  (triangle_inequality_1 : a + b > c)
  (triangle_inequality_2 : a + c > b)
  (triangle_inequality_3 : b + c > a) :
  ∃ A B C α β γ : ℝ, -- A, B, C are the angles opposite to sides a, b, c respectively
  0 < α ∧ α < β ∧ α < γ :=
sorry

end smallest_angle_in_triangle_l293_293336


namespace simplify_fraction_l293_293711

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l293_293711


namespace sum_integer_solutions_l293_293449

open Polynomial

theorem sum_integer_solutions : 
    (finset.univ.filter (λ x : ℤ, (x^2 - 21*x + 100 = 0)).sum (λ x, x)) = 0 :=
sorry

end sum_integer_solutions_l293_293449


namespace total_fires_l293_293927

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end total_fires_l293_293927


namespace probability_ace_king_queen_same_suit_l293_293397

theorem probability_ace_king_queen_same_suit :
  let total_probability := (1 : ℝ) / 52 * (1 : ℝ) / 51 * (1 : ℝ) / 50
  total_probability = (1 : ℝ) / 132600 :=
by
  sorry

end probability_ace_king_queen_same_suit_l293_293397


namespace tan_subtraction_example_l293_293645

noncomputable def tan_subtraction_identity (alpha beta : ℝ) : ℝ :=
  (Real.tan alpha - Real.tan beta) / (1 + Real.tan alpha * Real.tan beta)

theorem tan_subtraction_example (theta : ℝ) (h : Real.tan theta = 1 / 2) :
  Real.tan (π / 4 - theta) = 1 / 3 := 
by
  sorry

end tan_subtraction_example_l293_293645


namespace f_19_eq_2017_l293_293648

noncomputable def f : ℤ → ℤ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ m n : ℤ, f (m + n) = f m + f n + 3 * (4 * m * n - 1)

theorem f_19_eq_2017 : f 19 = 2017 := by
  sorry

end f_19_eq_2017_l293_293648


namespace hyperbola_foci_distance_l293_293134

open Real

-- Definitions from the condition:
def asymptote1 (x : ℝ) : ℝ := 2 * x - 2
def asymptote2 (x : ℝ) : ℝ := -2 * x + 6
def hyperbola_pass_point : Prod ℝ ℝ := (4, 4)

-- Problem statement: Prove the correct distance between the foci of the hyperbola.
theorem hyperbola_foci_distance : 
  ∀ (a b : ℝ), asymptote1(2) = 2 ∧ asymptote2(2) = 2 ∧ (((4-2)^2 / a^2) - ((4-2)^2 / b^2) = 1) ∧ 
  (a^2 = 8 ∧ b^2 = 4) →
  2 * sqrt(a^2 + b^2) = 4 * sqrt(3) :=
by
  sorry

end hyperbola_foci_distance_l293_293134


namespace f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l293_293813

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2^x + a

theorem f_monotone_on_0_to_2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2 :=
sorry

theorem find_range_a_part2 : (∀ x1 : ℝ, x1 ∈ (Set.Icc (1/2) 1) → 
  ∃ x2 : ℝ, x2 ∈ (Set.Icc 2 3) ∧ f x1 ≥ g x2 a) → a ≤ 1 :=
sorry

theorem find_range_a_part3 : (∃ x : ℝ, x ∈ (Set.Icc 0 2) ∧ f x ≤ g x a) → a ≥ 0 :=
sorry

end f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l293_293813


namespace arcsin_one_half_l293_293785

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l293_293785


namespace fraction_of_water_lost_l293_293399

theorem fraction_of_water_lost (pipe1_rate pipe2_rate total_fill_time effective_fill_time : ℚ)
  (h_pipe1 : pipe1_rate = 1 / 20)
  (h_pipe2 : pipe2_rate = 1 / 30)
  (h_total_fill : total_fill_time = 1 / 16) :
  ∃ L : ℚ, (1 - L) * (pipe1_rate + pipe2_rate) = total_fill_time ∧ L = 1 / 4 :=
by
  sorry

end fraction_of_water_lost_l293_293399


namespace second_option_cost_per_day_l293_293494

theorem second_option_cost_per_day :
  let distance_one_way := 150
  let rental_first_option := 50
  let kilometers_per_liter := 15
  let cost_per_liter := 0.9
  let savings := 22
  let total_distance := distance_one_way * 2
  let total_liters := total_distance / kilometers_per_liter
  let gasoline_cost := total_liters * cost_per_liter
  let total_cost_first_option := rental_first_option + gasoline_cost
  let second_option_cost := total_cost_first_option + savings
  second_option_cost = 90 :=
by
  sorry

end second_option_cost_per_day_l293_293494


namespace contractor_net_earnings_l293_293564

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l293_293564


namespace ball_highest_point_at_l293_293872

noncomputable def h (a b t : ℝ) : ℝ := a * t^2 + b * t

theorem ball_highest_point_at (a b : ℝ) :
  (h a b 3 = h a b 7) →
  t = 4.9 :=
by
  sorry

end ball_highest_point_at_l293_293872


namespace time_to_meet_l293_293794

-- Definitions based on conditions
def motorboat_speed_Serezha : ℝ := 20 -- km/h
def crossing_time_Serezha : ℝ := 0.5 -- hours (30 minutes)
def running_speed_Dima : ℝ := 6 -- km/h
def running_time_Dima : ℝ := 0.25 -- hours (15 minutes)
def combined_speed : ℝ := running_speed_Dima + running_speed_Dima -- equal speeds running towards each other
def distance_meet : ℝ := (running_speed_Dima * running_time_Dima) -- The distance they need to cover towards each other

-- Prove the time for them to meet
theorem time_to_meet : (distance_meet / combined_speed) = (7.5 / 60) :=
by
  sorry

end time_to_meet_l293_293794


namespace runway_show_time_l293_293256

theorem runway_show_time
  (models : ℕ)
  (bathing_suits_per_model : ℕ)
  (evening_wear_per_model : ℕ)
  (time_per_trip : ℕ)
  (total_models : models = 6)
  (bathing_suits_sets : bathing_suits_per_model = 2)
  (evening_wear_sets : evening_wear_per_model = 3)
  (trip_duration : time_per_trip = 2) :
  (models * bathing_suits_per_model + models * evening_wear_per_model) * time_per_trip = 60 := 
by
  rw [total_models, bathing_suits_sets, evening_wear_sets, trip_duration]
  -- Simplify expression (6 * 2 + 6 * 3) * 2
  simp
  -- Equals to 60
  exact rfl

end runway_show_time_l293_293256


namespace aquarium_counts_l293_293587

-- Defining the entities Otters, Seals, and Sea Lions
variables (O S L : ℕ)

-- Defining the conditions from the problem
def condition_1 : Prop := (O + S = 7)
def condition_2 : Prop := (L + S = 6)
def condition_3 : Prop := (O + L = 5)
def condition_4 : Prop := (min O S = 5)

-- Theorem: Proving the exact counts of Otters, Seals, and Sea Lions
theorem aquarium_counts (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) :
  O = 5 ∧ S = 7 ∧ L = 6 :=
sorry

end aquarium_counts_l293_293587


namespace interest_calculation_years_l293_293376

noncomputable def principal : ℝ := 625
noncomputable def rate : ℝ := 0.04
noncomputable def difference : ℝ := 1

theorem interest_calculation_years (n : ℕ) : 
    (principal * (1 + rate)^n - principal - (principal * rate * n) = difference) → 
    n = 2 :=
by sorry

end interest_calculation_years_l293_293376


namespace limit_of_exp_over_sin_l293_293192

theorem limit_of_exp_over_sin (α β : ℝ) : 
  tendsto (λ x : ℝ, (exp (α * x) - exp (β * x)) / (sin (α * x) - sin (β * x))) (𝓝 0) (𝓝 1) :=
begin
  sorry
end

end limit_of_exp_over_sin_l293_293192


namespace M_greater_than_N_l293_293808

variable (a : ℝ)

def M := 2 * a^2 - 4 * a
def N := a^2 - 2 * a - 3

theorem M_greater_than_N : M a > N a := by
  sorry

end M_greater_than_N_l293_293808


namespace intersection_dist_general_l293_293457

theorem intersection_dist_general {a b : ℝ} 
  (h1 : (a^2 + 1) * (a^2 + 4 * (b + 1)) = 34)
  (h2 : (a^2 + 1) * (a^2 + 4 * (b + 2)) = 42) : 
  ∀ x1 x2 : ℝ, 
  x1 ≠ x2 → 
  (x1 * x1 = a * x1 + b - 1 ∧ x2 * x2 = a * x2 + b - 1) → 
  |x2 - x1| = 3 * Real.sqrt 2 :=
by
  sorry

end intersection_dist_general_l293_293457


namespace manager_salary_is_3600_l293_293135

-- Definitions based on the conditions
def average_salary_20_employees := 1500
def number_of_employees := 20
def new_average_salary := 1600
def number_of_people_incl_manager := number_of_employees + 1

-- Calculate necessary total salaries and manager's salary
def total_salary_of_20_employees := number_of_employees * average_salary_20_employees
def new_total_salary_with_manager := number_of_people_incl_manager * new_average_salary
def manager_monthly_salary := new_total_salary_with_manager - total_salary_of_20_employees

-- The statement to be proved
theorem manager_salary_is_3600 : manager_monthly_salary = 3600 :=
by
  sorry

end manager_salary_is_3600_l293_293135


namespace stage_order_permutations_l293_293221

-- Define the problem in Lean terms
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem stage_order_permutations :
  let total_students := 6
  let predetermined_students := 3
  (permutations total_students) / (permutations predetermined_students) = 120 := by
  sorry

end stage_order_permutations_l293_293221


namespace how_many_grapes_l293_293254

-- Define the conditions given in the problem
def apples_to_grapes :=
  (3 / 4) * 12 = 6

-- Define the result to prove
def grapes_value :=
  (1 / 3) * 9 = 2

-- The statement combining the conditions and the problem to be proven
theorem how_many_grapes : apples_to_grapes → grapes_value :=
by
  intro h
  sorry

end how_many_grapes_l293_293254


namespace initial_ducks_count_l293_293778

theorem initial_ducks_count (D : ℕ) 
  (h1 : ∃ (G : ℕ), G = 2 * D - 10) 
  (h2 : ∃ (D_new : ℕ), D_new = D + 4) 
  (h3 : ∃ (G_new : ℕ), G_new = 2 * D - 20) 
  (h4 : ∀ (D_new G_new : ℕ), G_new = D_new + 1) : 
  D = 25 := by
  sorry

end initial_ducks_count_l293_293778


namespace tangent_line_at_01_l293_293137

noncomputable def tangent_line_equation (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_01 : ∃ (m b : ℝ), (m = 1) ∧ (b = 1) ∧ (∀ x, tangent_line_equation x = m * x + b) :=
by
  sorry

end tangent_line_at_01_l293_293137


namespace correct_statements_l293_293084

theorem correct_statements (a b c x : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -2 ∨ x ≥ 6)
  (hb : b = -4 * a)
  (hc : c = -12 * a) : 
  (a < 0) ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ -1/6 < x ∧ x < 1/2) ∧ 
  (a + b + c > 0) :=
by
  sorry

end correct_statements_l293_293084


namespace rose_needs_more_money_l293_293518

def cost_paintbrush : ℝ := 2.40
def cost_paints : ℝ := 9.20
def cost_easel : ℝ := 6.50
def money_rose_has : ℝ := 7.10

theorem rose_needs_more_money : 
  cost_paintbrush + cost_paints + cost_easel - money_rose_has = 11.00 :=
begin
  sorry
end

end rose_needs_more_money_l293_293518


namespace value_of_k_l293_293984

theorem value_of_k (k : ℤ) : (1/2)^(22) * (1/(81 : ℝ))^k = 1/(18 : ℝ)^(22) → k = 11 :=
by
  sorry

end value_of_k_l293_293984


namespace one_cow_one_bag_in_46_days_l293_293485

-- Defining the conditions
def cows_eat_husk (n_cows n_bags n_days : ℕ) := n_cows = n_bags ∧ n_cows = n_days ∧ n_bags = n_days

-- The main theorem to be proved
theorem one_cow_one_bag_in_46_days (h : cows_eat_husk 46 46 46) : 46 = 46 := by
  sorry

end one_cow_one_bag_in_46_days_l293_293485


namespace cos_180_eq_neg1_l293_293620

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l293_293620


namespace least_possible_square_area_l293_293278

theorem least_possible_square_area (measured_length : ℝ) (h : measured_length = 7) : 
  ∃ (actual_length : ℝ), 6.5 ≤ actual_length ∧ actual_length < 7.5 ∧ 
  (∀ (side : ℝ), 6.5 ≤ side ∧ side < 7.5 → side * side ≥ actual_length * actual_length) ∧ 
  actual_length * actual_length = 42.25 :=
by
  sorry

end least_possible_square_area_l293_293278


namespace prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l293_293150

-- Definitions
def total_products := 20
def defective_products := 5

-- Probability of drawing a defective product on the first draw
theorem prob_defective_first_draw : (defective_products / total_products : ℚ) = 1 / 4 :=
sorry

-- Probability of drawing defective products on both the first and the second draws
theorem prob_defective_both_draws : (defective_products / total_products * (defective_products - 1) / (total_products - 1) : ℚ) = 1 / 19 :=
sorry

-- Probability of drawing a defective product on the second draw given that the first was defective
theorem prob_defective_second_given_first : ((defective_products - 1) / (total_products - 1) / (defective_products / total_products) : ℚ) = 4 / 19 :=
sorry

end prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l293_293150


namespace cos_180_eq_minus_1_l293_293632

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l293_293632


namespace trigonometric_relationship_l293_293452

theorem trigonometric_relationship (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = (1 - Real.sin β) / Real.cos β) : 
  2 * α + β = π / 2 := 
sorry

end trigonometric_relationship_l293_293452


namespace find_positive_integer_n_l293_293057

theorem find_positive_integer_n (n : ℕ) (hpos : 0 < n) : 
  (n + 1) ∣ (2 * n^2 + 5 * n) ↔ n = 2 :=
by
  sorry

end find_positive_integer_n_l293_293057


namespace cos_180_eq_neg1_l293_293602

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l293_293602


namespace ribbon_difference_l293_293027

theorem ribbon_difference (L W H : ℕ) (hL : L = 22) (hW : W = 22) (hH : H = 11) : 
  (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) = 22 :=
by
  rw [hL, hW, hH]
  simp
  sorry

end ribbon_difference_l293_293027


namespace football_games_total_l293_293883

def total_football_games_per_season (games_per_month : ℝ) (num_months : ℝ) : ℝ :=
  games_per_month * num_months

theorem football_games_total (games_per_month : ℝ) (num_months : ℝ) (total_games : ℝ) :
  games_per_month = 323.0 ∧ num_months = 17.0 ∧ total_games = 5491.0 →
  total_football_games_per_season games_per_month num_months = total_games :=
by
  intros h
  have h1 : games_per_month = 323.0 := h.1
  have h2 : num_months = 17.0 := h.2.1
  have h3 : total_games = 5491.0 := h.2.2
  rw [h1, h2, h3]
  sorry

end football_games_total_l293_293883


namespace quadruplet_zero_solution_l293_293955

theorem quadruplet_zero_solution (a b c d : ℝ)
  (h1 : (a + b) * (a^2 + b^2) = (c + d) * (c^2 + d^2))
  (h2 : (a + c) * (a^2 + c^2) = (b + d) * (b^2 + d^2))
  (h3 : (a + d) * (a^2 + d^2) = (b + c) * (b^2 + c^2)) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
sorry

end quadruplet_zero_solution_l293_293955


namespace smallest_possible_value_of_n_l293_293729

theorem smallest_possible_value_of_n :
  ∃ n : ℕ, (60 * n = (x + 6) * x * (x + 6) ∧ (x > 0) ∧ gcd 60 n = x + 6) ∧ n = 93 :=
by
  sorry

end smallest_possible_value_of_n_l293_293729


namespace intersection_is_empty_l293_293442

-- Define the domain and range sets
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | 0 < x}

-- The Lean theorem to prove that the intersection of A and B is the empty set
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l293_293442


namespace sum_of_four_squares_l293_293294

theorem sum_of_four_squares (a b c : ℕ) 
    (h1 : 2 * a + b + c = 27)
    (h2 : 2 * b + a + c = 25)
    (h3 : 3 * c + a = 39) : 4 * c = 44 := 
  sorry

end sum_of_four_squares_l293_293294


namespace divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l293_293500

open Nat

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (factors n).eraseDups.length

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

variables {p q m n : ℕ}
variables (hp : is_prime p) (hq : is_prime q) (hdist : p ≠ q) (hm : 0 ≤ m) (hn : 0 ≤ n)

-- a) Prove the number of divisors of pq is 4
theorem divisors_pq : num_divisors (p * q) = 4 :=
sorry

-- b) Prove the number of divisors of p^2 q is 6
theorem divisors_p2q : num_divisors (p^2 * q) = 6 :=
sorry

-- c) Prove the number of divisors of p^2 q^2 is 9
theorem divisors_p2q2 : num_divisors (p^2 * q^2) = 9 :=
sorry

-- d) Prove the number of divisors of p^m q^n is (m + 1)(n + 1)
theorem divisors_pmqn : num_divisors (p^m * q^n) = (m + 1) * (n + 1) :=
sorry

end divisors_pq_divisors_p2q_divisors_p2q2_divisors_pmqn_l293_293500


namespace sum_of_digits_B_l293_293394

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· + ·) 0

def A : ℕ := sum_of_digits (4444 ^ 4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B : 
  sum_of_digits B = 7 := by
    sorry

end sum_of_digits_B_l293_293394


namespace math_problem_l293_293214

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end math_problem_l293_293214


namespace probability_four_friends_same_group_l293_293230

-- Define the conditions of the problem
def total_students : ℕ := 900
def groups : ℕ := 5
def friends : ℕ := 4
def probability_per_group : ℚ := 1 / groups

-- Define the statement we need to prove
theorem probability_four_friends_same_group :
  (probability_per_group * probability_per_group * probability_per_group) = 1 / 125 :=
sorry

end probability_four_friends_same_group_l293_293230


namespace range_of_a_l293_293345

open Real

theorem range_of_a (a : ℝ) (H : ∀ b : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs (x^2 + a * x + b) ≥ 1)) : a ≥ 1 ∨ a ≤ -3 :=
sorry

end range_of_a_l293_293345


namespace calc_fractional_product_l293_293583

theorem calc_fractional_product (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b :=
by
  sorry

end calc_fractional_product_l293_293583


namespace cos_180_degrees_l293_293628

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l293_293628


namespace math_problem_l293_293966

variable (a b c : ℤ)

theorem math_problem
  (h₁ : 3 * a + 4 * b + 5 * c = 0)
  (h₂ : |a| = 1)
  (h₃ : |b| = 1)
  (h₄ : |c| = 1) :
  a * (b + c) = - (3 / 5) :=
sorry

end math_problem_l293_293966


namespace sheets_borrowed_l293_293665

-- Definitions based on conditions
def total_pages : ℕ := 60  -- Hiram's algebra notes are 60 pages
def total_sheets : ℕ := 30  -- printed on 30 sheets of paper
def average_remaining : ℕ := 23  -- the average of the page numbers on all remaining sheets is 23

-- Let S_total be the sum of all page numbers initially
def S_total := (total_pages * (1 + total_pages)) / 2

-- Let c be the number of consecutive sheets borrowed
-- Let b be the number of sheets before the borrowed sheets
-- Calculate S_borrowed based on problem conditions
def S_borrowed (c b : ℕ) := 2 * c * (b + c) + c

-- Calculate the remaining sum and corresponding mean
def remaining_sum (c b : ℕ) := S_total - S_borrowed c b
def remaining_mean (c : ℕ) := (total_sheets * 2 - 2 * c)

-- The theorem we want to prove
theorem sheets_borrowed (c : ℕ) (h : 1830 - S_borrowed c 10 = 23 * (60 - 2 * c)) : c = 15 :=
  sorry

end sheets_borrowed_l293_293665


namespace R_and_D_calculation_l293_293581

-- Define the given conditions and required calculation
def R_and_D_t : ℝ := 2640.92
def delta_APL_t_plus_1 : ℝ := 0.12

theorem R_and_D_calculation :
  (R_and_D_t / delta_APL_t_plus_1) = 22008 := by sorry

end R_and_D_calculation_l293_293581


namespace area_T_l293_293683

variable (T : Set (ℝ × ℝ)) -- T is a region in the plane
variable (A : Matrix (Fin 2) (Fin 2) ℝ) -- A is a 2x2 matrix
variable (detA : ℝ) -- detA is the determinant of A

-- assumptions
axiom area_T : ∃ (area : ℝ), area = 9
axiom matrix_A : A = ![![3, 2], ![-1, 4]]
axiom determinant_A : detA = 14

-- statement to prove
theorem area_T' : ∃ area_T' : ℝ, area_T' = 126 :=
sorry

end area_T_l293_293683


namespace gcd_lcm_product_l293_293312

theorem gcd_lcm_product (a b : ℕ) (h_a : a = 24) (h_b : b = 60) : 
  Nat.gcd a b * Nat.lcm a b = 1440 := by 
  rw [h_a, h_b]
  apply Nat.gcd_mul_lcm
  sorry

end gcd_lcm_product_l293_293312


namespace intersection_point_l293_293804

def satisfies_first_line (p : ℝ × ℝ) : Prop :=
  8 * p.1 - 5 * p.2 = 40

def satisfies_second_line (p : ℝ × ℝ) : Prop :=
  6 * p.1 + 2 * p.2 = 14

theorem intersection_point :
  satisfies_first_line (75 / 23, -64 / 23) ∧ satisfies_second_line (75 / 23, -64 / 23) :=
by 
  sorry

end intersection_point_l293_293804


namespace meal_cost_is_correct_l293_293777

def samosa_quantity : ℕ := 3
def samosa_price : ℝ := 2
def pakora_quantity : ℕ := 4
def pakora_price : ℝ := 3
def mango_lassi_quantity : ℕ := 1
def mango_lassi_price : ℝ := 2
def biryani_quantity : ℕ := 2
def biryani_price : ℝ := 5.5
def naan_quantity : ℕ := 1
def naan_price : ℝ := 1.5

def tip_rate : ℝ := 0.18
def sales_tax_rate : ℝ := 0.07

noncomputable def total_meal_cost : ℝ :=
  let subtotal := (samosa_quantity * samosa_price) + (pakora_quantity * pakora_price) +
                  (mango_lassi_quantity * mango_lassi_price) + (biryani_quantity * biryani_price) +
                  (naan_quantity * naan_price)
  let sales_tax := subtotal * sales_tax_rate
  let total_before_tip := subtotal + sales_tax
  let tip := total_before_tip * tip_rate
  total_before_tip + tip

theorem meal_cost_is_correct : total_meal_cost = 41.04 := by
  sorry

end meal_cost_is_correct_l293_293777


namespace solve_equation_l293_293859

theorem solve_equation :
  ∃ x : ℝ, x = (Real.sqrt (x - 1/x)) + (Real.sqrt (1 - 1/x)) ∧ x = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end solve_equation_l293_293859


namespace find_x_l293_293445

theorem find_x (x : ℝ) (h₀ : ⌊x⌋ * x = 162) : x = 13.5 :=
sorry

end find_x_l293_293445


namespace find_x_l293_293231

/-- Let r be the result of doubling both the base and exponent of a^b, 
and b does not equal to 0. If r equals the product of a^b by x^b,
then x equals 4a. -/
theorem find_x (a b x: ℝ) (h₁ : b ≠ 0) (h₂ : (2*a)^(2*b) = a^b * x^b) : x = 4*a := 
  sorry

end find_x_l293_293231


namespace sin_minus_cos_eq_sqrt2_l293_293065

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l293_293065


namespace boxes_needed_l293_293004

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l293_293004


namespace choir_members_l293_293559

theorem choir_members (n : ℕ) :
  (150 < n) ∧ (n < 250) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 8 = 5) → n = 159 :=
by
  sorry

end choir_members_l293_293559


namespace simplify_expr_l293_293707

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l293_293707


namespace solution_inequality_1_solution_inequality_2_l293_293963

theorem solution_inequality_1 (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ (x < -1 ∨ x > 5) :=
by sorry

theorem solution_inequality_2 (x : ℝ) : 2*x^2 - 5*x + 2 ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by sorry

end solution_inequality_1_solution_inequality_2_l293_293963


namespace a3_equals_neg7_l293_293503

-- Definitions based on given conditions
noncomputable def a₁ := -11
noncomputable def d : ℤ := sorry -- this is derived but unknown presently
noncomputable def a(n : ℕ) : ℤ := a₁ + (n - 1) * d

axiom condition : a 4 + a 6 = -6

-- The proof problem statement
theorem a3_equals_neg7 : a 3 = -7 :=
by
  have h₁ : a₁ = -11 := rfl
  have h₂ : a 4 + a 6 = -6 := condition
  sorry

end a3_equals_neg7_l293_293503


namespace find_E_equals_2023_l293_293354

noncomputable def proof : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ (a^2 * (b + c) = 2023) ∧ (b^2 * (c + a) = 2023) ∧ (c^2 * (a + b) = 2023)

theorem find_E_equals_2023 : proof :=
by
  sorry

end find_E_equals_2023_l293_293354


namespace toby_total_time_l293_293885

def speed_unloaded := 20 -- Speed of Toby pulling unloaded sled in mph
def speed_loaded := 10   -- Speed of Toby pulling loaded sled in mph

def distance_part1 := 180 -- Distance for the first part (loaded sled) in miles
def distance_part2 := 120 -- Distance for the second part (unloaded sled) in miles
def distance_part3 := 80  -- Distance for the third part (loaded sled) in miles
def distance_part4 := 140 -- Distance for the fourth part (unloaded sled) in miles

def time_part1 := distance_part1 / speed_loaded -- Time for the first part in hours
def time_part2 := distance_part2 / speed_unloaded -- Time for the second part in hours
def time_part3 := distance_part3 / speed_loaded -- Time for the third part in hours
def time_part4 := distance_part4 / speed_unloaded -- Time for the fourth part in hours

def total_time := time_part1 + time_part2 + time_part3 + time_part4 -- Total time in hours

theorem toby_total_time : total_time = 39 :=
by 
  sorry

end toby_total_time_l293_293885


namespace greatest_possible_average_speed_l293_293434

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem greatest_possible_average_speed :
  ∀ (o₁ o₂ : ℕ) (v_max t : ℝ), 
  is_palindrome o₁ → 
  is_palindrome o₂ → 
  o₁ = 12321 → 
  t = 2 ∧ v_max = 65 → 
  (∃ d, d = o₂ - o₁ ∧ d / t <= v_max) → 
  d / t = v_max :=
sorry

end greatest_possible_average_speed_l293_293434


namespace weekly_allowance_l293_293507

theorem weekly_allowance
  (video_game_cost : ℝ)
  (sales_tax_percentage : ℝ)
  (weeks_to_save : ℕ)
  (total_with_tax : ℝ)
  (total_savings : ℝ) :
  video_game_cost = 50 →
  sales_tax_percentage = 0.10 →
  weeks_to_save = 11 →
  total_with_tax = video_game_cost * (1 + sales_tax_percentage) →
  total_savings = weeks_to_save * (0.5 * total_savings) →
  total_savings = total_with_tax →
  total_savings = 55 :=
by
  intros
  sorry

end weekly_allowance_l293_293507


namespace cube_side_ratio_l293_293387

theorem cube_side_ratio (a b : ℝ) (h : (6 * a^2) / (6 * b^2) = 36) : a / b = 6 :=
by
  sorry

end cube_side_ratio_l293_293387


namespace volume_between_concentric_spheres_l293_293882

theorem volume_between_concentric_spheres
  (r1 r2 : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 10) :
  (4 / 3 * Real.pi * r2^3 - 4 / 3 * Real.pi * r1^3) = (3500 / 3) * Real.pi :=
by
  rw [h_r1, h_r2]
  sorry

end volume_between_concentric_spheres_l293_293882


namespace incorrect_option_l293_293656

noncomputable def f : ℝ → ℝ := sorry
def is_odd (g : ℝ → ℝ) := ∀ x, g (-(2 * x + 1)) = -g (2 * x + 1)
def is_even (g : ℝ → ℝ) := ∀ x, g (x + 2) = g (-x + 2)

theorem incorrect_option (h₁ : is_odd f) (h₂ : is_even f) (h₃ : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = 3 - x) :
  ¬ (∀ x, f x = f (-x - 2)) :=
by
  sorry

end incorrect_option_l293_293656


namespace discount_equivalence_l293_293574

variable (Original_Price : ℝ)

theorem discount_equivalence (h1 : Real) (h2 : Real) :
  (h1 = 0.5 * Original_Price) →
  (h2 = 0.7 * h1) →
  (Original_Price - h2) / Original_Price = 0.65 :=
by
  intros
  sorry

end discount_equivalence_l293_293574


namespace ratio_of_potatoes_l293_293111

-- Definitions as per conditions
def initial_potatoes : ℕ := 300
def given_to_gina : ℕ := 69
def remaining_potatoes : ℕ := 47
def k : ℕ := 2  -- Identify k is 2 based on the ratio

-- Calculate given_to_tom and total given away
def given_to_tom : ℕ := k * given_to_gina
def given_to_anne : ℕ := given_to_tom / 3

-- Arithmetical conditions derived from the problem
def total_given_away : ℕ := given_to_gina + given_to_tom + given_to_anne + remaining_potatoes

-- Proof statement to show the ratio between given_to_tom and given_to_gina is 2
theorem ratio_of_potatoes :
  k = 2 → total_given_away = initial_potatoes → given_to_tom / given_to_gina = 2 := by
  intros h1 h2
  sorry

end ratio_of_potatoes_l293_293111


namespace total_pieces_of_gum_and_candy_l293_293861

theorem total_pieces_of_gum_and_candy 
  (packages_A : ℕ) (pieces_A : ℕ) (packages_B : ℕ) (pieces_B : ℕ) 
  (packages_C : ℕ) (pieces_C : ℕ) (packages_X : ℕ) (pieces_X : ℕ)
  (packages_Y : ℕ) (pieces_Y : ℕ) 
  (hA : packages_A = 10) (hA_pieces : pieces_A = 4)
  (hB : packages_B = 5) (hB_pieces : pieces_B = 8)
  (hC : packages_C = 13) (hC_pieces : pieces_C = 12)
  (hX : packages_X = 8) (hX_pieces : pieces_X = 6)
  (hY : packages_Y = 6) (hY_pieces : pieces_Y = 10) : 
  packages_A * pieces_A + packages_B * pieces_B + packages_C * pieces_C + 
  packages_X * pieces_X + packages_Y * pieces_Y = 344 := 
by
  sorry

end total_pieces_of_gum_and_candy_l293_293861


namespace slower_bike_longer_time_by_1_hour_l293_293007

/-- Speed of the slower bike in kmph -/
def speed_slow : ℕ := 60

/-- Speed of the faster bike in kmph -/
def speed_fast : ℕ := 64

/-- Distance both bikes travel in km -/
def distance : ℕ := 960

/-- Time taken to travel the distance by a bike going at a certain speed -/
def time (speed : ℕ) : ℕ :=
  distance / speed

/-- Proof that the slower bike takes 1 hour longer to cover the distance compared to the faster bike -/
theorem slower_bike_longer_time_by_1_hour : 
  (time speed_slow) = (time speed_fast) + 1 := by
sorry

end slower_bike_longer_time_by_1_hour_l293_293007


namespace find_second_term_l293_293210

theorem find_second_term 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h_sum : ∀ n, S n = n * (2 * n + 1))
  (h_S1 : S 1 = a 1) 
  (h_S2 : S 2 = a 1 + a 2) 
  (h_a1 : a 1 = 3) : 
  a 2 = 7 := 
sorry

end find_second_term_l293_293210


namespace length_difference_squares_l293_293133

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end length_difference_squares_l293_293133


namespace sum_of_reciprocal_squares_of_roots_l293_293258

theorem sum_of_reciprocal_squares_of_roots (a b c : ℝ) 
    (h_roots : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c) :
    a + b + c = 6 ∧ ab + bc + ca = 11 ∧ abc = 6 → 
    (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 := 
by
  sorry

end sum_of_reciprocal_squares_of_roots_l293_293258


namespace find_k_l293_293652

-- Definitions for the line, circle, point, and distance
variable (x y k : ℝ)

def line (k : ℝ) : ℝ := k * x + y + 4
def circle_center : ℝ × ℝ := (0, 1)
def circle_radius : ℝ := 1
def tangent_length : ℝ := 2

-- The function to calculate the distance from a point to a line
def distance_point_to_line (k : ℝ) (px py : ℝ) : ℝ :=
  abs (k * px + py + 4) / (sqrt (k^2 + 1))

-- The math problem translated to a proof problem
theorem find_k (h : distance_point_to_line k 0 1 = sqrt 5) (hk : k > 0) :
  k = 2 :=
sorry  -- proof is not required, hence marked as sorry for now

end find_k_l293_293652


namespace linear_function_through_point_parallel_line_l293_293534

noncomputable def function_expr (x : ℝ) : ℝ := 2 * x + 3

def point_A : ℝ × ℝ := (-2, -1)

def parallel_line (x : ℝ) : ℝ := 2 * x - 3

theorem linear_function_through_point_parallel_line :
  ∃ b : ℝ, (∀ x : ℝ, function_expr x = 2 * x + b) ∧ (function_expr (fst point_A) = snd point_A) :=
by
  use 3
  split
  . intro x
    refl
  . simp [function_expr, point_A]
    sorry

end linear_function_through_point_parallel_line_l293_293534


namespace multiplier_for_doberman_puppies_l293_293366

theorem multiplier_for_doberman_puppies 
  (D : ℕ) (S : ℕ) (M : ℝ) 
  (hD : D = 20) 
  (hS : S = 55) 
  (h : D * M + (D - S) = 90) : 
  M = 6.25 := 
by 
  sorry

end multiplier_for_doberman_puppies_l293_293366


namespace find_a_values_l293_293385

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l293_293385


namespace arithmetic_sequence_general_formula_l293_293828

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, 0 < n → (a n - 2 * a (n + 1) + a (n + 2) = 0)) : ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l293_293828


namespace area_of_pentagon_PTRSQ_l293_293510

theorem area_of_pentagon_PTRSQ (PQRS : Type) [geometry PQRS]
  {P Q R S T : PQRS} 
  (h1 : square P Q R S) 
  (h2 : perp PT TR) 
  (h3 : distance P T = 5) 
  (h4 : distance T R = 12) : 
  area_pentagon PTRSQ = 139 :=
sorry

end area_of_pentagon_PTRSQ_l293_293510


namespace micah_water_intake_l293_293350

def morning : ℝ := 1.5
def early_afternoon : ℝ := 2 * morning
def late_afternoon : ℝ := 3 * morning
def evening : ℝ := late_afternoon - 0.25 * late_afternoon
def night : ℝ := 2 * evening
def total_water_intake : ℝ := morning + early_afternoon + late_afternoon + evening + night

theorem micah_water_intake :
  total_water_intake = 19.125 := by
  sorry

end micah_water_intake_l293_293350


namespace minimum_value_l293_293346

theorem minimum_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) : 
  x^2 + 8 * x * y + 16 * y^2 + 4 * z^2 ≥ 192 := 
  sorry

end minimum_value_l293_293346


namespace find_lower_percentage_l293_293171

theorem find_lower_percentage (P : ℝ) : 
  (12000 * 0.15 * 2 - 720 = 12000 * (P / 100) * 2) → P = 12 := by
  sorry

end find_lower_percentage_l293_293171


namespace max_value_of_perfect_sequence_l293_293183

def isPerfectSequence (c : ℕ → ℕ) : Prop := ∀ n m : ℕ, 1 ≤ m ∧ m ≤ (Finset.range (n + 1)).sum (fun k => c k) → 
  ∃ (a : ℕ → ℕ), m = (Finset.range (n + 1)).sum (fun k => c k / a k)

theorem max_value_of_perfect_sequence (n : ℕ) : 
  ∃ c : ℕ → ℕ, isPerfectSequence c ∧
    (∀ i, i ≤ n → c i ≤ if i = 1 then 2 else 4 * 3^(i - 2)) ∧
    c n = if n = 1 then 2 else 4 * 3^(n - 2) :=
by
  sorry

end max_value_of_perfect_sequence_l293_293183


namespace find_multiple_l293_293295

/-- 
Given:
1. Hank Aaron hit 755 home runs.
2. Dave Winfield hit 465 home runs.
3. Hank Aaron has 175 fewer home runs than a certain multiple of the number that Dave Winfield has.

Prove:
The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to is 2.
-/
def multiple_of_dave_hr (ha_hr dw_hr diff : ℕ) (m : ℕ) : Prop :=
  ha_hr + diff = m * dw_hr

theorem find_multiple :
  multiple_of_dave_hr 755 465 175 2 :=
by
  sorry

end find_multiple_l293_293295


namespace combined_savings_after_four_weeks_l293_293356

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l293_293356


namespace probability_standard_parts_l293_293644

theorem probability_standard_parts (parts_machine1 parts_machine2 standard_parts_machine1 standard_parts_machine2 : ℕ)
  (h1 : parts_machine1 = 200) (h2 : parts_machine2 = 300)
  (h3 : standard_parts_machine1 = 190) (h4 : standard_parts_machine2 = 280) :
  let total_parts := parts_machine1 + parts_machine2,
      total_standard_parts := standard_parts_machine1 + standard_parts_machine2,
      P_A := (total_standard_parts : ℝ) / total_parts,
      P_A_given_B := (standard_parts_machine1 : ℝ) / parts_machine1,
      P_A_given_not_B := (standard_parts_machine2 : ℝ) / parts_machine2
  in P_A = 0.94 ∧ P_A_given_B = 0.95 ∧ P_A_given_not_B = 14/15 := 
by {
  sorry
}

end probability_standard_parts_l293_293644


namespace solve_for_x_l293_293054

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  5 * y ^ 2 + 3 * y + 2 = 3 * (8 * x ^ 2 + y + 1) ↔ x = 1 / Real.sqrt 21 ∨ x = -1 / Real.sqrt 21 :=
by
  sorry

end solve_for_x_l293_293054


namespace gazprom_R_and_D_expenditure_l293_293580

def research_and_development_expenditure (R_t : ℝ) (delta_APL_t1 : ℝ) : ℝ :=
  R_t / delta_APL_t1

theorem gazprom_R_and_D_expenditure :
  research_and_development_expenditure 2640.92 0.12 = 22008 :=
by
  sorry

end gazprom_R_and_D_expenditure_l293_293580


namespace sqrt_inequality_l293_293299

theorem sqrt_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : 
  x^2 + y^2 + 1 ≤ Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) :=
sorry

end sqrt_inequality_l293_293299


namespace cosine_180_degree_l293_293616

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l293_293616


namespace min_value_of_x_squared_plus_6x_l293_293888

theorem min_value_of_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  sorry
end

end min_value_of_x_squared_plus_6x_l293_293888


namespace diagonal_length_of_octagon_l293_293421

theorem diagonal_length_of_octagon 
  (r : ℝ) (s : ℝ) (has_symmetry_axes : ℕ) 
  (inscribed : r = 6) (side_length : s = 5) 
  (symmetry_condition : has_symmetry_axes = 4) : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 40 := 
by 
  sorry

end diagonal_length_of_octagon_l293_293421


namespace move_line_down_l293_293576

theorem move_line_down (x y : ℝ) : (y = -3 * x + 5) → (y = -3 * x + 2) :=
by
  sorry

end move_line_down_l293_293576


namespace find_number_l293_293544

theorem find_number (N : ℕ) (k : ℕ) (Q : ℕ)
  (h1 : N = 9 * k)
  (h2 : Q = 25 * 9 + 7)
  (h3 : N / 9 = Q) :
  N = 2088 :=
by
  sorry

end find_number_l293_293544


namespace problem_x_y_z_l293_293983

theorem problem_x_y_z (x y z : ℕ) (h1 : xy + z = 47) (h2 : yz + x = 47) (h3 : xz + y = 47) : x + y + z = 48 :=
sorry

end problem_x_y_z_l293_293983


namespace find_a_of_parabola_l293_293873

theorem find_a_of_parabola (a b c : ℤ) (h_vertex : (2, 5) = (2, 5)) (h_point : 8 = a * (3 - 2) ^ 2 + 5) :
  a = 3 :=
sorry

end find_a_of_parabola_l293_293873


namespace solve_trigonometric_equation_count_solutions_l293_293524

theorem solve_trigonometric_equation :
  ∀ x : ℝ, 2000 ≤ x ∧ x ≤ 3000 →
  2 * real.sqrt 2 * real.sin (real.pi * x / 4) ^ 3 = real.sin (real.pi / 4 * (1 + x)) →
  ∃! (n : ℤ), 500 ≤ n ∧ n ≤ 749 ∧ x = 1 + 4 * n :=
sorry

-- Count the unique solutions within the given range
theorem count_solutions :
  let num_solutions := (749 - 500 + 1 : ℤ) in
  num_solutions = 250 :=
by
  simp [Int.ofNat_sub, Int.add_one, Int.ofNat_one, Int.ofNat_add]
  linarith

end solve_trigonometric_equation_count_solutions_l293_293524


namespace positive_difference_l293_293148

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l293_293148


namespace correct_calculation_result_l293_293407

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end correct_calculation_result_l293_293407


namespace average_primes_4_to_15_l293_293198

theorem average_primes_4_to_15 :
  (5 + 7 + 11 + 13) / 4 = 9 :=
by sorry

end average_primes_4_to_15_l293_293198


namespace cube_surface_area_l293_293224

theorem cube_surface_area (PQ a b : ℝ) (x : ℝ) 
  (h1 : PQ = a / 2) 
  (h2 : PQ = Real.sqrt (3 * x^2)) : 
  b = 6 * x^2 → b = a^2 / 2 := 
by
  intros h_surface
  -- sorry is added here to skip the proof step and ensure the code builds successfully.
  sorry

end cube_surface_area_l293_293224


namespace simplify_expr_l293_293708

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l293_293708


namespace four_digit_number_l293_293724

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end four_digit_number_l293_293724


namespace find_range_of_a_l293_293816

noncomputable def A (a : ℝ) := { x : ℝ | 1 ≤ x ∧ x ≤ a}
noncomputable def B (a : ℝ) := { y : ℝ | ∃ x : ℝ, y = 5 * x - 6 ∧ 1 ≤ x ∧ x ≤ a }
noncomputable def C (a : ℝ) := { m : ℝ | ∃ x : ℝ, m = x^2 ∧ 1 ≤ x ∧ x ≤ a }

theorem find_range_of_a (a : ℝ) (h : B a ∩ C a = C a) : 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end find_range_of_a_l293_293816


namespace pointA_in_second_quadrant_l293_293839

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l293_293839


namespace total_pages_l293_293369

theorem total_pages (history_pages geography_additional math_factor science_factor : ℕ) 
  (h1 : history_pages = 160)
  (h2 : geography_additional = 70)
  (h3 : math_factor = 2)
  (h4 : science_factor = 2) 
  : let geography_pages := history_pages + geography_additional in
    let sum_history_geography := history_pages + geography_pages in
    let math_pages := sum_history_geography / math_factor in
    let science_pages := history_pages * science_factor in
    history_pages + geography_pages + math_pages + science_pages = 905 :=
by
  sorry

end total_pages_l293_293369


namespace even_number_representation_l293_293072

-- Definitions for conditions
def even_number (k : Int) : Prop := ∃ m : Int, k = 2 * m
def perfect_square (n : Int) : Prop := ∃ p : Int, n = p * p
def sum_representation (a b : Int) : Prop := ∃ k : Int, a + b = 2 * k ∧ perfect_square (a * b)
def difference_representation (d k e : Int) : Prop := d * (d - 2 * k) = e * e

-- The theorem statement
theorem even_number_representation {k : Int} (hk : even_number k) :
  (∃ a b : Int, sum_representation a b ∧ 2 * k = a + b) ∨
  (∃ d e : Int, difference_representation d k e ∧ d ≠ 0) :=
sorry

end even_number_representation_l293_293072


namespace min_value_l293_293460

theorem min_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 1) :
  (a + 1)^2 + 4 * b^2 + 9 * c^2 ≥ 144 / 49 :=
sorry

end min_value_l293_293460


namespace sum_of_three_consecutive_even_integers_l293_293733

theorem sum_of_three_consecutive_even_integers : 
  ∃ (n : ℤ), n * (n + 2) * (n + 4) = 480 → n + (n + 2) + (n + 4) = 24 :=
by
  sorry

end sum_of_three_consecutive_even_integers_l293_293733


namespace example_theorem_l293_293063

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l293_293063


namespace linear_regression_intercept_l293_293762

theorem linear_regression_intercept :
  let x_values := [1, 2, 3, 4, 5]
  let y_values := [0.5, 0.8, 1.0, 1.2, 1.5]
  let x_mean := (x_values.sum / x_values.length : ℝ)
  let y_mean := (y_values.sum / y_values.length : ℝ)
  let slope := 0.24
  (x_mean = 3) →
  (y_mean = 1) →
  y_mean = slope * x_mean + 0.28 :=
by
  sorry

end linear_regression_intercept_l293_293762


namespace cos_180_eq_neg_one_l293_293611

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l293_293611


namespace general_term_formula_T_n_less_than_one_sixth_l293_293086

noncomputable def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := if n = 0 then 0 else 2*n + 1

def b (n : ℕ) : ℕ := if n = 0 then 0 else 1 / (a n) * (a (n+1))

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k => (b k : ℝ))

theorem general_term_formula (n : ℕ) (hn : n ≠ 0) : 
  a n = 2*n + 1 :=
by sorry

theorem T_n_less_than_one_sixth (n : ℕ) : 
  T n < (1 / 6 : ℝ) :=
by sorry

end general_term_formula_T_n_less_than_one_sixth_l293_293086


namespace arithmetic_sequence_first_term_l293_293878

theorem arithmetic_sequence_first_term (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 500) 
  (h2 : 30 * (2 * a + 179 * d) = 2900) : 
  a = -34 / 3 := 
sorry

end arithmetic_sequence_first_term_l293_293878


namespace base_five_product_l293_293270

open Nat

/-- Definition of the base 5 representation of 131 and 21 --/
def n131 := 1 * 5^2 + 3 * 5^1 + 1 * 5^0
def n21 := 2 * 5^1 + 1 * 5^0

/-- Definition of the expected result in base 5 --/
def expected_result := 3 * 5^3 + 2 * 5^2 + 5 * 5^1 + 1 * 5^0

/-- Claim to prove that the product of 131_5 and 21_5 equals 3251_5 --/
theorem base_five_product : n131 * n21 = expected_result := by sorry

end base_five_product_l293_293270


namespace parallel_lines_l293_293841

open Real -- Open the real number namespace

/-- Definition of line l1 --/
def line_l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y - 1 = 0

/-- Definition of line l2 --/
def line_l2 (a : ℝ) (x y : ℝ) := x + (a + 1) * y + 4 = 0

/-- The proof statement --/
theorem parallel_lines (a : ℝ) : (a = 1) → (line_l1 a x y) → (line_l2 a x y) := 
sorry

end parallel_lines_l293_293841


namespace cos_180_eq_neg_one_l293_293600

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l293_293600


namespace students_use_red_color_l293_293418

theorem students_use_red_color
  (total_students : ℕ)
  (students_use_green : ℕ)
  (students_use_both : ℕ)
  (total_students_eq : total_students = 70)
  (students_use_green_eq : students_use_green = 52)
  (students_use_both_eq : students_use_both = 38) :
  ∃ (students_use_red : ℕ), students_use_red = 56 :=
by
  -- We will skip the proof part as specified
  sorry

end students_use_red_color_l293_293418


namespace speed_of_second_train_l293_293886

-- Definitions of given conditions
def length_first_train : ℝ := 60 
def length_second_train : ℝ := 280 
def speed_first_train : ℝ := 30 
def time_clear : ℝ := 16.998640108791296 

-- The Lean statement for the proof problem
theorem speed_of_second_train : 
  let relative_distance_km := (length_first_train + length_second_train) / 1000
  let time_clear_hr := time_clear / 3600
  (speed_first_train + (relative_distance_km / time_clear_hr)) = 72.00588235294118 → 
  ∃ V : ℝ, V = 42.00588235294118 :=
by 
  -- Placeholder for the proof
  sorry

end speed_of_second_train_l293_293886


namespace percentage_of_same_grade_is_48_l293_293423

def students_with_same_grade (grades : ℕ × ℕ → ℕ) : ℕ :=
  grades (0, 0) + grades (1, 1) + grades (2, 2) + grades (3, 3) + grades (4, 4)

theorem percentage_of_same_grade_is_48
  (grades : ℕ × ℕ → ℕ)
  (h : grades (0, 0) = 3 ∧ grades (1, 1) = 6 ∧ grades (2, 2) = 8 ∧ grades (3, 3) = 4 ∧ grades (4, 4) = 3)
  (total_students : ℕ) (h_students : total_students = 50) :
  (students_with_same_grade grades / 50 : ℚ) * 100 = 48 :=
by
  sorry

end percentage_of_same_grade_is_48_l293_293423


namespace average_length_of_remaining_strings_l293_293741

theorem average_length_of_remaining_strings :
  ∀ (n_cat : ℕ) 
    (avg_len_total avg_len_one_fourth avg_len_one_third : ℝ)
    (total_length total_length_one_fourth total_length_one_third remaining_length : ℝ),
    n_cat = 12 →
    avg_len_total = 90 →
    avg_len_one_fourth = 75 →
    avg_len_one_third = 65 →
    total_length = n_cat * avg_len_total →
    total_length_one_fourth = (n_cat / 4) * avg_len_one_fourth →
    total_length_one_third = (n_cat / 3) * avg_len_one_third →
    remaining_length = total_length - (total_length_one_fourth + total_length_one_third) →
    remaining_length / (n_cat - (n_cat / 4 + n_cat / 3)) = 119 :=
by sorry

end average_length_of_remaining_strings_l293_293741


namespace square_placement_conditions_l293_293339

-- Definitions for natural numbers at vertices and center
def top_left := 14
def top_right := 6
def bottom_right := 15
def bottom_left := 35
def center := 210

theorem square_placement_conditions :
  (∃ gcd1 > 1, gcd1 = Nat.gcd top_left top_right) ∧
  (∃ gcd2 > 1, gcd2 = Nat.gcd top_right bottom_right) ∧
  (∃ gcd3 > 1, gcd3 = Nat.gcd bottom_right bottom_left) ∧
  (∃ gcd4 > 1, gcd4 = Nat.gcd bottom_left top_left) ∧
  (Nat.gcd top_left bottom_right = 1) ∧
  (Nat.gcd top_right bottom_left = 1) ∧
  (Nat.gcd top_left center > 1) ∧
  (Nat.gcd top_right center > 1) ∧
  (Nat.gcd bottom_right center > 1) ∧
  (Nat.gcd bottom_left center > 1) 
 := by
sorry

end square_placement_conditions_l293_293339


namespace ab_div_c_eq_one_l293_293413

theorem ab_div_c_eq_one (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hne1 : A ≠ B) (hne2 : A ≠ C) (hne3 : B ≠ C) :
  (1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / 1))) → (A + B) / C = 1 :=
by sorry

end ab_div_c_eq_one_l293_293413


namespace rain_puddle_depth_l293_293696

theorem rain_puddle_depth
  (rain_rate : ℝ) (wait_time : ℝ) (puddle_area : ℝ) 
  (h_rate : rain_rate = 10) (h_time : wait_time = 3) (h_area : puddle_area = 300) :
  ∃ (depth : ℝ), depth = rain_rate * wait_time :=
by
  use 30
  simp [h_rate, h_time]
  sorry

end rain_puddle_depth_l293_293696


namespace eight_b_equals_neg_eight_l293_293092

theorem eight_b_equals_neg_eight (a b : ℤ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := 
by
  sorry

end eight_b_equals_neg_eight_l293_293092


namespace minimum_value_real_l293_293890

theorem minimum_value_real (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  use -9,
  sorry

end minimum_value_real_l293_293890


namespace checkerboard_red_squares_l293_293781

/-- Define the properties of the checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (colors : ℕ → ℕ → String)
  (corner_color : String)

/-- Our checkerboard patterning function -/
def checkerboard_colors (i j : ℕ) : String :=
  match (i + j) % 3 with
  | 0 => "blue"
  | 1 => "yellow"
  | _ => "red"

/-- Our checkerboard of size 33x33 -/
def chubby_checkerboard : Checkerboard :=
  { size := 33,
    colors := checkerboard_colors,
    corner_color := "blue" }

/-- Proof that the number of red squares is 363 -/
theorem checkerboard_red_squares (b : Checkerboard) (h1 : b.size = 33) (h2 : b.colors = checkerboard_colors) : ∃ n, n = 363 :=
  by sorry

end checkerboard_red_squares_l293_293781


namespace area_larger_sphere_red_is_83_point_25_l293_293916

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l293_293916


namespace smallest_positive_shift_l293_293526

noncomputable def g : ℝ → ℝ := sorry

theorem smallest_positive_shift
  (H1 : ∀ x, g (x - 20) = g x) : 
  ∃ a > 0, (∀ x, g ((x - a) / 10) = g (x / 10)) ∧ a = 200 :=
sorry

end smallest_positive_shift_l293_293526


namespace sum_of_squares_of_products_eq_factorial_minus_one_l293_293436

noncomputable def sum_of_squares_of_products : ℕ → ℕ
| 0       := 0
| (n + 1) := (Finset.powerset (Finset.range (n + 1))).filter (λ s, ∀ x ∈ s, ∀ y ∈ s, y ≤ x + 1 → y = x → x = y).sum (λ s, s.prod (λ x, (x + 1)^2))

theorem sum_of_squares_of_products_eq_factorial_minus_one (n : ℕ) :
  sum_of_squares_of_products n = (n + 1)! - 1 := 
sorry

end sum_of_squares_of_products_eq_factorial_minus_one_l293_293436


namespace arcsin_one_half_l293_293784

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end arcsin_one_half_l293_293784


namespace range_of_m_l293_293320

open Set Real

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B (m : ℝ) := {x : ℝ | -1 < x ∧ x < m}

theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → 3 < m :=
by sorry

end range_of_m_l293_293320


namespace cos_180_eq_neg1_l293_293603

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l293_293603


namespace max_value_2ac_minus_abc_l293_293854

theorem max_value_2ac_minus_abc (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 7) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c <= 4) : 
  2 * a * c - a * b * c ≤ 28 :=
sorry

end max_value_2ac_minus_abc_l293_293854


namespace sum_dihedral_angles_gt_360_l293_293248

-- Define the structure Tetrahedron
structure Tetrahedron (α : Type*) :=
  (A B C D : α)

-- Define the dihedral angles function
noncomputable def sum_dihedral_angles {α : Type*} (T : Tetrahedron α) : ℝ := 
  -- Placeholder for the actual sum of dihedral angles of T
  sorry

-- Statement of the problem
theorem sum_dihedral_angles_gt_360 {α : Type*} (T : Tetrahedron α) :
  sum_dihedral_angles T > 360 := 
sorry

end sum_dihedral_angles_gt_360_l293_293248


namespace find_m_values_l293_293466

def has_unique_solution (m : ℝ) (A : Set ℝ) : Prop :=
  ∀ x1 x2, x1 ∈ A → x2 ∈ A → x1 = x2

theorem find_m_values :
  {m : ℝ | ∃ A : Set ℝ, has_unique_solution m A ∧ (A = {x | m * x^2 + 2 * x + 3 = 0})} = {0, 1/3} :=
by
  sorry

end find_m_values_l293_293466


namespace div_by_eleven_l293_293232

theorem div_by_eleven (a b : ℤ) (h : (a^2 + 9 * a * b + b^2) % 11 = 0) : 
  (a^2 - b^2) % 11 = 0 :=
sorry

end div_by_eleven_l293_293232


namespace chessboard_L_T_equivalence_l293_293031

theorem chessboard_L_T_equivalence (n : ℕ) :
  ∃ L_count T_count : ℕ, 
  (L_count = T_count) ∧ -- number of L-shaped pieces is equal to number of T-shaped pieces
  (L_count + T_count = n * (n + 1)) := 
sorry

end chessboard_L_T_equivalence_l293_293031


namespace total_workers_in_workshop_l293_293410

-- Definition of average salary calculation
def average_salary (total_salary : ℕ) (workers : ℕ) : ℕ := total_salary / workers

theorem total_workers_in_workshop :
  ∀ (W T R : ℕ),
  T = 5 →
  average_salary ((W - T) * 750) (W - T) = 700 →
  average_salary (T * 900) T = 900 →
  average_salary (W * 750) W = 750 →
  W = T + R →
  W = 20 :=
by
  sorry

end total_workers_in_workshop_l293_293410


namespace age_ratio_l293_293023

theorem age_ratio (x : ℕ) (h : (5 * x - 4) = (3 * x + 4)) :
    (5 * x + 4) / (3 * x - 4) = 3 :=
by sorry

end age_ratio_l293_293023


namespace find_a_l293_293815

open Set

theorem find_a :
  ∀ (A B : Set ℕ) (a : ℕ),
    A = {1, 2, 3} →
    B = {2, a} →
    A ∪ B = {0, 1, 2, 3} →
    a = 0 :=
by
  intros A B a hA hB hUnion
  rw [hA, hB] at hUnion
  sorry

end find_a_l293_293815


namespace cos_180_eq_neg_one_l293_293598

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l293_293598


namespace watermelon_seeds_l293_293275

variable (G Y B : ℕ)

theorem watermelon_seeds (h1 : Y = 3 * G) (h2 : G > B) (h3 : B = 300) (h4 : G + Y + B = 1660) : G = 340 := by
  sorry

end watermelon_seeds_l293_293275


namespace number_of_true_propositions_l293_293391

variable {a b c : ℝ}

theorem number_of_true_propositions :
  (2 = (if (a > b → a * c ^ 2 > b * c ^ 2) then 1 else 0) +
       (if (a * c ^ 2 > b * c ^ 2 → a > b) then 1 else 0) +
       (if (¬(a * c ^ 2 > b * c ^ 2) → ¬(a > b)) then 1 else 0) +
       (if (¬(a > b) → ¬(a * c ^ 2 > b * c ^ 2)) then 1 else 0)) :=
sorry

end number_of_true_propositions_l293_293391


namespace no_solution_for_system_l293_293353

theorem no_solution_for_system (x y z : ℝ) 
  (h1 : |x| < |y - z|) 
  (h2 : |y| < |z - x|) 
  (h3 : |z| < |x - y|) : 
  false :=
sorry

end no_solution_for_system_l293_293353


namespace cosine_180_degree_l293_293615

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l293_293615


namespace original_number_of_people_l293_293491

variable (x : ℕ)
-- Conditions
axiom one_third_left : x / 3 > 0
axiom half_dancing : 18 = x / 3

-- Theorem Statement
theorem original_number_of_people (x : ℕ) (one_third_left : x / 3 > 0) (half_dancing : 18 = x / 3) : x = 54 := sorry

end original_number_of_people_l293_293491


namespace inequality_solution_set_l293_293991

theorem inequality_solution_set (m n : ℝ) 
    (h₁ : ∀ x : ℝ, mx - n > 0 ↔ x < 1 / 3) 
    (h₂ : m + n < 0) 
    (h₃ : m = 3 * n) 
    (h₄ : n < 0) : 
    ∀ x : ℝ, (m + n) * x < n - m ↔ x > -1 / 2 :=
by
  sorry

end inequality_solution_set_l293_293991


namespace percentage_of_water_in_first_liquid_l293_293914

theorem percentage_of_water_in_first_liquid (x : ℝ) 
  (h1 : 0 < x ∧ x ≤ 1)
  (h2 : 0.35 = 0.35)
  (h3 : 10 = 10)
  (h4 : 4 = 4)
  (h5 : 0.24285714285714285 = 0.24285714285714285) :
  ((10 * x + 4 * 0.35) / (10 + 4) = 0.24285714285714285) → (x = 0.2) :=
sorry

end percentage_of_water_in_first_liquid_l293_293914


namespace total_number_of_boys_in_all_class_sections_is_380_l293_293486

theorem total_number_of_boys_in_all_class_sections_is_380 :
  let students_section1 := 160
  let students_section2 := 200
  let students_section3 := 240
  let girls_section1 := students_section1 / 4
  let boys_section1 := students_section1 - girls_section1
  let boys_section2 := (3 / 5) * students_section2
  let total_parts := 7 + 5
  let boys_section3 := (7 / total_parts) * students_section3
  boys_section1 + boys_section2 + boys_section3 = 380 :=
sorry

end total_number_of_boys_in_all_class_sections_is_380_l293_293486


namespace div_by_133_l293_293352

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end div_by_133_l293_293352


namespace simplify_fraction_l293_293713

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l293_293713


namespace weight_of_D_l293_293226

open Int

def weights (A B C D : Int) : Prop :=
  A < B ∧ B < C ∧ C < D ∧ 
  A + B = 45 ∧ A + C = 49 ∧ A + D = 55 ∧ 
  B + C = 54 ∧ B + D = 60 ∧ C + D = 64

theorem weight_of_D {A B C D : Int} (h : weights A B C D) : D = 35 := 
  by
    sorry

end weight_of_D_l293_293226


namespace cos_180_eq_neg_one_l293_293594

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l293_293594


namespace point_A_in_second_quadrant_l293_293834

-- Define the conditions as functions
def x_coord : ℝ := -3
def y_coord : ℝ := 4

-- Define a set of quadrants for clarity
inductive Quadrant
| first
| second
| third
| fourth

-- Prove that if the point has specific coordinates, it lies in the specific quadrant
def point_in_quadrant (x: ℝ) (y: ℝ) : Quadrant :=
  if x < 0 ∧ y > 0 then Quadrant.second
  else if x > 0 ∧ y > 0 then Quadrant.first
  else if x < 0 ∧ y < 0 then Quadrant.third
  else Quadrant.fourth

-- The statement to prove:
theorem point_A_in_second_quadrant : point_in_quadrant x_coord y_coord = Quadrant.second :=
by 
  -- Proof would go here but is not required per instructions
  sorry

end point_A_in_second_quadrant_l293_293834


namespace accurate_place_24000_scientific_notation_46400000_l293_293528

namespace MathProof

def accurate_place (n : ℕ) : String :=
  if n = 24000 then "hundred's place" else "unknown"

def scientific_notation (n : ℕ) : String :=
  if n = 46400000 then "4.64 × 10^7" else "unknown"

theorem accurate_place_24000 : accurate_place 24000 = "hundred's place" :=
by
  sorry

theorem scientific_notation_46400000 : scientific_notation 46400000 = "4.64 × 10^7" :=
by
  sorry

end MathProof

end accurate_place_24000_scientific_notation_46400000_l293_293528


namespace buses_required_l293_293182

theorem buses_required (students : ℕ) (bus_capacity : ℕ) (h_students : students = 325) (h_bus_capacity : bus_capacity = 45) : 
∃ n : ℕ, n = 8 ∧ bus_capacity * n ≥ students :=
by
  sorry

end buses_required_l293_293182


namespace inscribed_polygon_regular_if_odd_sides_l293_293125

-- Given a polygon with an odd number of sides which is inscribed in a circle,
-- and all its sides are equal, prove that the polygon is regular.
theorem inscribed_polygon_regular_if_odd_sides {n : ℕ} (h_odd : n % 2 = 1) (h_n_ge_3 : 3 ≤ n) 
  (circumcircle : Type) (P : ℕ → circumcircle)
  (is_inscribed : ∀ i j, i ≠ j → P i ≠ P j)
  (equal_sides : ∀ i j, ∃ k l, i ≠ j → P i = P k → P j = P l -> (k ≤ n ∧ l ≤ n))
  : ∀ i j, (i ≠ j → dist (P i) (P j) = dist (P (i+1)) (P (j+1))) := 
sorry

end inscribed_polygon_regular_if_odd_sides_l293_293125


namespace cos_180_degree_l293_293606

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l293_293606


namespace negation_of_universal_statement_l293_293261

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ^ 2 ≠ x) ↔ ∃ x : ℝ, x ^ 2 = x :=
by
  sorry

end negation_of_universal_statement_l293_293261


namespace simplify_expr_l293_293710

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l293_293710


namespace base8_to_base10_correct_l293_293438

def base8_to_base10_conversion : Prop :=
  (2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166)

theorem base8_to_base10_correct : base8_to_base10_conversion :=
by
  sorry

end base8_to_base10_correct_l293_293438


namespace number_of_rabbits_is_38_l293_293444

-- Conditions: 
def ducks : ℕ := 52
def chickens : ℕ := 78
def condition (ducks rabbits chickens : ℕ) : Prop := 
  chickens = ducks + rabbits - 12

-- Statement: Prove that the number of rabbits is 38
theorem number_of_rabbits_is_38 : ∃ R : ℕ, condition ducks R chickens ∧ R = 38 := by
  sorry

end number_of_rabbits_is_38_l293_293444


namespace find_angle_C_l293_293341

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end find_angle_C_l293_293341


namespace total_fires_l293_293930

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end total_fires_l293_293930


namespace contractor_total_amount_l293_293567

-- Define the conditions
def days_engaged := 30
def pay_per_day := 25
def fine_per_absent_day := 7.50
def days_absent := 10
def days_worked := days_engaged - days_absent

-- Define the earnings and fines
def total_earnings := days_worked * pay_per_day
def total_fine := days_absent * fine_per_absent_day

-- Prove the total amount the contractor gets
theorem contractor_total_amount : total_earnings - total_fine = 425 := by
  sorry

end contractor_total_amount_l293_293567


namespace simplify_fraction_l293_293520

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : (2 + 4 * i) / (1 - 5 * i) = (-9 / 13) + (7 / 13) * i :=
by sorry

end simplify_fraction_l293_293520


namespace red_marked_area_on_larger_sphere_l293_293921

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end red_marked_area_on_larger_sphere_l293_293921


namespace regions_divided_by_chords_l293_293344

open Function

theorem regions_divided_by_chords (P : Finset ℤ) (hP : P.card = 20) 
  (h_non_concurrent : ∀ {a b c d : Finset ℤ}, {a, b, c, d} ⊆ P → ({a, b}, {c, d} ∈ P ∧ {a, c}, {b, d} ∈ P)
    → IsLinearIndep ℝ ![(a : ℝ), (b : ℝ)] ([(c : ℝ), (d : ℝ)])) :
  let V := 20 + (Finset.card (Finset.powersetLen 4 P))
      E := (20 * 21 + 4 * (Finset.card (Finset.powersetLen 4 P))) / 2
  in (E - V + 2 - 1) = 5036 :=
by
  sorry

end regions_divided_by_chords_l293_293344


namespace exists_colored_subset_l293_293844

theorem exists_colored_subset (n : ℕ) (h_positive : n > 0) (colors : ℕ → ℕ) (h_colors : ∀ a b : ℕ, a < b → a + b ≤ n → 
  (colors a = colors b ∨ colors b = colors (a + b) ∨ colors a = colors (a + b))) :
  ∃ c, ∃ s : Finset ℕ, s.card ≥ (2 * n / 5) ∧ ∀ x ∈ s, colors x = c :=
sorry

end exists_colored_subset_l293_293844


namespace perfect_square_trinomial_l293_293269

theorem perfect_square_trinomial (x : ℝ) : 
  let a := x
  let b := 1 / 2
  2 * a * b = x :=
by
  sorry

end perfect_square_trinomial_l293_293269


namespace max_length_of_u_l293_293024

variables (v w : ℝ^3)

def initial_vectors : list (ℝ^3) :=
  [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

def move (v w : ℝ^3) : ℝ^3 × ℝ^3 :=
  ((1 / real.sqrt 2) • (v + w), (1 / real.sqrt 2) • (v - w))

theorem max_length_of_u :
  ∃ u : ℝ^3, (initial_vectors.v_sum = u) → ∥u∥ ≤ 2 * real.sqrt 3 :=
sorry

end max_length_of_u_l293_293024


namespace cos_180_eq_neg1_l293_293590

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l293_293590


namespace age_difference_l293_293035

-- Definitions based on the problem statement
def son_present_age : ℕ := 33

-- Represent the problem in terms of Lean
theorem age_difference (M : ℕ) (h : M + 2 = 2 * (son_present_age + 2)) : M - son_present_age = 35 :=
by
  sorry

end age_difference_l293_293035


namespace regular_decagon_triangle_probability_l293_293807

theorem regular_decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3
  let favorable_triangles := 10
  let probability := favorable_triangles / total_triangles
  probability = (1 : ℚ) / 12 :=
by
  sorry

end regular_decagon_triangle_probability_l293_293807


namespace scientific_notation_correct_l293_293225

theorem scientific_notation_correct :
  ∃! (n : ℝ) (a : ℝ), 0.000000012 = a * 10 ^ n ∧ a = 1.2 ∧ n = -8 :=
by
  sorry

end scientific_notation_correct_l293_293225


namespace find_x_value_l293_293059

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l293_293059


namespace value_of_b_minus_a_l293_293390

theorem value_of_b_minus_a (a b : ℕ) (h1 : a * b = 2 * (a + b) + 1) (h2 : b = 7) : b - a = 4 :=
by
  sorry

end value_of_b_minus_a_l293_293390


namespace count_distinct_four_digit_numbers_ending_in_25_l293_293470

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end count_distinct_four_digit_numbers_ending_in_25_l293_293470


namespace quadrilateral_trapezoid_or_parallelogram_l293_293287

theorem quadrilateral_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ)
  (hs : s1^2 = s2 * s4) :
  (exists (is_trapezoid : Prop), is_trapezoid) ∨ (exists (is_parallelogram : Prop), is_parallelogram) :=
by
  sorry

end quadrilateral_trapezoid_or_parallelogram_l293_293287


namespace eccentricity_range_l293_293979

open Real

variable (a b x_o c e : ℝ)
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : x_o > a)
variable (h4 : c = sqrt (a^2 + b^2))
variable (h5 : ∀ P : ℝ × ℝ, 
  ∃ P : ℝ × ℝ, 
    P.1^2 / a^2 - P.2^2 / b^2 = 1 ∧
    (sin (arcsin ((P.1 + c) / sqrt ((P.1 - c)^2 + P.2^2)) / arcsin ((P.1 - c) / sqrt ((P.1 + c)^2 + P.2^2))) = a / c))

theorem eccentricity_range : 1 < e ∧ e < sqrt 2 + 1 :=
sorry

end eccentricity_range_l293_293979


namespace weight_ratio_l293_293740

noncomputable def weight_ratio_proof : Prop :=
  ∃ (R S : ℝ), 
  (R + S = 72) ∧ 
  (1.10 * R + 1.17 * S = 82.8) ∧ 
  (R / S = 1 / 2.5)

theorem weight_ratio : weight_ratio_proof := 
  by
    sorry

end weight_ratio_l293_293740


namespace helen_cookies_till_last_night_l293_293822

theorem helen_cookies_till_last_night 
  (cookies_yesterday : Nat := 31) 
  (cookies_day_before_yesterday : Nat := 419) : 
  cookies_yesterday + cookies_day_before_yesterday = 450 := 
by
  sorry

end helen_cookies_till_last_night_l293_293822


namespace cylinder_height_relation_l293_293152

variables (r1 h1 r2 h2 : ℝ)
variables (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2) (r2_eq_1_2_r1 : r2 = 1.2 * r1)

theorem cylinder_height_relation : h1 = 1.44 * h2 :=
by
  sorry

end cylinder_height_relation_l293_293152


namespace gcd_lcm_product_l293_293309

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l293_293309


namespace count_distinct_four_digit_numbers_ending_in_25_l293_293471

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end count_distinct_four_digit_numbers_ending_in_25_l293_293471


namespace willam_tax_payment_correct_l293_293800

noncomputable def willamFarmTax : ℝ :=
  let totalTax := 3840
  let willamPercentage := 0.2777777777777778
  totalTax * willamPercentage

-- Lean theorem statement for the problem
theorem willam_tax_payment_correct : 
  willamFarmTax = 1066.67 :=
by
  sorry

end willam_tax_payment_correct_l293_293800


namespace problem1_problem2_l293_293695

variables (x a : ℝ)

-- Proposition definitions
def proposition_p (a : ℝ) (x : ℝ) : Prop :=
  a > 0 ∧ (-x^2 + 4*a*x - 3*a^2) > 0

def proposition_q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) < 0

-- Problems
theorem problem1 : (proposition_p 1 x ∧ proposition_q x) ↔ 2 < x ∧ x < 3 :=
by sorry

theorem problem2 : (¬ ∃ x, proposition_p a x) → (∀ x, ¬ proposition_q x) →
  1 ≤ a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l293_293695


namespace shift_sine_left_by_pi_over_2_l293_293250

theorem shift_sine_left_by_pi_over_2 :
  let f : ℝ → ℝ := λ x, Real.cos x in
  ∀ x : ℝ, f x = Real.sin (x + Real.pi / 2) ∧
           (∀ x, f x = Real.cos x) ∧
           (∀ x, Real.cos x = 0 → (x = Real.pi / 2 ∨ x = -Real.pi / 2)) →
           ∃ c : ℝ, c = -Real.pi / 2 ∧ 0 = f c :=
by
  sorry

end shift_sine_left_by_pi_over_2_l293_293250


namespace find_a_values_l293_293384

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l293_293384


namespace cos_180_eq_neg1_l293_293634

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l293_293634


namespace part1_intersection_part2_sufficient_not_necessary_l293_293646

open Set

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def set_B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

-- Part (1)
theorem part1_intersection (a : ℝ) (h : a = -2) : set_A a ∩ set_B = {x | -3 ≤ x ∧ x ≤ -2} := by
  sorry

-- Part (2)
theorem part2_sufficient_not_necessary (p q : Prop) (hp : ∀ x, set_A a x → set_B x) (h_suff : p → q) (h_not_necess : ¬(q → p)) : set_A a ⊆ set_B → a ∈ Iic (-3) ∪ Ici 4 := by
  sorry

end part1_intersection_part2_sufficient_not_necessary_l293_293646


namespace find_n_cosine_l293_293446

theorem find_n_cosine (n : ℤ) (h1 : 100 ≤ n ∧ n ≤ 300) (h2 : Real.cos (n : ℝ) = Real.cos 140) : n = 220 :=
by
  sorry

end find_n_cosine_l293_293446


namespace min_female_students_l293_293032

theorem min_female_students (males females : ℕ) (total : ℕ) (percent_participated : ℕ) (participated : ℕ) (min_females : ℕ)
  (h1 : males = 22) 
  (h2 : females = 18) 
  (h3 : total = males + females)
  (h4 : percent_participated = 60) 
  (h5 : participated = (percent_participated * total) / 100)
  (h6 : min_females = participated - males) :
  min_females = 2 := 
sorry

end min_female_students_l293_293032


namespace balance_blue_balls_l293_293692

variables (G B Y W : ℝ)

-- Definitions based on conditions
def condition1 : Prop := 3 * G = 6 * B
def condition2 : Prop := 2 * Y = 5 * B
def condition3 : Prop := 6 * B = 4 * W

-- Statement of the problem
theorem balance_blue_balls (h1 : condition1 G B) (h2 : condition2 Y B) (h3 : condition3 B W) :
  4 * G + 2 * Y + 2 * W = 16 * B :=
sorry

end balance_blue_balls_l293_293692


namespace function_range_l293_293539

def function_defined (x : ℝ) : Prop := x ≠ 5

theorem function_range (x : ℝ) : x ≠ 5 → function_defined x :=
by
  intro h
  exact h

end function_range_l293_293539


namespace digits_subtraction_eq_zero_l293_293865

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end digits_subtraction_eq_zero_l293_293865


namespace correct_options_l293_293465

-- Definitions for lines l and n
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def line_n (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- The condition for lines to be parallel, equating the slopes
def parallel_lines (a : ℝ) : Prop := -(a + 2) / a = -(a - 2) / 3

-- The condition that line l passes through the point (1, -1)
def passes_through_point (a : ℝ) : Prop := line_l a 1 (-1)

-- The theorem statement
theorem correct_options (a : ℝ) :
  (parallel_lines a → a = 6 ∨ a = -1) ∧ (passes_through_point a) :=
by
  sorry

end correct_options_l293_293465


namespace house_selling_price_l293_293245

theorem house_selling_price
  (original_price : ℝ := 80000)
  (profit_rate : ℝ := 0.20)
  (commission_rate : ℝ := 0.05):
  original_price + (original_price * profit_rate) + (original_price * commission_rate) = 100000 := by
  sorry

end house_selling_price_l293_293245


namespace arithmetic_sequence_problem_l293_293676

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific terms in arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Conditions given in the problem
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The proof goal
theorem arithmetic_sequence_problem : a 9 - 1/3 * a 11 = 16 :=
by
  sorry

end arithmetic_sequence_problem_l293_293676


namespace palindromic_example_exists_count_palindromic_divisible_by_5_l293_293238

def is_palindromic (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem palindromic_example_exists : ∃ n : ℕ, n = 51715 ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by
  sorry

theorem count_palindromic_divisible_by_5 : ∃ (count : ℕ), count = 100 ∧
  (count = finset.card (finset.filter (λ n, is_palindromic n ∧ is_divisible_by_5 n)
                                       (finset.Icc 10000 99999))) :=
by
  sorry

end palindromic_example_exists_count_palindromic_divisible_by_5_l293_293238


namespace count_complex_numbers_l293_293826

theorem count_complex_numbers (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + b ≤ 5) : 
  ∃ n : ℕ, n = 10 :=
by
  sorry

end count_complex_numbers_l293_293826


namespace derivative_of_y_l293_293067

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (log 7) * (sin (7 * x)) ^ 2) / (7 * cos (14 * x))

theorem derivative_of_y (x : ℝ) : deriv y x = (cos (log 7) * tan (14 * x)) / cos (14 * x) := sorry

end derivative_of_y_l293_293067


namespace domain_f_l293_293441

def domain_of_f (x : ℝ) : Prop :=
  (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x < 4)

theorem domain_f :
  ∀ x, domain_of_f x ↔ (x ≥ 2 ∧ x < 4) ∧ x ≠ 3 :=
by
  sorry

end domain_f_l293_293441


namespace solve_for_x_l293_293158

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (h_eq : (8 * x) ^ 16 = (32 * x) ^ 8) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l293_293158


namespace transport_cost_l293_293720

theorem transport_cost (cost_per_kg : ℝ) (weight_g : ℝ) : 
  (cost_per_kg = 30000) → (weight_g = 400) → 
  ((weight_g / 1000) * cost_per_kg = 12000) :=
by
  intros h1 h2
  sorry

end transport_cost_l293_293720


namespace first_quarter_spending_l293_293727

variables (spent_february_start spent_march_end spent_april_end : ℝ)

-- Given conditions
def begin_february_spent : Prop := spent_february_start = 0.5
def end_march_spent : Prop := spent_march_end = 1.5
def end_april_spent : Prop := spent_april_end = 2.0

-- Proof statement
theorem first_quarter_spending (h1 : begin_february_spent spent_february_start) 
                               (h2 : end_march_spent spent_march_end) 
                               (h3 : end_april_spent spent_april_end) : 
                                spent_march_end - spent_february_start = 1.5 :=
by sorry

end first_quarter_spending_l293_293727


namespace exists_constants_c1_c2_l293_293553

def S (m : ℕ) : ℕ := m.digits.sum

def f (n : ℕ) : ℕ := sorry -- Define f(n) as necessary depending on the problem specifics

theorem exists_constants_c1_c2 (n : ℕ) (h : n ≥ 2) :
  ∃ (c1 c2 : ℝ), 0 < c1 ∧ c1 < c2 ∧ (c1 * real.log10 n) < f(n) ∧ f(n) < (c2 * real.log10 n) :=
sorry

end exists_constants_c1_c2_l293_293553


namespace total_amount_received_l293_293160

theorem total_amount_received (B : ℝ) (h1 : (1/3) * B = 36) : (2/3 * B) * 4 = 288 :=
by
  sorry

end total_amount_received_l293_293160


namespace interval_of_n_l293_293806

noncomputable def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem interval_of_n (n : ℕ) (hn : 0 < n ∧ n < 2000)
  (h1 : divides n 9999)
  (h2 : divides (n + 4) 999999) :
  801 ≤ n ∧ n ≤ 1200 :=
sorry

end interval_of_n_l293_293806


namespace fertilizer_needed_l293_293862

def p_flats := 4
def p_per_flat := 8
def p_ounces := 8

def r_flats := 3
def r_per_flat := 6
def r_ounces := 3

def s_flats := 5
def s_per_flat := 10
def s_ounces := 6

def o_flats := 2
def o_per_flat := 4
def o_ounces := 4

def vf_quantity := 2
def vf_ounces := 2

def total_fertilizer : ℕ := 
  p_flats * p_per_flat * p_ounces +
  r_flats * r_per_flat * r_ounces +
  s_flats * s_per_flat * s_ounces +
  o_flats * o_per_flat * o_ounces +
  vf_quantity * vf_ounces

theorem fertilizer_needed : total_fertilizer = 646 := by
  -- proof goes here
  sorry

end fertilizer_needed_l293_293862


namespace sum_m_n_for_jar_candies_l293_293175

theorem sum_m_n_for_jar_candies :
  let total_prob_same_comb : ℚ :=
    (55 / 1615) + (70 / 4845) + (3696 / 14535)
  let simplified_prob := (256 / 909)
  (total_prob_same_comb = simplified_prob) ∧ (256 + 909 = 1165) :=
by
  sorry

end sum_m_n_for_jar_candies_l293_293175


namespace avg_height_eq_61_l293_293228

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61_l293_293228


namespace max_tickets_with_120_l293_293637

-- Define the cost of tickets
def cost_ticket (n : ℕ) : ℕ :=
  if n ≤ 5 then n * 15
  else 5 * 15 + (n - 5) * 12

-- Define the maximum number of tickets Jane can buy with 120 dollars
def max_tickets (money : ℕ) : ℕ :=
  if money ≤ 75 then money / 15
  else 5 + (money - 75) / 12

-- Prove that with 120 dollars, the maximum number of tickets Jane can buy is 8
theorem max_tickets_with_120 : max_tickets 120 = 8 :=
by
  sorry

end max_tickets_with_120_l293_293637


namespace trains_meet_at_10_am_l293_293153

def distance (speed time : ℝ) : ℝ := speed * time

theorem trains_meet_at_10_am
  (distance_pq : ℝ)
  (speed_train_from_p : ℝ)
  (start_time_from_p : ℝ)
  (speed_train_from_q : ℝ)
  (start_time_from_q : ℝ)
  (meeting_time : ℝ) :
  distance_pq = 110 → 
  speed_train_from_p = 20 → 
  start_time_from_p = 7 → 
  speed_train_from_q = 25 → 
  start_time_from_q = 8 → 
  meeting_time = 10 :=
by
  sorry

end trains_meet_at_10_am_l293_293153


namespace solve_for_b_l293_293212

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 :=
by sorry

end solve_for_b_l293_293212


namespace problem1_problem2_l293_293661

-- Problem 1: Prove f(x) ≥ 3 implies x ≤ -1 or x ≥ 1 given f(x) = |x + 1| + |2x - 1| and m = 1
theorem problem1 (x : ℝ) : (|x + 1| + |2 * x - 1| >= 3) ↔ (x <= -1 ∨ x >= 1) :=
by
 sorry

-- Problem 2: Prove ½ f(x) ≤ |x + 1| holds for x ∈ [m, 2m²] implies ½ < m ≤ 1 given f(x) = |x + m| + |2x - 1| and m > 0
theorem problem2 (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : m ≤ x ∧ x ≤ 2 * m^2) : (1/2 * (|x + m| + |2 * x - 1|) ≤ |x + 1|) ↔ (1/2 < m ∧ m ≤ 1) :=
by
 sorry

end problem1_problem2_l293_293661


namespace bracelet_pairing_impossible_l293_293797

/--
Elizabeth has 100 different bracelets, and each day she wears three of them to school. 
Prove that it is impossible for any pair of bracelets to appear together on her wrist exactly once.
-/
theorem bracelet_pairing_impossible : 
  (∃ (bracelet_set : Finset (Finset (Fin 100))), 
    (∀ (a b : Fin 100), a ≠ b → ∃ t ∈ bracelet_set, {a, b} ⊆ t) ∧ (∀ t ∈ bracelet_set, t.card = 3) ∧ (bracelet_set.card * 3 / 2 ≠ 99)) :=
sorry

end bracelet_pairing_impossible_l293_293797


namespace sequence_formula_l293_293078

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) + 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^n - 2 :=
by 
sorry

end sequence_formula_l293_293078


namespace quotient_of_division_l293_293271

theorem quotient_of_division:
  ∀ (n d r q : ℕ), n = 165 → d = 18 → r = 3 → q = (n - r) / d → q = 9 :=
by sorry

end quotient_of_division_l293_293271


namespace minimum_possible_length_of_third_side_l293_293987

theorem minimum_possible_length_of_third_side (a b : ℝ) (h : a = 8 ∧ b = 15 ∨ a = 15 ∧ b = 8) : 
  ∃ c : ℝ, (c * c = a * a + b * b ∨ c * c = a * a - b * b ∨ c * c = b * b - a * a) ∧ c = Real.sqrt 161 :=
by
  sorry

end minimum_possible_length_of_third_side_l293_293987


namespace common_ratio_of_geometric_sequence_l293_293202

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 2) 
  (h2 : a 5 = 1 / 4) : 
  ( ∃ a1 : ℝ, a n = a1 * q ^ (n - 1)) 
    :=
sorry

end common_ratio_of_geometric_sequence_l293_293202


namespace cos_180_eq_neg1_l293_293604

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l293_293604


namespace s_plus_t_l293_293116

def g (x : ℝ) : ℝ := 3 * x ^ 4 + 9 * x ^ 3 - 7 * x ^ 2 + 2 * x + 4
def h (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

noncomputable def s (x : ℝ) : ℝ := 3 * x ^ 2 + 3
noncomputable def t (x : ℝ) : ℝ := 3 * x + 6

theorem s_plus_t : s 1 + t (-1) = 9 := by
  sorry

end s_plus_t_l293_293116


namespace apples_in_each_basket_l293_293690

theorem apples_in_each_basket (total_apples : ℕ) (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : baskets = 19) 
  (h3 : apples_per_basket = total_apples / baskets) : 
  apples_per_basket = 26 :=
by 
  rw [h1, h2] at h3
  exact h3

end apples_in_each_basket_l293_293690


namespace contractor_total_amount_l293_293568

-- Define the conditions
def days_engaged := 30
def pay_per_day := 25
def fine_per_absent_day := 7.50
def days_absent := 10
def days_worked := days_engaged - days_absent

-- Define the earnings and fines
def total_earnings := days_worked * pay_per_day
def total_fine := days_absent * fine_per_absent_day

-- Prove the total amount the contractor gets
theorem contractor_total_amount : total_earnings - total_fine = 425 := by
  sorry

end contractor_total_amount_l293_293568


namespace range_of_a_l293_293218

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 2 else Real.log (-x) / Real.log (1/2)

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) :=
by
  sorry

end range_of_a_l293_293218


namespace find_louis_age_l293_293100

variables (C L : ℕ)

-- Conditions:
-- 1. In some years, Carla will be 30 years old
-- 2. The sum of the current ages of Carla and Louis is 55

theorem find_louis_age (h1 : ∃ n, C + n = 30) (h2 : C + L = 55) : L = 25 :=
by {
  sorry
}

end find_louis_age_l293_293100


namespace sqrt_sum_simplification_l293_293306

theorem sqrt_sum_simplification : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) = 10 := by
  sorry

end sqrt_sum_simplification_l293_293306


namespace min_value_of_x_squared_plus_6x_l293_293889

theorem min_value_of_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  sorry
end

end min_value_of_x_squared_plus_6x_l293_293889


namespace mutually_exclusive_but_not_complementary_l293_293305

open Classical

namespace CardDistribution

inductive Card
| red | yellow | blue | white

inductive Person
| A | B | C | D

def Event_A_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.A = Card.red

def Event_D_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.D = Card.red

theorem mutually_exclusive_but_not_complementary :
  ∀ (distrib: Person → Card),
  (Event_A_gets_red distrib → ¬Event_D_gets_red distrib) ∧
  ¬(∀ (distrib: Person → Card), Event_A_gets_red distrib ∨ Event_D_gets_red distrib) := 
by
  sorry

end CardDistribution

end mutually_exclusive_but_not_complementary_l293_293305


namespace indeterminate_original_value_percentage_l293_293911

-- Lets define the problem as a structure with the given conditions
structure StockData where
  yield_percent : ℚ
  market_value : ℚ

-- We need to prove this condition
theorem indeterminate_original_value_percentage (d : StockData) :
  d.yield_percent = 8 ∧ d.market_value = 125 → false :=
by
  sorry

end indeterminate_original_value_percentage_l293_293911


namespace rectangle_ABCD_area_l293_293860

def rectangle_area (x : ℕ) : ℕ :=
  let side_lengths := [x, x+1, x+2, x+3];
  let width := side_lengths.sum;
  let height := width - x;
  width * height

theorem rectangle_ABCD_area : rectangle_area 1 = 143 :=
by
  sorry

end rectangle_ABCD_area_l293_293860


namespace purely_imaginary_sol_l293_293217

theorem purely_imaginary_sol {m : ℝ} (h : (m^2 - 3 * m) = 0) (h2 : (m^2 - 5 * m + 6) ≠ 0) : m = 0 :=
sorry

end purely_imaginary_sol_l293_293217


namespace square_value_l293_293827

theorem square_value {square : ℚ} (h : 8 / 12 = square / 3) : square = 2 :=
sorry

end square_value_l293_293827


namespace average_is_20_l293_293529

-- Define the numbers and the variable n
def a := 3
def b := 16
def c := 33
def n := 27
def d := n + 1

-- Define the sum of the numbers
def sum := a + b + c + d

-- Define the average as sum divided by 4
def average := sum / 4

-- Prove that the average is 20
theorem average_is_20 : average = 20 := by
  sorry

end average_is_20_l293_293529


namespace simplify_fraction_l293_293700

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l293_293700


namespace initial_weasels_count_l293_293099

theorem initial_weasels_count (initial_rabbits : ℕ) (foxes : ℕ) (weasels_per_fox : ℕ) (rabbits_per_fox : ℕ) 
                              (weeks : ℕ) (remaining_rabbits_weasels : ℕ) (initial_weasels : ℕ) 
                              (total_rabbits_weasels : ℕ) : 
    initial_rabbits = 50 → foxes = 3 → weasels_per_fox = 4 → rabbits_per_fox = 2 → weeks = 3 → 
    remaining_rabbits_weasels = 96 → total_rabbits_weasels = initial_rabbits + initial_weasels → initial_weasels = 100 :=
by
  sorry

end initial_weasels_count_l293_293099


namespace minimum_value_fraction_l293_293461

theorem minimum_value_fraction (m n : ℝ) (h0 : 0 ≤ m) (h1 : 0 ≤ n) (h2 : m + n = 1) :
  ∃ min_val, min_val = (1 / 4) ∧ (∀ m n, 0 ≤ m → 0 ≤ n → m + n = 1 → (m^2) / (m + 2) + (n^2) / (n + 1) ≥ min_val) :=
sorry

end minimum_value_fraction_l293_293461


namespace combined_rate_of_three_cars_l293_293126

theorem combined_rate_of_three_cars
  (m : ℕ)
  (ray_avg : ℕ)
  (tom_avg : ℕ)
  (alice_avg : ℕ)
  (h1 : ray_avg = 30)
  (h2 : tom_avg = 15)
  (h3 : alice_avg = 20) :
  let total_distance := 3 * m
  let total_gasoline := m / ray_avg + m / tom_avg + m / alice_avg
  (total_distance / total_gasoline) = 20 := 
by
  sorry

end combined_rate_of_three_cars_l293_293126


namespace hyperbola_asymptotes_l293_293464

theorem hyperbola_asymptotes 
  (a b : ℝ)
  (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (F₁ F₂ P : ℝ × ℝ)
  (line_perpendicular : ∃ c : ℝ, (F₂.1 = c))
  (angle_condition : ∃ (θ : ℝ), θ = 𝜋 / 6 ∧ angle P F₁ F₂ = θ) :
  asymptotes ? := sorry -- to be defined

end hyperbola_asymptotes_l293_293464


namespace interval_of_expression_l293_293440

theorem interval_of_expression (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧ 
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
by sorry

end interval_of_expression_l293_293440


namespace product_of_number_and_sum_of_digits_l293_293489

-- Definitions according to the conditions
def units_digit (a b : ℕ) : Prop := b = a + 2
def number_equals_24 (a b : ℕ) : Prop := 10 * a + b = 24

-- The main statement to prove the product of the number and the sum of its digits
theorem product_of_number_and_sum_of_digits :
  ∃ (a b : ℕ), units_digit a b ∧ number_equals_24 a b ∧ (24 * (a + b) = 144) :=
sorry

end product_of_number_and_sum_of_digits_l293_293489


namespace cos_180_eq_minus_1_l293_293631

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l293_293631


namespace original_price_of_shoes_l293_293348

-- Define the conditions.
def discount_rate : ℝ := 0.20
def amount_paid : ℝ := 480

-- Statement of the theorem.
theorem original_price_of_shoes (P : ℝ) (h₀ : P * (1 - discount_rate) = amount_paid) : 
  P = 600 :=
by
  sorry

end original_price_of_shoes_l293_293348


namespace range_of_function_l293_293392

theorem range_of_function :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 2 ≤ x^2 - 2 * x + 3 ∧ x^2 - 2 * x + 3 ≤ 6) :=
by {
  sorry
}

end range_of_function_l293_293392


namespace min_box_height_l293_293947

noncomputable def height_of_box (x : ℝ) := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem min_box_height (x h : ℝ) (h₁ : h = height_of_box x) (h₂ : surface_area x ≥ 130) : h ≥ 25 / 3 :=
by sorry

end min_box_height_l293_293947


namespace find_ratio_l293_293968

open Real

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)

-- The geometric sequence conditions
def geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

-- Sum of the first n terms for the geometric sequence
def sum_of_first_n_terms := ∀ n : ℕ, S n = (a 0) * (1 - q ^ n) / (1 - q)

-- Given conditions
def given_conditions :=
  a 0 + a 2 = 5 / 2 ∧
  a 1 + a 3 = 5 / 4

-- The goal to prove
theorem find_ratio (geo_seq : geometric_sequence a q) (sum_terms : sum_of_first_n_terms a S q) (cond : given_conditions a) :
  S 4 / a 4 = 31 :=
  sorry

end find_ratio_l293_293968


namespace proof_problem_l293_293199

theorem proof_problem (p q : Prop) : (p ∧ q) ↔ ¬ (¬ p ∨ ¬ q) :=
sorry

end proof_problem_l293_293199


namespace Irene_hours_worked_l293_293492

open Nat

theorem Irene_hours_worked (x totalHours : ℕ) : 
  (500 + 20 * x = 700) → 
  (totalHours = 40 + x) → 
  totalHours = 50 :=
by
  sorry

end Irene_hours_worked_l293_293492


namespace b_plus_d_l293_293495

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem b_plus_d 
  (a b c d : ℝ) 
  (h1 : f a b c d 1 = 20) 
  (h2 : f a b c d (-1) = 16) 
: b + d = 18 :=
sorry

end b_plus_d_l293_293495


namespace transformation_correct_l293_293272

theorem transformation_correct (a b c : ℝ) : a = b → ac = bc :=
by sorry

end transformation_correct_l293_293272


namespace second_train_length_is_correct_l293_293019

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (time_crossing_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train_mps := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_crossing_seconds
  total_distance - length_first_train

theorem second_train_length_is_correct : length_of_second_train 360 120 80 9 = 139.95 :=
by
  sorry

end second_train_length_is_correct_l293_293019


namespace average_age_l293_293868
open Nat

def age_to_months (years : ℕ) (months : ℕ) : ℕ := years * 12 + months

theorem average_age :
  let age1 := age_to_months 14 9
  let age2 := age_to_months 15 1
  let age3 := age_to_months 14 8
  let total_months := age1 + age2 + age3
  let avg_months := total_months / 3
  let avg_years := avg_months / 12
  let avg_remaining_months := avg_months % 12
  avg_years = 14 ∧ avg_remaining_months = 10 := by
  sorry

end average_age_l293_293868


namespace AC_total_l293_293923

theorem AC_total (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 450) (h3 : C = 100) : A + C = 250 := by
  sorry

end AC_total_l293_293923


namespace tan_theta_minus_pi_over4_l293_293825

theorem tan_theta_minus_pi_over4 (θ : Real) (h : Real.cos θ - 3 * Real.sin θ = 0) : 
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over4_l293_293825


namespace avg_height_eq_61_l293_293229

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61_l293_293229


namespace max_count_larger_than_20_l293_293738

noncomputable def max_larger_than_20 (int_list : List Int) : Nat :=
  (int_list.filter (λ n => n > 20)).length

theorem max_count_larger_than_20 (a1 a2 a3 a4 a5 a6 a7 a8 : Int)
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 10) :
  ∃ (k : Nat), k = 7 ∧ max_larger_than_20 [a1, a2, a3, a4, a5, a6, a7, a8] = k :=
sorry

end max_count_larger_than_20_l293_293738


namespace abs_triangle_inequality_l293_293986

theorem abs_triangle_inequality {a : ℝ} (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 :=
sorry

end abs_triangle_inequality_l293_293986


namespace gcd_lcm_product_l293_293311

theorem gcd_lcm_product (a b : ℕ) (h_a : a = 24) (h_b : b = 60) : 
  Nat.gcd a b * Nat.lcm a b = 1440 := by 
  rw [h_a, h_b]
  apply Nat.gcd_mul_lcm
  sorry

end gcd_lcm_product_l293_293311


namespace correct_transformation_l293_293752

-- Definitions of the points and their mapped coordinates
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -3)
def C' : ℝ × ℝ := (1, 2)
def D' : ℝ × ℝ := (-2, 3)

-- Transformation function (as given in the problem)
def skew_reflection_and_vertical_shrink (p : ℝ × ℝ) : ℝ × ℝ :=
  match p with
  | (x, y) => (-y, x)

-- Theorem statement to be proved
theorem correct_transformation :
  skew_reflection_and_vertical_shrink C = C' ∧ skew_reflection_and_vertical_shrink D = D' :=
sorry

end correct_transformation_l293_293752


namespace ratio_of_costs_l293_293191

theorem ratio_of_costs (R N : ℝ) (hR : 3 * R = 0.25 * (3 * R + 3 * N)) : N / R = 3 := 
sorry

end ratio_of_costs_l293_293191


namespace cyclist_average_speed_l293_293569

noncomputable def total_distance : ℝ := 10 + 5 + 15 + 20 + 30
noncomputable def time_first_segment : ℝ := 10 / 12
noncomputable def time_second_segment : ℝ := 5 / 6
noncomputable def time_third_segment : ℝ := 15 / 16
noncomputable def time_fourth_segment : ℝ := 20 / 14
noncomputable def time_fifth_segment : ℝ := 30 / 20

noncomputable def total_time : ℝ := time_first_segment + time_second_segment + time_third_segment + time_fourth_segment + time_fifth_segment

noncomputable def average_speed : ℝ := total_distance / total_time

theorem cyclist_average_speed : average_speed = 12.93 := by
  sorry

end cyclist_average_speed_l293_293569


namespace cos_180_eq_neg_one_l293_293596

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l293_293596


namespace correct_operation_result_l293_293758

variable (x : ℕ)

theorem correct_operation_result 
  (h : x / 15 = 6) : 15 * x = 1350 :=
sorry

end correct_operation_result_l293_293758


namespace fred_spending_correct_l293_293643

noncomputable def fred_total_spending : ℝ :=
  let football_price_each := 2.73
  let football_quantity := 2
  let football_tax_rate := 0.05
  let pokemon_price := 4.01
  let pokemon_tax_rate := 0.08
  let baseball_original_price := 10
  let baseball_discount_rate := 0.10
  let baseball_tax_rate := 0.06
  let football_total_before_tax := football_price_each * football_quantity
  let football_total_tax := football_total_before_tax * football_tax_rate
  let football_total := football_total_before_tax + football_total_tax
  let pokemon_total_tax := pokemon_price * pokemon_tax_rate
  let pokemon_total := pokemon_price + pokemon_total_tax
  let baseball_discount := baseball_original_price * baseball_discount_rate
  let baseball_discounted_price := baseball_original_price - baseball_discount
  let baseball_total_tax := baseball_discounted_price * baseball_tax_rate
  let baseball_total := baseball_discounted_price + baseball_total_tax
  football_total + pokemon_total + baseball_total

theorem fred_spending_correct :
  fred_total_spending = 19.6038 := 
  by
    sorry

end fred_spending_correct_l293_293643


namespace alice_wins_rational_game_l293_293077

theorem alice_wins_rational_game (r : ℚ) (h_r_gt_1 : r > 1) :
  ∃ d : ℕ, 1 ≤ d ∧ d ≤ 1010 ∧ r = 1 + (1 / d) ↔
  ∃ k ≤ 2021, ∃ x y : ℝ, (0 < y) ∧ (x = 0) ∧ (y = r^k * (y - x)) ∧ (x = 1) := 
by  sorry

end alice_wins_rational_game_l293_293077


namespace total_gymnasts_l293_293316

theorem total_gymnasts (n : ℕ) : 
  (∃ (t : ℕ) (c : t = 4) (h : n * (n-1) / 2 + 4 * 6 = 595), n = 34) :=
by {
  -- skipping the detailed proof here, just ensuring the problem is stated as a theorem
  sorry
}

end total_gymnasts_l293_293316


namespace production_days_l293_293965

theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 90) / (n + 1) = 52) : n = 19 :=
by
  sorry

end production_days_l293_293965


namespace range_of_a_l293_293206

theorem range_of_a (x : ℝ) (a : ℝ) (hx : 0 < x ∧ x < 4) : |x - 1| < a → a ≥ 3 := sorry

end range_of_a_l293_293206


namespace cos_180_degree_l293_293605

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l293_293605


namespace area_larger_sphere_red_is_83_point_25_l293_293917

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l293_293917


namespace cos_180_eq_neg1_l293_293618

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l293_293618


namespace cos_180_eq_neg_one_l293_293610

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l293_293610


namespace find_unknown_polynomial_l293_293478

theorem find_unknown_polynomial (m : ℤ) : 
  ∃ q : ℤ, (q + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1) → q = 2 * m^2 + 3 * m - 4 :=
by {
  sorry
}

end find_unknown_polynomial_l293_293478


namespace cos_180_eq_neg_one_l293_293593

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l293_293593


namespace solve_for_x_l293_293439

def operation (a b : ℝ) : ℝ := a^2 - 3*a + b

theorem solve_for_x (x : ℝ) : operation x 2 = 6 → (x = -1 ∨ x = 4) :=
by
  sorry

end solve_for_x_l293_293439


namespace even_function_l293_293093

theorem even_function (f : ℝ → ℝ) (not_zero : ∃ x, f x ≠ 0) 
  (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b) : 
  ∀ x : ℝ, f (-x) = f x := 
sorry

end even_function_l293_293093


namespace min_value_4a_plus_b_l293_293988

theorem min_value_4a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : 4*a + b = 9 :=
sorry

end min_value_4a_plus_b_l293_293988


namespace proof_p_and_q_true_l293_293319

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > x

theorem proof_p_and_q_true : p ∧ q :=
by
  -- Assume you have already proven that p and q are true separately
  sorry

end proof_p_and_q_true_l293_293319


namespace ribbon_length_difference_l293_293028

-- Variables representing the dimensions of the box
variables (a b c : ℕ)

-- Conditions specifying the dimensions of the box
def box_dimensions := (a = 22) ∧ (b = 22) ∧ (c = 11)

-- Calculating total ribbon length for Method 1
def ribbon_length_method_1 := 2 * a + 2 * b + 4 * c + 24

-- Calculating total ribbon length for Method 2
def ribbon_length_method_2 := 2 * a + 4 * b + 2 * c + 24

-- The proof statement: difference in ribbon length equals one side of the box
theorem ribbon_length_difference : 
  box_dimensions ∧ 
  ribbon_length_method_2 - ribbon_length_method_1 = a :=
by
  -- The proof is omitted
  sorry

end ribbon_length_difference_l293_293028


namespace chuck_play_area_l293_293300

-- Define the conditions for the problem in Lean
def shed_length1 : ℝ := 3
def shed_length2 : ℝ := 4
def leash_length : ℝ := 4

-- State the theorem we want to prove
theorem chuck_play_area :
  let sector_area1 := (3 / 4) * Real.pi * (leash_length ^ 2)
  let sector_area2 := (1 / 4) * Real.pi * (1 ^ 2)
  sector_area1 + sector_area2 = (49 / 4) * Real.pi := 
by
  -- The proof is omitted for brevity
  sorry

end chuck_play_area_l293_293300


namespace Bethany_total_riding_hours_l293_293935

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l293_293935


namespace stock_yield_percentage_l293_293556

theorem stock_yield_percentage (face_value market_price : ℝ) (annual_dividend_rate : ℝ) 
  (h_face_value : face_value = 100)
  (h_market_price : market_price = 140)
  (h_annual_dividend_rate : annual_dividend_rate = 0.14) :
  (annual_dividend_rate * face_value / market_price) * 100 = 10 :=
by
  -- computation here
  sorry

end stock_yield_percentage_l293_293556


namespace browser_usage_information_is_false_l293_293427

def num_people_using_A : ℕ := 316
def num_people_using_B : ℕ := 478
def num_people_using_both_A_and_B : ℕ := 104
def num_people_only_using_one_browser : ℕ := 567

theorem browser_usage_information_is_false :
  num_people_only_using_one_browser ≠ (num_people_using_A - num_people_using_both_A_and_B) + (num_people_using_B - num_people_using_both_A_and_B) :=
by
  sorry

end browser_usage_information_is_false_l293_293427


namespace simplify_polynomial_subtraction_l293_293864

/--
  Given the polynomials (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) and (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5),
  prove that their difference simplifies to x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3.
-/
theorem simplify_polynomial_subtraction  (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) = x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 :=
sorry

end simplify_polynomial_subtraction_l293_293864


namespace cheesecake_factory_savings_l293_293358

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l293_293358


namespace find_x_l293_293105

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end find_x_l293_293105


namespace number_of_boys_l293_293691

theorem number_of_boys (total_students girls : ℕ) (h1 : total_students = 13) (h2 : girls = 6) :
  total_students - girls = 7 :=
by 
  -- We'll skip the proof as instructed
  sorry

end number_of_boys_l293_293691


namespace inequality_solution_l293_293364

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ (x < 1 ∨ x > 3) ∧ (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :=
by
  sorry

end inequality_solution_l293_293364


namespace box_dimensions_correct_l293_293025

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l293_293025


namespace simplify_fraction_l293_293706

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l293_293706


namespace probability_greater_than_4_l293_293901

theorem probability_greater_than_4 :
  let total_faces := 6
  let successful_faces := 2
  (successful_faces : ℚ) / total_faces = 1 / 3 :=
by
  sorry

end probability_greater_than_4_l293_293901


namespace bethany_total_hours_l293_293936

-- Define the hours Bethany rode on each set of days
def hours_mon_wed_fri : ℕ := 3  -- 1 hour each on Monday, Wednesday, and Friday
def hours_tue_thu : ℕ := 1  -- 30 min each on Tuesday and Thursday
def hours_sat : ℕ := 2  -- 2 hours on Saturday

-- Define the total hours per week
def total_hours_per_week : ℕ := hours_mon_wed_fri + hours_tue_thu + hours_sat

-- Define the total hours in 2 weeks
def total_hours_in_2_weeks : ℕ := total_hours_per_week * 2

-- Prove that the total hours in 2 weeks is 12
theorem bethany_total_hours : total_hours_in_2_weeks = 12 :=
by
  -- Replace the definitions with their values and check the equality
  rw [total_hours_in_2_weeks, total_hours_per_week, hours_mon_wed_fri, hours_tue_thu, hours_sat]
  simp
  norm_num
  sorry

end bethany_total_hours_l293_293936


namespace parallel_vectors_solution_l293_293980

theorem parallel_vectors_solution 
  (x : ℝ) 
  (a : ℝ × ℝ := (-1, 3)) 
  (b : ℝ × ℝ := (x, 1)) 
  (h : ∃ k : ℝ, a = k • b) :
  x = -1 / 3 :=
by
  sorry

end parallel_vectors_solution_l293_293980


namespace primes_in_arithmetic_sequence_have_specific_ones_digit_l293_293073

-- Define the properties of the primes and the arithmetic sequence
theorem primes_in_arithmetic_sequence_have_specific_ones_digit
  (p q r s : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (prime_s : Nat.Prime s)
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4 ∧ s = r + 4)
  (p_gt_3 : p > 3) : 
  p % 10 = 9 := 
sorry

end primes_in_arithmetic_sequence_have_specific_ones_digit_l293_293073


namespace total_CDs_in_stores_l293_293338

def shelvesA := 5
def racksPerShelfA := 7
def cdsPerRackA := 8

def shelvesB := 4
def racksPerShelfB := 6
def cdsPerRackB := 7

def totalCDsA := shelvesA * racksPerShelfA * cdsPerRackA
def totalCDsB := shelvesB * racksPerShelfB * cdsPerRackB

def totalCDs := totalCDsA + totalCDsB

theorem total_CDs_in_stores :
  totalCDs = 448 := 
by 
  sorry

end total_CDs_in_stores_l293_293338


namespace man_l293_293285

theorem man's_speed_with_current (v c : ℝ) (h1 : c = 4.3) (h2 : v - c = 12.4) : v + c = 21 :=
by {
  sorry
}

end man_l293_293285


namespace weeks_in_year_span_l293_293071

def is_week_spanned_by_year (days_in_year : ℕ) (days_in_week : ℕ) (min_days_for_week : ℕ) : Prop :=
  ∃ weeks ∈ {53, 54}, days_in_year < weeks * days_in_week + min_days_for_week

theorem weeks_in_year_span (days_in_week : ℕ) (min_days_for_week : ℕ) :
  (is_week_spanned_by_year 365 days_in_week min_days_for_week ∨ is_week_spanned_by_year 366 days_in_week min_days_for_week) :=
by
  sorry

end weeks_in_year_span_l293_293071


namespace cos_180_eq_neg1_l293_293635

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l293_293635


namespace rose_needs_more_money_l293_293516

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l293_293516


namespace point_in_quadrant_l293_293985

theorem point_in_quadrant (m n : ℝ) (h₁ : 2 * (m - 1)^2 - 7 = -5) (h₂ : n > 3) :
  (m = 0 → 2*m - 3 < 0 ∧ (3*n - m)/2 > 0) ∧ 
  (m = 2 → 2*m - 3 > 0 ∧ (3*n - m)/2 > 0) :=
by 
  sorry

end point_in_quadrant_l293_293985


namespace find_x_l293_293104

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end find_x_l293_293104


namespace positiveDifferenceEquation_l293_293142

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l293_293142


namespace seventh_grader_count_l293_293120

variables {x n : ℝ}

noncomputable def number_of_seventh_graders (x n : ℝ) :=
  10 * x = 10 * x ∧  -- Condition 1
  4.5 * n = 4.5 * n ∧  -- Condition 2
  11 * x = 11 * x ∧  -- Condition 3
  5.5 * n = 5.5 * n ∧  -- Condition 4
  5.5 * n = (11 * x * (11 * x - 1)) / 2 ∧  -- Condition 5
  n = x * (11 * x - 1)  -- Condition 6

theorem seventh_grader_count (x n : ℝ) (h : number_of_seventh_graders x n) : x = 1 :=
  sorry

end seventh_grader_count_l293_293120


namespace min_value_expr_l293_293068

theorem min_value_expr (x y : ℝ) : ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + x * y + y^2 ≥ m) ∧ m = 0 :=
by
  sorry

end min_value_expr_l293_293068


namespace contracting_arrangements_correct_l293_293642

noncomputable def contracting_arrangements (projects : ℕ) (teams : ℕ) : ℕ :=
  if projects = 5 ∧ teams = 3 then 60 else sorry

theorem contracting_arrangements_correct :
  contracting_arrangements 5 3 = 60 := by
  sorry

end contracting_arrangements_correct_l293_293642


namespace three_pow_two_digits_count_l293_293476

theorem three_pow_two_digits_count : 
  ∃ n_set : Finset ℕ, (∀ n ∈ n_set, 10 ≤ 3^n ∧ 3^n < 100) ∧ n_set.card = 2 := 
sorry

end three_pow_two_digits_count_l293_293476


namespace cos_180_eq_neg1_l293_293592

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l293_293592


namespace min_abs_sum_half_l293_293138

theorem min_abs_sum_half :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  (∀ x, g x = Real.sin (2 * x + Real.pi / 3)) →
  (∀ x1 x2 : ℝ, g x1 * g x2 = -1 ∧ x1 ≠ x2 → abs ((x1 + x2) / 2) = Real.pi / 6) := by
-- Definitions and conditions are set, now we can state the theorem.
  sorry

end min_abs_sum_half_l293_293138


namespace impossible_to_empty_pile_l293_293881

theorem impossible_to_empty_pile (a b c : ℕ) (h : a = 1993 ∧ b = 199 ∧ c = 19) : 
  ¬ (∃ x y z : ℕ, (x + y + z = 0) ∧ (x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧ z = a ∨ z = b ∨ z = c)) := 
sorry

end impossible_to_empty_pile_l293_293881


namespace double_apply_l293_293069

def op1 (x : ℤ) : ℤ := 9 - x 
def op2 (x : ℤ) : ℤ := x - 9

theorem double_apply (x : ℤ) : op1 (op2 x) = 3 := by
  sorry

end double_apply_l293_293069


namespace probability_point_between_l_and_m_l293_293995

def l (x : ℝ) : ℝ := -2 * x + 8
def m (x : ℝ) : ℝ := -3 * x + 9

def area_under_l : ℝ := 0.5 * 4 * 8
def area_under_m : ℝ := 0.5 * 3 * 9

theorem probability_point_between_l_and_m : 
  (area_under_l - area_under_m) / area_under_l = 0.16 :=
by
  -- Variables to store areas for clarity
  have area_l : ℝ := 0.5 * 4 * 8
  have area_m : ℝ := 0.5 * 3 * 9

  -- Probability calculation
  calc (area_l - area_m) / area_l = 2.5 / 16 : by sorry
  ... = 0.15625 : by sorry
  ... ≈ 0.16 : by sorry

end probability_point_between_l_and_m_l293_293995


namespace red_marked_area_on_larger_sphere_l293_293920

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end red_marked_area_on_larger_sphere_l293_293920


namespace number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l293_293951

-- Definition for the number of sides in the polygon
def n : ℕ := 150

-- Definition of the formula for the number of diagonals
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the formula for the sum of interior angles
def sum_of_interior_angles (n : ℕ) : ℕ :=
  180 * (n - 2)

-- Theorem statements to be proved
theorem number_of_diagonals_is_correct : number_of_diagonals n = 11025 := sorry

theorem sum_of_interior_angles_is_correct : sum_of_interior_angles n = 26640 := sorry

end number_of_diagonals_is_correct_sum_of_interior_angles_is_correct_l293_293951


namespace john_task_completion_time_l293_293840

/-- John can complete a task alone in 18 days given the conditions. -/
theorem john_task_completion_time :
  ∀ (John Jane taskDays : ℝ), 
    Jane = 12 → 
    taskDays = 10.8 → 
    (10.8 - 6) * (1 / 12) + 10.8 * (1 / John) = 1 → 
    John = 18 :=
by
  intros John Jane taskDays hJane hTaskDays hWorkDone
  sorry

end john_task_completion_time_l293_293840


namespace find_f_3_l293_293810

def f (x : ℝ) : ℝ := x^2 + 4 * x + 8

theorem find_f_3 : f 3 = 29 := by
  sorry

end find_f_3_l293_293810


namespace solution_set_empty_l293_293736

theorem solution_set_empty (x : ℝ) : ¬ (|x| + |2023 - x| < 2023) :=
by
  sorry

end solution_set_empty_l293_293736


namespace cosine_180_degree_l293_293613

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l293_293613


namespace unique_int_function_l293_293448

noncomputable def unique_int_function_eq : Prop :=
  ∃! (f : ℤ → ℤ), ∀ (a b : ℤ), f(a + b) - f(ab) = f(a) * f(b) - 1

-- insert sorry here to indicate the proof is omitted
theorem unique_int_function : unique_int_function_eq :=
sorry

end unique_int_function_l293_293448


namespace triple_nested_application_l293_293502

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2 * n + 3

theorem triple_nested_application : g (g (g 3)) = 49 := by
  sorry

end triple_nested_application_l293_293502


namespace contractor_earnings_l293_293562

theorem contractor_earnings (total_days: ℕ) (wage_per_day: ℝ) (fine_per_absent_day: ℝ) (absent_days: ℕ) :
  total_days = 30 ∧ wage_per_day = 25 ∧ fine_per_absent_day = 7.5 ∧ absent_days = 10 →
  let worked_days := total_days - absent_days in
  let total_earned := worked_days * wage_per_day in
  let total_fine := absent_days * fine_per_absent_day in
  let final_amount := total_earned - total_fine in
  final_amount = 425 :=
begin
  sorry
end

end contractor_earnings_l293_293562


namespace find_making_lines_parallel_l293_293731

theorem find_making_lines_parallel (m : ℝ) : 
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2 
  (line1_slope = line2_slope) ↔ (m = 1) := 
by
  -- definitions
  intros
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2
  -- equation for slopes to be equal
  have slope_equation : line1_slope = line2_slope ↔ (m = 1)
  sorry

  exact slope_equation

end find_making_lines_parallel_l293_293731


namespace negation_of_p_l293_293989

variable {x : ℝ}

def proposition_p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

theorem negation_of_p :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
sorry

end negation_of_p_l293_293989


namespace speed_of_train_approx_29_0088_kmh_l293_293044

noncomputable def speed_of_train_in_kmh := 
  let length_train : ℝ := 288
  let length_bridge : ℝ := 101
  let time_seconds : ℝ := 48.29
  let total_distance : ℝ := length_train + length_bridge
  let speed_m_per_s : ℝ := total_distance / time_seconds
  speed_m_per_s * 3.6

theorem speed_of_train_approx_29_0088_kmh :
  abs (speed_of_train_in_kmh - 29.0088) < 0.001 := 
by
  sorry

end speed_of_train_approx_29_0088_kmh_l293_293044


namespace find_box_depth_l293_293178

-- Definitions and conditions
noncomputable def length : ℝ := 1.6
noncomputable def width : ℝ := 1.0
noncomputable def edge : ℝ := 0.2
noncomputable def number_of_blocks : ℝ := 120

-- The goal is to find the depth of the box
theorem find_box_depth (d : ℝ) :
  length * width * d = number_of_blocks * (edge ^ 3) →
  d = 0.6 := 
sorry

end find_box_depth_l293_293178


namespace sum_of_multiples_l293_293538

-- Define the three consecutive multiples of 5
def mult1 (x : ℝ) : ℝ := 5 * x - 5
def mult2 (x : ℝ) : ℝ := 5 * x
def mult3 (x : ℝ) : ℝ := 5 * x + 5

-- Define the product of the three multiples
def product (x : ℝ) : ℝ := (mult1 x) * (mult2 x) * (mult3 x)

-- Define the sum of the three multiples
def sum (x : ℝ) : ℝ := (mult1 x) + (mult2 x) + (mult3 x)

-- Define the condition based on the problem statement
noncomputable def problem_condition (x : ℝ) : Prop := product x = 30 * sum x

theorem sum_of_multiples : ∃ x, problem_condition x ∧ sum x = 30 + 15 * Real.sqrt 27 :=
by
  sorry

end sum_of_multiples_l293_293538


namespace solution_set_max_value_l293_293463

-- Given function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |x - 1|

-- (I) Prove the solution set of f(x) ≤ 4 is {x | -2/3 ≤ x ≤ 2}
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 2} :=
sorry

-- (II) Given m is the minimum value of f(x)
def m := 1 / 2

-- Given a, b, c ∈ ℝ^+ and a + b + c = m
variables (a b c : ℝ)
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a + b + c = m)

-- Prove the maximum value of √(2a + 1) + √(2b + 1) + √(2c + 1) is 2√3
theorem max_value : (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1)) ≤ 2 * Real.sqrt 3 :=
sorry

end solution_set_max_value_l293_293463


namespace modulo_arithmetic_l293_293941

theorem modulo_arithmetic :
  (222 * 15 - 35 * 9 + 2^3) % 18 = 17 :=
by
  sorry

end modulo_arithmetic_l293_293941


namespace simplify_fraction_l293_293702

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l293_293702


namespace not_square_a2_b2_ab_l293_293114

theorem not_square_a2_b2_ab (n : ℕ) (h_n : n > 2) (a : ℕ) (b : ℕ) (h_b : b = 2^(2^n))
  (h_a_odd : a % 2 = 1) (h_a_le_b : a ≤ b) (h_b_le_2a : b ≤ 2 * a) :
  ¬ ∃ k : ℕ, a^2 + b^2 - a * b = k^2 :=
by
  sorry

end not_square_a2_b2_ab_l293_293114


namespace tangent_line_circle_l293_293098

theorem tangent_line_circle (m : ℝ) : 
  (∀ (x y : ℝ), x + y + m = 0 → x^2 + y^2 = m) → m = 2 :=
by
  sorry

end tangent_line_circle_l293_293098


namespace probability_of_two_pairs_of_same_value_is_correct_l293_293522

def total_possible_outcomes := 6^6
def number_of_ways_to_form_pairs := 15
def choose_first_pair := 6
def choose_second_pair := 15
def choose_third_pair := 6
def choose_fourth_die := 4
def choose_fifth_die := 3

def successful_outcomes := number_of_ways_to_form_pairs *
                           choose_first_pair *
                           choose_second_pair *
                           choose_third_pair *
                           choose_fourth_die *
                           choose_fifth_die

def probability_of_two_pairs_of_same_value := (successful_outcomes : ℚ) / total_possible_outcomes

theorem probability_of_two_pairs_of_same_value_is_correct :
  probability_of_two_pairs_of_same_value = 25 / 72 :=
by
  -- proof omitted
  sorry

end probability_of_two_pairs_of_same_value_is_correct_l293_293522


namespace puppies_per_female_dog_l293_293375

theorem puppies_per_female_dog
  (number_of_dogs : ℕ)
  (percent_female : ℝ)
  (fraction_female_giving_birth : ℝ)
  (remaining_puppies : ℕ)
  (donated_puppies : ℕ)
  (total_puppies : ℕ)
  (number_of_female_dogs : ℕ)
  (number_female_giving_birth : ℕ)
  (puppies_per_dog : ℕ) :
  number_of_dogs = 40 →
  percent_female = 0.60 →
  fraction_female_giving_birth = 0.75 →
  remaining_puppies = 50 →
  donated_puppies = 130 →
  total_puppies = remaining_puppies + donated_puppies →
  number_of_female_dogs = percent_female * number_of_dogs →
  number_female_giving_birth = fraction_female_giving_birth * number_of_female_dogs →
  puppies_per_dog = total_puppies / number_female_giving_birth →
  puppies_per_dog = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end puppies_per_female_dog_l293_293375


namespace tomatoes_on_each_plant_l293_293298

/-- Andy harvests all the tomatoes from 18 plants that have a certain number of tomatoes each.
    He dries half the tomatoes and turns a third of the remainder into marinara sauce. He has
    42 tomatoes left. Prove that the number of tomatoes on each plant is 7.  -/
theorem tomatoes_on_each_plant (T : ℕ) (h1 : ∀ n, n = 18 * T)
  (h2 : ∀ m, m = (18 * T) / 2)
  (h3 : ∀ k, k = m / 3)
  (h4 : ∀ final, final = m - k ∧ final = 42) : T = 7 :=
by
  sorry

end tomatoes_on_each_plant_l293_293298


namespace find_alpha_plus_beta_l293_293451

theorem find_alpha_plus_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = (Real.sqrt 5) / 5) (h4 : Real.sin β = (3 * Real.sqrt 10) / 10) : 
  α + β = 3 * π / 4 :=
sorry

end find_alpha_plus_beta_l293_293451


namespace boxes_needed_to_complete_flooring_l293_293002

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l293_293002


namespace fair_tickets_more_than_twice_baseball_tickets_l293_293732

theorem fair_tickets_more_than_twice_baseball_tickets :
  ∃ (fair_tickets baseball_tickets : ℕ), 
    fair_tickets = 25 ∧ baseball_tickets = 56 ∧ 
    fair_tickets + 87 = 2 * baseball_tickets := 
by
  sorry

end fair_tickets_more_than_twice_baseball_tickets_l293_293732


namespace smallest_angle_terminal_side_l293_293262

theorem smallest_angle_terminal_side (θ : ℝ) (H : θ = 2011) :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 360 ∧ (∃ k : ℤ, φ = θ - 360 * k) ∧ φ = 211 :=
by
  sorry

end smallest_angle_terminal_side_l293_293262


namespace greatest_integer_radius_l293_293216

theorem greatest_integer_radius (r : ℕ) :
  (π * (r: ℝ)^2 < 30 * π) ∧ (2 * π * (r: ℝ) > 10 * π) → r = 5 :=
by
  sorry

end greatest_integer_radius_l293_293216


namespace even_function_symmetric_y_axis_l293_293431

theorem even_function_symmetric_y_axis (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = f (-x) := by
  sorry

end even_function_symmetric_y_axis_l293_293431


namespace min_rectilinear_distance_to_parabola_l293_293903

theorem min_rectilinear_distance_to_parabola :
  ∃ t : ℝ, ∀ t', (|t' + 1| + t'^2) ≥ (|t + 1| + t^2) ∧ (|t + 1| + t^2) = 3 / 4 := sorry

end min_rectilinear_distance_to_parabola_l293_293903


namespace square_side_length_difference_l293_293130

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end square_side_length_difference_l293_293130


namespace fraction_zero_condition_l293_293332

theorem fraction_zero_condition (x : ℝ) (h1 : (3 - |x|) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end fraction_zero_condition_l293_293332


namespace minimum_value_real_l293_293891

theorem minimum_value_real (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  use -9,
  sorry

end minimum_value_real_l293_293891


namespace point_A_in_second_quadrant_l293_293836

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l293_293836


namespace mixture_problem_l293_293667

theorem mixture_problem
  (x : ℝ)
  (c1 c2 c_final : ℝ)
  (v1 v2 v_final : ℝ)
  (h1 : c1 = 0.60)
  (h2 : c2 = 0.75)
  (h3 : c_final = 0.72)
  (h4 : v1 = 4)
  (h5 : x = 16)
  (h6 : v2 = x)
  (h7 : v_final = v1 + v2) :
  v_final = 20 ∧ c_final * v_final = c1 * v1 + c2 * v2 :=
by
  sorry

end mixture_problem_l293_293667


namespace marigold_ratio_l293_293693

theorem marigold_ratio :
  ∃ x, 14 + 25 + x = 89 ∧ x / 25 = 2 := by
  sorry

end marigold_ratio_l293_293693


namespace incorrect_expression_l293_293671

variable {x y : ℚ}

theorem incorrect_expression (h : x / y = 5 / 3) : (x - 2 * y) / y ≠ 1 / 3 := by
  have h1 : x / y = 5 / 3 := h
  have h2 : (x - 2 * y) / y = (x / y) - (2 * y) / y := by sorry
  have h3 : (x - 2 * y) / y = (5 / 3) - 2 := by sorry
  have h4 : (x - 2 * y) / y = (5 / 3) - (6 / 3) := by sorry
  have h5 : (x - 2 * y) / y = -1 / 3 := by sorry
  exact sorry

end incorrect_expression_l293_293671


namespace cashier_overestimation_l293_293169

def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

def nickels_counted_as_dimes := 15
def quarters_counted_as_half_dollars := 10

noncomputable def overestimation_due_to_nickels_as_dimes : Nat := 
  (dime_value - nickel_value) * nickels_counted_as_dimes

noncomputable def overestimation_due_to_quarters_as_half_dollars : Nat := 
  (half_dollar_value - quarter_value) * quarters_counted_as_half_dollars

noncomputable def total_overestimation : Nat := 
  overestimation_due_to_nickels_as_dimes + overestimation_due_to_quarters_as_half_dollars

theorem cashier_overestimation : total_overestimation = 325 := by
  sorry

end cashier_overestimation_l293_293169


namespace right_triangle_legs_sum_l293_293730

theorem right_triangle_legs_sum
  (x : ℕ)
  (h_even : Even x)
  (h_eq : x^2 + (x + 2)^2 = 34^2) :
  x + (x + 2) = 50 := 
by
  sorry

end right_triangle_legs_sum_l293_293730


namespace geometric_sequence_divisible_by_ten_million_l293_293234

theorem geometric_sequence_divisible_by_ten_million 
  (a1 a2 : ℝ)
  (h1 : a1 = 1 / 2)
  (h2 : a2 = 50) :
  ∀ n : ℕ, (n ≥ 5) → (∃ k : ℕ, (a1 * (a2 / a1)^(n - 1)) = k * 10^7) :=
by
  sorry

end geometric_sequence_divisible_by_ten_million_l293_293234


namespace beads_left_in_container_l293_293742

theorem beads_left_in_container 
  (initial_beads green brown red total_beads taken_beads remaining_beads : Nat) 
  (h1 : green = 1) (h2 : brown = 2) (h3 : red = 3) 
  (h4 : total_beads = green + brown + red)
  (h5 : taken_beads = 2) 
  (h6 : remaining_beads = total_beads - taken_beads) : 
  remaining_beads = 4 := 
by
  sorry

end beads_left_in_container_l293_293742


namespace findAnalyticalExpression_l293_293533

-- Defining the point A as a structure with x and y coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Defining a line as having a slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Condition: Line 1 is parallel to y = 2x - 3
def line1 : Line := {slope := 2, intercept := -3}

-- Condition: Line 2 passes through point A
def point_A : Point := {x := -2, y := -1}

-- The theorem statement:
theorem findAnalyticalExpression : 
  ∃ b : ℝ, (∀ x : ℝ, (point_A.y = line1.slope * point_A.x + b) → b = 3) ∧ 
            ∀ x : ℝ, (line1.slope * x + b = 2 * x + 3) :=
sorry

end findAnalyticalExpression_l293_293533


namespace positive_difference_between_two_numbers_l293_293145

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l293_293145


namespace problem1_problem2_l293_293586

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end problem1_problem2_l293_293586


namespace number_of_students_l293_293050

theorem number_of_students (S : ℕ) (hS1 : S ≥ 2) (hS2 : S ≤ 80) 
                          (hO : ∀ n : ℕ, (n * S) % 120 = 0) : 
    S = 40 :=
sorry

end number_of_students_l293_293050


namespace gcd_2000_7700_l293_293400

theorem gcd_2000_7700 : Nat.gcd 2000 7700 = 100 := by
  -- Prime factorizations of 2000 and 7700
  have fact_2000 : 2000 = 2^4 * 5^3 := sorry
  have fact_7700 : 7700 = 2^2 * 5^2 * 7 * 11 := sorry
  -- Proof of gcd
  sorry

end gcd_2000_7700_l293_293400


namespace least_positive_integer_solution_l293_293447

theorem least_positive_integer_solution :
  ∃ x : ℕ, (x + 7391) % 12 = 167 % 12 ∧ x = 8 :=
by 
  sorry

end least_positive_integer_solution_l293_293447


namespace exists_k_seq_zero_to_one_l293_293080

noncomputable def seq (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) := a

theorem exists_k_seq_zero_to_one (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) :
  ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 :=
sorry

end exists_k_seq_zero_to_one_l293_293080


namespace total_amount_received_l293_293909

-- Definitions based on conditions
def days_A : Nat := 6
def days_B : Nat := 8
def days_ABC : Nat := 3

def share_A : Nat := 300
def share_B : Nat := 225
def share_C : Nat := 75

-- The theorem stating the total amount received for the work
theorem total_amount_received (dA dB dABC : Nat) (sA sB sC : Nat)
  (h1 : dA = days_A) (h2 : dB = days_B) (h3 : dABC = days_ABC)
  (h4 : sA = share_A) (h5 : sB = share_B) (h6 : sC = share_C) : 
  sA + sB + sC = 600 := by
  sorry

end total_amount_received_l293_293909


namespace length_BC_l293_293998

noncomputable def center (O : Type) : Prop := sorry   -- Center of the circle.

noncomputable def diameter (AD : Type) : Prop := sorry   -- AD is a diameter.

noncomputable def chord (ABC : Type) : Prop := sorry   -- ABC is a chord.

noncomputable def radius_equal (BO : ℝ) : Prop := BO = 8   -- BO = 8.

noncomputable def angle_ABO (α : ℝ) : Prop := α = 45   -- ∠ABO = 45°.

noncomputable def arc_CD (β : ℝ) : Prop := β = 90   -- Arc CD subtended by ∠AOD = 90°.

theorem length_BC (O AD ABC : Type) (BO : ℝ) (α β γ : ℝ)
  (h1 : center O)
  (h2 : diameter AD)
  (h3 : chord ABC)
  (h4 : radius_equal BO)
  (h5 : angle_ABO α)
  (h6 : arc_CD β)
  : γ = 8 := 
sorry

end length_BC_l293_293998


namespace simplify_fraction_l293_293714

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l293_293714


namespace perpendicular_line_x_intercept_l293_293008

noncomputable def slope (a b : ℚ) : ℚ := - a / b

noncomputable def line_equation (m y_intercept : ℚ) (x : ℚ) : ℚ :=
  m * x + y_intercept

theorem perpendicular_line_x_intercept :
  let m1 := slope 4 5,
      m2 := (5 / 4),
      y_int := -3
  in 
  ∀ x, line_equation m2 y_int x = 0 → x = 12 / 5 :=
by
  intro x hx
  sorry

end perpendicular_line_x_intercept_l293_293008


namespace cos_180_eq_neg1_l293_293633

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l293_293633


namespace slope_and_angle_of_inclination_l293_293540

noncomputable def line_slope_and_inclination : Prop :=
  ∀ (x y : ℝ), (x - y - 3 = 0) → (∃ m : ℝ, m = 1) ∧ (∃ θ : ℝ, θ = 45)

theorem slope_and_angle_of_inclination (x y : ℝ) (h : x - y - 3 = 0) : line_slope_and_inclination :=
by
  sorry

end slope_and_angle_of_inclination_l293_293540


namespace find_union_A_B_r_find_range_m_l293_293082

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x m : ℝ) : Prop := (x - m) * (x - m - 1) ≥ 0

theorem find_union_A_B_r (x : ℝ) : A x ∨ B x 1 := by
  sorry

theorem find_range_m (m : ℝ) (x : ℝ) : (∀ x, A x ↔ B x m) ↔ (m ≥ 3 ∨ m ≤ -2) := by
  sorry

end find_union_A_B_r_find_range_m_l293_293082


namespace min_polyline_distance_l293_293831

-- Define the polyline distance between two points P(x1, y1) and Q(x2, y2).
noncomputable def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Define the circle x^2 + y^2 = 1.
def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 = 1

-- Define the line 2x + y = 2√5.
def on_line (P : ℝ × ℝ) : Prop :=
  2 * P.1 + P.2 = 2 * Real.sqrt 5

-- Statement of the minimum distance problem.
theorem min_polyline_distance : 
  ∀ P Q : ℝ × ℝ, on_circle P → on_line Q → 
  polyline_distance P Q ≥ Real.sqrt 5 / 2 :=
sorry

end min_polyline_distance_l293_293831


namespace dima_and_serezha_meet_time_l293_293791

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end dima_and_serezha_meet_time_l293_293791


namespace hyperbola_asymptotes_l293_293205

theorem hyperbola_asymptotes (a : ℝ) (x y : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  (∃ M : ℝ × ℝ, M.1 ^ 2 / a ^ 2 - M.2 ^ 2 = 1 ∧ M.2 ^ 2 = 8 * M.1 ∧ abs (dist M F) = 5) →
  (F.1 = 2 ∧ F.2 = 0) →
  (a = 3 / 5) → 
  (∀ x y : ℝ, (5 * x + 3 * y = 0) ∨ (5 * x - 3 * y = 0)) :=
by
  sorry

end hyperbola_asymptotes_l293_293205


namespace VincentLearnedAtCamp_l293_293746

def VincentSongsBeforeSummerCamp : ℕ := 56
def VincentSongsAfterSummerCamp : ℕ := 74

theorem VincentLearnedAtCamp :
  VincentSongsAfterSummerCamp - VincentSongsBeforeSummerCamp = 18 := by
  sorry

end VincentLearnedAtCamp_l293_293746


namespace distance_between_points_l293_293802

theorem distance_between_points :
  let p1 := (-4, 17)
  let p2 := (12, -1)
  let distance := Real.sqrt ((12 - (-4))^2 + (-1 - 17)^2)
  distance = 2 * Real.sqrt 145 := sorry

end distance_between_points_l293_293802


namespace sin_cos_values_trigonometric_expression_value_l293_293657

-- Define the conditions
variables (α : ℝ)
def point_on_terminal_side (x y : ℝ) (r : ℝ) : Prop :=
  (x = 3) ∧ (y = 4) ∧ (r = 5)

-- Define the problem statements
theorem sin_cos_values (x y r : ℝ) (h: point_on_terminal_side x y r) : 
  (Real.sin α = 4 / 5) ∧ (Real.cos α = 3 / 5) :=
sorry

theorem trigonometric_expression_value (h1: Real.sin α = 4 / 5) (h2: Real.cos α = 3 / 5) :
  (2 * Real.cos (π / 2 - α) - Real.cos (π + α)) / (2 * Real.sin (π - α)) = 11 / 8 :=
sorry

end sin_cos_values_trigonometric_expression_value_l293_293657


namespace fraction_simplification_l293_293521

theorem fraction_simplification : (98 / 210 : ℚ) = 7 / 15 := 
by 
  sorry

end fraction_simplification_l293_293521


namespace find_x_plus_y_l293_293824

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 7 * y = 17) (h2 : 3 * x + 5 * y = 11) : x + y = 83 / 23 :=
sorry

end find_x_plus_y_l293_293824


namespace find_number_l293_293046

theorem find_number (x : ℝ) (h : (((x + 45) / 2) / 2) + 45 = 85) : x = 115 :=
by
  sorry

end find_number_l293_293046


namespace log_property_l293_293303

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem log_property (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m + f n :=
by
  sorry

end log_property_l293_293303


namespace cos_180_proof_l293_293622

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l293_293622


namespace expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l293_293047

-- Definitions to add parentheses in the given expressions to achieve the desired results.
def expr1 := 7 * (9 + 12 / 3)
def expr2 := (7 * 9 + 12) / 3
def expr3 := 7 * (9 + 12) / 3
def expr4 := (48 * 6) / (48 * 6)

-- Proof statements
theorem expr1_is_91 : expr1 = 91 := 
by sorry

theorem expr2_is_25 : expr2 = 25 :=
by sorry

theorem expr3_is_49 : expr3 = 49 :=
by sorry

theorem expr4_is_1 : expr4 = 1 :=
by sorry

end expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l293_293047


namespace cos_180_proof_l293_293623

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l293_293623


namespace correct_A_correct_B_intersection_A_B_complement_B_l293_293975

noncomputable def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem correct_A : A = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem correct_B : B = {x : ℝ | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem complement_B : (Bᶜ) = {x : ℝ | x < 1 ∨ x > 4} :=
by
  sorry

end correct_A_correct_B_intersection_A_B_complement_B_l293_293975


namespace fraction_compare_l293_293666

theorem fraction_compare (a b c d e : ℚ) : 
  a = 0.3333333 → 
  b = 1 / (3 * 10^6) →
  ∃ x : ℚ, 
  x = 1 / 3 ∧ 
  (x > a + d ∧ 
   x = a + b ∧
   d = b ∧
   d = -1 / (3 * 10^6)) := 
  sorry

end fraction_compare_l293_293666


namespace cos_180_proof_l293_293624

def cos_180_eq : Prop :=
  cos (Real.pi) = -1

theorem cos_180_proof : cos_180_eq := by
  sorry

end cos_180_proof_l293_293624


namespace quad_inequality_solution_set_is_reals_l293_293531

theorem quad_inequality_solution_set_is_reals (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) := 
sorry

end quad_inequality_solution_set_is_reals_l293_293531


namespace range_of_m_l293_293215

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)) :
  0 < m ∧ m ≤ 1 / Real.exp 1 :=
sorry

end range_of_m_l293_293215


namespace tedra_tomato_harvest_l293_293867

theorem tedra_tomato_harvest (W T F : ℝ) 
    (h1 : T = W / 2) 
    (h2 : W + T + F = 2000) 
    (h3 : F - 700 = 700) : 
    W = 400 := 
sorry

end tedra_tomato_harvest_l293_293867


namespace find_B_l293_293323

theorem find_B (A B : Nat) (hA : A ≤ 9) (hB : B ≤ 9) (h_eq : 6 * A + 10 * B + 2 = 77) : B = 1 :=
by
-- proof steps would go here
sorry

end find_B_l293_293323


namespace sum_of_numbers_of_large_cube_l293_293907

def sum_faces_of_die := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice := 125

def number_of_faces_per_die := 6

def total_exposed_faces (side_length: ℕ) : ℕ := 6 * (side_length * side_length)

theorem sum_of_numbers_of_large_cube (side_length : ℕ) (dice_count : ℕ) 
    (sum_per_die : ℕ) (opposite_face_sum : ℕ) :
    dice_count = 125 →
    total_exposed_faces side_length = 150 →
    sum_per_die = 21 →
    (∀ f1 f2, (f1 + f2 = opposite_face_sum)) →
    dice_count * sum_per_die = 2625 →
    (210 ≤ dice_count * sum_per_die ∧ dice_count * sum_per_die ≤ 840) :=
by 
  intro h_dice_count
  intro h_exposed_faces
  intro h_sum_per_die
  intro h_opposite_faces
  intro h_total_sum
  sorry

end sum_of_numbers_of_large_cube_l293_293907


namespace sarah_can_gather_info_l293_293488

noncomputable def probability_gather_info_both_classes : ℚ :=
  let total_students := 30
  let german_students := 22
  let italian_students := 26
  let both_classes_students := german_students + italian_students - total_students
  let only_german := german_students - both_classes_students
  let only_italian := italian_students - both_classes_students
  let total_pairs := (total_students * (total_students - 1)) / 2
  let german_pairs := (only_german * (only_german - 1)) / 2
  let italian_pairs := (only_italian * (only_italian - 1)) / 2
  let unfavorable_pairs := german_pairs + italian_pairs
  let favorable_pairs := total_pairs - unfavorable_pairs
  favorable_pairs /. total_pairs

theorem sarah_can_gather_info :
  probability_gather_info_both_classes = 401 / 435 :=
sorry

end sarah_can_gather_info_l293_293488


namespace loaf_bread_cost_correct_l293_293897

-- Given conditions
def total : ℕ := 32
def candy_bar : ℕ := 2
def final_remaining : ℕ := 18

-- Intermediate calculations as definitions
def remaining_after_candy_bar : ℕ := total - candy_bar
def turkey_cost : ℕ := remaining_after_candy_bar / 3
def remaining_after_turkey : ℕ := remaining_after_candy_bar - turkey_cost
def loaf_bread_cost : ℕ := remaining_after_turkey - final_remaining

-- Theorem stating the problem question and expected answer
theorem loaf_bread_cost_correct : loaf_bread_cost = 2 :=
sorry

end loaf_bread_cost_correct_l293_293897


namespace solution_set_of_inequality_l293_293735

theorem solution_set_of_inequality: 
  {x : ℝ | (2 * x - 1) / x < 1} = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_inequality_l293_293735


namespace total_amount_shared_l293_293853

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end total_amount_shared_l293_293853


namespace exists_c_d_in_set_of_13_reals_l293_293246

theorem exists_c_d_in_set_of_13_reals (a : Fin 13 → ℝ) :
  ∃ (c d : ℝ), c ∈ Set.range a ∧ d ∈ Set.range a ∧ 0 < (c - d) / (1 + c * d) ∧ (c - d) / (1 + c * d) < 2 - Real.sqrt 3 := 
by
  sorry

end exists_c_d_in_set_of_13_reals_l293_293246


namespace simplify_expression_l293_293477

theorem simplify_expression (x y : ℝ) (h : y = x / (1 - 2 * x)) :
    (2 * x - 3 * x * y - 2 * y) / (y + x * y - x) = -7 / 3 := 
by {
  sorry
}

end simplify_expression_l293_293477


namespace xena_escape_l293_293274

theorem xena_escape
    (head_start : ℕ)
    (safety_distance : ℕ)
    (xena_speed : ℕ)
    (dragon_speed : ℕ)
    (effective_gap : ℕ := head_start - safety_distance)
    (speed_difference : ℕ := dragon_speed - xena_speed) :
    (time_to_safety : ℕ := effective_gap / speed_difference) →
    time_to_safety = 32 :=
by
  sorry

end xena_escape_l293_293274


namespace perpendicular_line_x_intercept_l293_293011

theorem perpendicular_line_x_intercept (x y : ℝ) :
  (4 * x + 5 * y = 10) →
  (1 * y + 0 = y → y = (5 / 4) * x - 3) →
  y = 0 →
  x = 12 / 5 :=
begin
  sorry
end

end perpendicular_line_x_intercept_l293_293011


namespace contractor_earnings_l293_293561

theorem contractor_earnings (total_days: ℕ) (wage_per_day: ℝ) (fine_per_absent_day: ℝ) (absent_days: ℕ) :
  total_days = 30 ∧ wage_per_day = 25 ∧ fine_per_absent_day = 7.5 ∧ absent_days = 10 →
  let worked_days := total_days - absent_days in
  let total_earned := worked_days * wage_per_day in
  let total_fine := absent_days * fine_per_absent_day in
  let final_amount := total_earned - total_fine in
  final_amount = 425 :=
begin
  sorry
end

end contractor_earnings_l293_293561


namespace airplane_time_in_air_l293_293775

-- Define conditions
def distance_seaport_island := 840  -- Total distance in km
def speed_icebreaker := 20          -- Speed of the icebreaker in km/h
def time_icebreaker := 22           -- Total time the icebreaker traveled in hours
def speed_airplane := 120           -- Speed of the airplane in km/h

-- Prove the time the airplane spent in the air
theorem airplane_time_in_air : (distance_seaport_island - speed_icebreaker * time_icebreaker) / speed_airplane = 10 / 3 := by
  -- This is where the proof steps would go, but we're placing sorry to skip it for now.
  sorry

end airplane_time_in_air_l293_293775


namespace find_x_l293_293055

noncomputable def arctan := Real.arctan

theorem find_x :
  (∃ x : ℝ, 3 * arctan (1 / 4) + arctan (1 / 5) + arctan (1 / x) = π / 4 ∧ x = -250 / 37) :=
  sorry

end find_x_l293_293055


namespace cosine_180_degree_l293_293614

def on_unit_circle (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

def rotate_point_180 (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

example : on_unit_circle 1 0 :=
by {
  unfold on_unit_circle,
  norm_num,
}

example : rotate_point_180 1 0 = (-1, 0) :=
by {
  unfold rotate_point_180,
  norm_num,
}

theorem cosine_180_degree : ∃ x : ℝ, on_unit_circle x 0 ∧ rotate_point_180 x 0 = (-1, 0) ∧ x = -1 :=
by {
  use -1,
  split,
  {
    unfold on_unit_circle,
    norm_num,
  },
  split,
  {
    unfold rotate_point_180,
    norm_num,
  },
  norm_num,
}

end cosine_180_degree_l293_293614


namespace scientific_notation_example_l293_293036

theorem scientific_notation_example : (5.2 * 10^5) = 520000 := sorry

end scientific_notation_example_l293_293036


namespace gasoline_added_l293_293281

variable (tank_capacity : ℝ := 42)
variable (initial_fill_fraction : ℝ := 3/4)
variable (final_fill_fraction : ℝ := 9/10)

theorem gasoline_added :
  let initial_amount := tank_capacity * initial_fill_fraction
  let final_amount := tank_capacity * final_fill_fraction
  final_amount - initial_amount = 6.3 :=
by
  sorry

end gasoline_added_l293_293281


namespace swap_values_l293_293811

theorem swap_values (A B : ℕ) (h₁ : A = 10) (h₂ : B = 20) : 
    let C := A 
    let A := B 
    let B := C
    A = 20 ∧ B = 10 := by
  let C := A
  let A := B
  let B := C
  have h₃ : C = 10 := h₁
  have h₄ : A = 20 := h₂
  have h₅ : B = 10 := h₃
  exact And.intro h₄ h₅

end swap_values_l293_293811


namespace find_a_l293_293972

noncomputable def f (t : ℝ) (a : ℝ) : ℝ := (1 / (Real.cos t)) + (a / (1 - (Real.cos t)))

theorem find_a (t : ℝ) (a : ℝ) (h1 : 0 < t) (h2 : t < (Real.pi / 2)) (h3 : 0 < a) (h4 : ∀ t, 0 < t ∧ t < (Real.pi / 2) → f t a = 16) :
  a = 9 :=
sorry

end find_a_l293_293972


namespace problem1_problem2_l293_293347

-- Definitions of M and N
def setM : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def setN (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

-- Problem 1: Prove that if M ∩ N has only one element, then k = -1
theorem problem1 (h : ∀ x, x ∈ setM ∩ setN k → x = -1) : k = -1 := by 
  sorry

-- Problem 2: Given k = 2, prove the sets M ∩ N and M ∪ N
theorem problem2 (hk : k = 2) : (setM ∩ setN k = {x | -1 ≤ x ∧ x ≤ 2}) ∧ (setM ∪ setN k = {x | x ≤ 5}) := by
  sorry

end problem1_problem2_l293_293347


namespace inequality_proof_l293_293455

variable (x y z : ℝ)

theorem inequality_proof (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  x * (1 - 2 * x) * (1 - 3 * x) + y * (1 - 2 * y) * (1 - 3 * y) + z * (1 - 2 * z) * (1 - 3 * z) ≥ 0 := 
sorry

end inequality_proof_l293_293455


namespace problem1_problem2_l293_293780

theorem problem1 : 24 - (-16) + (-25) - 15 = 0 :=
by
  sorry

theorem problem2 : (-81) + 2 * (1 / 4) * (4 / 9) / (-16) = -81 - (1 / 16) :=
by
  sorry

end problem1_problem2_l293_293780


namespace parallelogram_base_length_l293_293459

theorem parallelogram_base_length (b h : ℝ) (area : ℝ) (angle : ℝ) (h_area : area = 200) 
(h_altitude : h = 2 * b) (h_angle : angle = 60) : b = 10 :=
by
  -- Placeholder for proof
  sorry

end parallelogram_base_length_l293_293459


namespace power_equal_20mn_l293_293497

theorem power_equal_20mn (m n : ℕ) (P Q : ℕ) (hP : P = 2^m) (hQ : Q = 5^n) : 
  P^(2 * n) * Q^m = (20^(m * n)) :=
by
  sorry

end power_equal_20mn_l293_293497


namespace right_triangle_acute_angle_l293_293769

theorem right_triangle_acute_angle (x : ℝ) 
  (h1 : 5 * x = 90) : x = 18 :=
by sorry

end right_triangle_acute_angle_l293_293769


namespace area_on_larger_sphere_l293_293919

-- Define the variables representing the radii and the given area on the smaller sphere
variable (r1 r2 : ℝ) (area1 : ℝ)

-- Given conditions
def conditions : Prop :=
  r1 = 4 ∧ r2 = 6 ∧ area1 = 37

-- Define the statement that we need to prove
theorem area_on_larger_sphere (h : conditions r1 r2 area1) : 
  let area2 := area1 * (r2^2 / r1^2) in
  area2 = 83.25 :=
by
  -- Insert the proof here
  sorry

end area_on_larger_sphere_l293_293919


namespace pizza_slices_leftover_l293_293186

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end pizza_slices_leftover_l293_293186


namespace probability_same_color_shoes_l293_293020

theorem probability_same_color_shoes (pairs : ℕ) (total_shoes : ℕ)
  (each_pair_diff_color : pairs * 2 = total_shoes)
  (select_2_without_replacement : total_shoes = 10 ∧ pairs = 5) :
  let successful_outcomes := pairs
  let total_outcomes := (total_shoes * (total_shoes - 1)) / 2
  successful_outcomes / total_outcomes = 1 / 9 :=
by
  sorry

end probability_same_color_shoes_l293_293020


namespace sum_series_l293_293787

theorem sum_series : ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2)) = 1 := 
sorry

end sum_series_l293_293787


namespace part_a_l293_293899

theorem part_a (a : ℤ) : (a^2 < 4) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := 
sorry

end part_a_l293_293899


namespace pushups_total_l293_293504

theorem pushups_total (x melanie david karen john : ℕ) 
  (hx : x = 51)
  (h_melanie : melanie = 2 * x - 7)
  (h_david : david = x + 22)
  (h_avg : (x + melanie + david) / 3 = (x + (2 * x - 7) + (x + 22)) / 3)
  (h_karen : karen = (x + (2 * x - 7) + (x + 22)) / 3 - 5)
  (h_john : john = (x + 22) - 4) :
  john + melanie + karen = 232 := by
  sorry

end pushups_total_l293_293504


namespace acute_angle_of_rhombus_l293_293957

theorem acute_angle_of_rhombus (a α : ℝ) (V1 V2 : ℝ) (OA BD AN AB : ℝ) 
  (h_volumes : V1 / V2 = 1 / (2 * Real.sqrt 5)) 
  (h_V1 : V1 = (1 / 3) * Real.pi * (OA^2) * BD)
  (h_V2 : V2 = Real.pi * (AN^2) * AB)
  (h_OA : OA = a * Real.sin (α / 2))
  (h_BD : BD = 2 * a * Real.cos (α / 2))
  (h_AN : AN = a * Real.sin α)
  (h_AB : AB = a)
  : α = Real.arccos (1 / 9) :=
sorry

end acute_angle_of_rhombus_l293_293957


namespace fraction_after_adding_liters_l293_293570

-- Given conditions
variables (c w : ℕ)
variables (h1 : w = c / 3)
variables (h2 : (w + 5) / c = 2 / 5)

-- The proof statement
theorem fraction_after_adding_liters (h1 : w = c / 3) (h2 : (w + 5) / c = 2 / 5) : 
  (w + 9) / c = 34 / 75 :=
sorry -- Proof omitted

end fraction_after_adding_liters_l293_293570


namespace simplify_fraction_l293_293704

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l293_293704


namespace solve_for_x_l293_293102

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end solve_for_x_l293_293102


namespace circle_m_range_l293_293379

theorem circle_m_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2 * x + 6 * y + m = 0 → m < 10) :=
sorry

end circle_m_range_l293_293379


namespace intersection_M_N_l293_293819

noncomputable def M := {x : ℝ | x > 1}
noncomputable def N := {x : ℝ | x < 2}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l293_293819


namespace three_out_of_five_correct_prob_l293_293543

theorem three_out_of_five_correct_prob :
  let n := 5 in
  let favorable := (numberOfCombinations n 3) * derangements 2 in
  let total := factorial n in
  favorable / total = 1 / 12 := by
  sorry

end three_out_of_five_correct_prob_l293_293543


namespace five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l293_293236

def is_palindromic (n : Nat) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_5 (n : Nat) : Prop := 
  n % 5 = 0

theorem five_digit_palindromic_div_by_5_example : 
  ∃ n : Nat, 10000 ≤ n ∧ n < 100000 ∧ is_palindromic n ∧ is_divisible_by_5 n ∧ n = 51715 :=
by
  sorry

theorem count_five_digit_palindromic_div_by_5 : 
  ∑ n in (Finset.range 100000).filter (λ n => 10000 ≤ n ∧ is_palindromic n ∧ is_divisible_by_5 n), 1 = 100 :=
by
  sorry

end five_digit_palindromic_div_by_5_example_count_five_digit_palindromic_div_by_5_l293_293236


namespace graph_of_eqn_is_pair_of_lines_l293_293728

theorem graph_of_eqn_is_pair_of_lines : 
  ∃ (l₁ l₂ : ℝ × ℝ → Prop), 
  (∀ x y, l₁ (x, y) ↔ x = 2 * y) ∧ 
  (∀ x y, l₂ (x, y) ↔ x = -2 * y) ∧ 
  (∀ x y, (x^2 - 4 * y^2 = 0) ↔ (l₁ (x, y) ∨ l₂ (x, y))) :=
by
  sorry

end graph_of_eqn_is_pair_of_lines_l293_293728


namespace simplify_fraction_l293_293701

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l293_293701


namespace real_cube_inequality_l293_293842

theorem real_cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end real_cube_inequality_l293_293842


namespace johns_phone_price_l293_293773

-- Define Alan's phone price
def alans_price : ℝ := 2000

-- Define the percentage increase
def percentage_increase : ℝ := 2/100

-- Define John's phone price
def johns_price := alans_price * (1 + percentage_increase)

-- The main theorem
theorem johns_phone_price : johns_price = 2040 := by
  sorry

end johns_phone_price_l293_293773


namespace monotonic_intervals_value_of_a_inequality_a_minus_one_l293_293085

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem monotonic_intervals (a : ℝ) :
  (∀ x, 0 < x → 0 ≤ a → 0 < (a * x + 1) / x) ∧
  (∀ x, 0 < x → a < 0 → (0 < x ∧ x < -1/a → 0 < (a * x + 1) / x) ∧
    (-1/a < x → 0 > (a * x + 1) / x)) :=
sorry

theorem value_of_a (a : ℝ) (h_a : a < 0) (h_max : (∀ x, x ∈ Set.Icc 0 e → f a x ≤ -2) ∧ (∃ x, x ∈ Set.Icc 0 e ∧ f a x = -2)) :
  a = -Real.exp 1 := 
sorry

theorem inequality_a_minus_one (a : ℝ) (h_a : a = -1) :
  (∀ x, 0 < x → x * |f a x| > Real.log x + 1/2 * x) :=
sorry

end monotonic_intervals_value_of_a_inequality_a_minus_one_l293_293085


namespace base_area_cone_l293_293283

theorem base_area_cone (V h : ℝ) (s_cylinder s_cone : ℝ) 
  (cylinder_volume : V = s_cylinder * h) 
  (cone_volume : V = (1 / 3) * s_cone * h) 
  (s_cylinder_val : s_cylinder = 15) : s_cone = 45 := 
by 
  sorry

end base_area_cone_l293_293283


namespace football_goals_l293_293430

theorem football_goals :
  (exists A B C : ℕ,
    (A = 3 ∧ B ≠ 1 ∧ (C = 5 ∧ V = 6 ∧ A ≠ 2 ∧ V = 5)) ∨
    (A ≠ 3 ∧ B = 1 ∧ (C ≠ 5 ∧ V = 6 ∧ A = 2 ∧ V ≠ 5))) →
  A + B + C ≠ 10 :=
by {
  sorry
}

end football_goals_l293_293430


namespace price_difference_is_correct_l293_293950

-- Definitions from the problem conditions
def list_price : ℝ := 58.80
def tech_shop_discount : ℝ := 12.00
def value_mart_discount_rate : ℝ := 0.20

-- Calculating the sale prices from definitions
def tech_shop_sale_price : ℝ := list_price - tech_shop_discount
def value_mart_sale_price : ℝ := list_price * (1 - value_mart_discount_rate)

-- The proof problem statement
theorem price_difference_is_correct :
  value_mart_sale_price - tech_shop_sale_price = 0.24 :=
by
  sorry

end price_difference_is_correct_l293_293950


namespace cos_180_degree_l293_293608

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end cos_180_degree_l293_293608


namespace solve_B_l293_293260

theorem solve_B (B : ℕ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 7 ∣ (4000 + 110 * B + 2)) : B = 4 :=
by
  sorry

end solve_B_l293_293260


namespace rose_needs_more_money_l293_293515

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end rose_needs_more_money_l293_293515


namespace maple_logs_correct_l293_293343

/-- Each pine tree makes 80 logs. -/
def pine_logs := 80

/-- Each walnut tree makes 100 logs. -/
def walnut_logs := 100

/-- Jerry cuts up 8 pine trees. -/
def pine_trees := 8

/-- Jerry cuts up 3 maple trees. -/
def maple_trees := 3

/-- Jerry cuts up 4 walnut trees. -/
def walnut_trees := 4

/-- The total number of logs is 1220. -/
def total_logs := 1220

/-- The number of logs each maple tree makes. -/
def maple_logs := 60

theorem maple_logs_correct :
  (pine_trees * pine_logs) + (maple_trees * maple_logs) + (walnut_trees * walnut_logs) = total_logs :=
by
  -- (8 * 80) + (3 * 60) + (4 * 100) = 1220
  sorry

end maple_logs_correct_l293_293343


namespace find_m_value_l293_293654

theorem find_m_value (x y m : ℤ) (h₁ : x = 2) (h₂ : y = -3) (h₃ : 5 * x + m * y + 2 = 0) : m = 4 := 
by 
  sorry

end find_m_value_l293_293654


namespace find_q_l293_293075

variable {a d q : ℝ}
variables (M N : Set ℝ)

theorem find_q (hM : M = {a, a + d, a + 2 * d}) 
              (hN : N = {a, a * q, a * q^2})
              (ha : a ≠ 0)
              (heq : M = N) :
  q = -1 / 2 :=
sorry

end find_q_l293_293075


namespace least_number_to_add_l293_293549

theorem least_number_to_add (n : ℕ) : (3457 + n) % 103 = 0 ↔ n = 45 :=
by sorry

end least_number_to_add_l293_293549


namespace sqrt_twentyfive_eq_five_l293_293554

theorem sqrt_twentyfive_eq_five : Real.sqrt 25 = 5 := by
  sorry

end sqrt_twentyfive_eq_five_l293_293554


namespace find_y_minus_x_l293_293501

theorem find_y_minus_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : Real.sqrt x + Real.sqrt y = 1) 
  (h5 : Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3) : 
  y - x = 1 / 2 :=
sorry

end find_y_minus_x_l293_293501


namespace pencil_eraser_cost_l293_293845

theorem pencil_eraser_cost (p e : ℕ) (hp : p > e) (he : e > 0)
  (h : 20 * p + 4 * e = 160) : p + e = 12 :=
sorry

end pencil_eraser_cost_l293_293845


namespace maximum_value_of_a_l293_293481

theorem maximum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) ↔ a ≤ 5 :=
by
  sorry

end maximum_value_of_a_l293_293481


namespace decagon_intersection_points_l293_293924

theorem decagon_intersection_points : 
  let n := 10 in 
  ∑ i in (Fintype.elems (Finset.range (n+1)).image (λ k, (Finset.card (Finset.choose 4 (Finset.range n)) i)), 
    i = 210 := 
begin
  sorry
end

end decagon_intersection_points_l293_293924


namespace cancel_terms_valid_equation_l293_293081

theorem cancel_terms_valid_equation {m n : ℕ} 
  (x : Fin n → ℕ) (y : Fin m → ℕ) 
  (h_sum_eq : (Finset.univ.sum x) = (Finset.univ.sum y))
  (h_sum_lt : (Finset.univ.sum x) < (m * n)) : 
  ∃ x' : Fin n → ℕ, ∃ y' : Fin m → ℕ, 
    (Finset.univ.sum x' = Finset.univ.sum y') ∧ x' ≠ x ∧ y' ≠ y :=
sorry

end cancel_terms_valid_equation_l293_293081


namespace mrs_hilt_candy_l293_293506

theorem mrs_hilt_candy : 2 * 9 + 3 * 9 + 1 * 9 = 54 :=
by
  sorry

end mrs_hilt_candy_l293_293506


namespace original_amount_spent_l293_293122

noncomputable def price_per_mango : ℝ := 383.33 / 115
noncomputable def new_price_per_mango : ℝ := 0.9 * price_per_mango

theorem original_amount_spent (N : ℝ) (H1 : (N + 12) * new_price_per_mango = N * price_per_mango) : 
  N * price_per_mango = 359.64 :=
by 
  sorry

end original_amount_spent_l293_293122


namespace valentine_giveaway_l293_293849

theorem valentine_giveaway (initial : ℕ) (left : ℕ) (given : ℕ) (h1 : initial = 30) (h2 : left = 22) : given = initial - left → given = 8 :=
by
  sorry

end valentine_giveaway_l293_293849


namespace three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l293_293858

-- Problem (1)
theorem three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty : 3^40 > 4^30 ∧ 4^30 > 5^20 := 
by
  sorry

-- Problem (2)
theorem sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one : 16^31 > 8^41 ∧ 8^41 > 4^61 :=
by 
  sorry

-- Problem (3)
theorem a_lt_b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a^5 = 2) (h4 : b^7 = 3) : a < b :=
by
  sorry

end three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l293_293858


namespace hillary_sunday_spend_l293_293089

noncomputable def spend_per_sunday (total_spent : ℕ) (weeks : ℕ) (weekday_price : ℕ) (weekday_papers : ℕ) : ℕ :=
  (total_spent - weeks * weekday_papers * weekday_price) / weeks

theorem hillary_sunday_spend :
  spend_per_sunday 2800 8 50 3 = 200 :=
sorry

end hillary_sunday_spend_l293_293089


namespace race_distance_A_beats_C_l293_293759

variables (race_distance1 race_distance2 race_distance3 : ℕ)
           (distance_AB distance_BC distance_AC : ℕ)

theorem race_distance_A_beats_C :
  race_distance1 = 500 →
  race_distance2 = 500 →
  distance_AB = 50 →
  distance_BC = 25 →
  distance_AC = 58 →
  race_distance3 = 400 :=
by
  sorry

end race_distance_A_beats_C_l293_293759


namespace min_value_fraction_l293_293326

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) : 
  (1 / x) + (1 / (3 * y)) ≥ 3 :=
sorry

end min_value_fraction_l293_293326


namespace john_took_away_oranges_l293_293244

-- Define the initial number of oranges Melissa had.
def initial_oranges : ℕ := 70

-- Define the number of oranges Melissa has left.
def oranges_left : ℕ := 51

-- Define the expected number of oranges John took away.
def oranges_taken : ℕ := 19

-- The theorem that needs to be proven.
theorem john_took_away_oranges :
  initial_oranges - oranges_left = oranges_taken :=
by
  sorry

end john_took_away_oranges_l293_293244


namespace example_theorem_l293_293061

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l293_293061


namespace runway_show_time_l293_293255

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end runway_show_time_l293_293255


namespace juice_cost_l293_293688

-- Given conditions
def sandwich_cost : ℝ := 0.30
def total_money : ℝ := 2.50
def num_friends : ℕ := 4

-- Cost calculation
def total_sandwich_cost : ℝ := num_friends * sandwich_cost
def remaining_money : ℝ := total_money - total_sandwich_cost

-- The theorem to prove
theorem juice_cost : (remaining_money / num_friends) = 0.325 := by
  sorry

end juice_cost_l293_293688


namespace find_denominator_l293_293331

-- Define the conditions given in the problem
variables (p q : ℚ)
variable (denominator : ℚ)

-- Assuming the conditions
variables (h1 : p / q = 4 / 5)
variables (h2 : 11 / 7 + (2 * q - p) / denominator = 2)

-- State the theorem we want to prove
theorem find_denominator : denominator = 14 :=
by
  -- The proof will be constructed later
  sorry

end find_denominator_l293_293331


namespace math_problem_l293_293213

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end math_problem_l293_293213


namespace cos_180_eq_neg1_l293_293601

-- Conditions
def cosine_in_unit_circle (θ : ℝ) : ℝ :=
  let (x, y) := Real.angle θ in x

-- Objective
theorem cos_180_eq_neg1 : cosine_in_unit_circle (Real.pi) = -1 := by
  sorry


end cos_180_eq_neg1_l293_293601


namespace ratio_of_Jordyn_age_to_Zrinka_age_is_2_l293_293846

variable (Mehki_age : ℕ) (Jordyn_age : ℕ) (Zrinka_age : ℕ)

-- Conditions
def Mehki_is_10_years_older_than_Jordyn := Mehki_age = Jordyn_age + 10
def Zrinka_age_is_6 := Zrinka_age = 6
def Mehki_age_is_22 := Mehki_age = 22

-- Theorem statement: the ratio of Jordyn's age to Zrinka's age is 2.
theorem ratio_of_Jordyn_age_to_Zrinka_age_is_2
  (h1 : Mehki_is_10_years_older_than_Jordyn Mehki_age Jordyn_age)
  (h2 : Zrinka_age_is_6 Zrinka_age)
  (h3 : Mehki_age_is_22 Mehki_age) : Jordyn_age / Zrinka_age = 2 :=
by
  -- The proof would go here
  sorry

end ratio_of_Jordyn_age_to_Zrinka_age_is_2_l293_293846


namespace committee_formation_l293_293560

/-- Problem statement: In how many ways can a 5-person executive committee be formed if one of the 
members must be the president, given there are 30 members. --/
theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 30) (h2 : k = 5) : 
  (n * Nat.choose (n - 1) (k - 1) = 712530 ) :=
by
  sorry

end committee_formation_l293_293560


namespace minimum_value_l293_293653

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y / x = 1) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = (1 / x + x / y) → z ≥ m :=
sorry

end minimum_value_l293_293653


namespace eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l293_293165

-- Proof 1: Line through intersection and parallel
theorem eqn_of_line_through_intersection_parallel :
  ∃ k : ℝ, (9 : ℝ) * (x: ℝ) + (18: ℝ) * (y: ℝ) - 4 = 0 ∧
           (∀ x y : ℝ, (2 * x + 3 * y - 5 = 0) → (7 * x + 15 * y + 1 = 0) → (x + 2 * y + k = 0)) :=
sorry

-- Proof 2: Line perpendicular and specific distance from origin
theorem eqn_of_line_perpendicular_distance :
  ∃ k : ℝ, (∃ m : ℝ, (k = 30 ∨ k = -30) ∧ (4 * (x: ℝ) - 3 * (y: ℝ) + m = 0 ∧ (∃ d : ℝ, d = 6 ∧ (|m| / (4 ^ 2 + (-3) ^ 2).sqrt) = d))) :=
sorry

end eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l293_293165


namespace find_other_diagonal_l293_293999

noncomputable def convex_quadrilateral (A B C D : ℝ) (area : ℝ) (sum_of_sides_diagonal : ℝ) (other_diagonal : ℝ) : Prop :=
  A + B + D = sum_of_sides_diagonal ∧ 
  (∃ S S' : ℝ, S + S' = area ∧ S = 1/2 * A * D * sin (0.5) ∧
   S' = 1/2 * C * D * sin (0.5) ∧ 32 ≤ 1/2 * (A + C) * D ∧ 
  A + C = D ∧ 2 * D = sum_of_sides_diagonal → 
  other_diagonal = sqrt 2 * D → other_diagonal = 8 * sqrt 2)

theorem find_other_diagonal :
  ∀ (A B C D other_diagonal : ℝ),
  convex_quadrilateral A B C D 32 (A + B + D) other_diagonal → other_diagonal = 8 * sqrt 2 :=
begin
  intros A B C D other_diagonal h,
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩,
  sorry
end

end find_other_diagonal_l293_293999


namespace find_polynomial_parameters_and_minimum_value_l293_293976

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_polynomial_parameters_and_minimum_value 
  (a b c : ℝ)
  (h1 : f (-1) a b c = 7)
  (h2 : 3 * (-1)^2 + 2 * a * (-1) + b = 0)
  (h3 : 3 * 3^2 + 2 * a * 3 + b = 0)
  (h4 : a = -3)
  (h5 : b = -9)
  (h6 : c = 2) :
  f 3 (-3) (-9) 2 = -25 :=
by
  sorry

end find_polynomial_parameters_and_minimum_value_l293_293976


namespace josh_remaining_marbles_l293_293680

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7
def remaining_marbles : ℕ := 9

theorem josh_remaining_marbles : initial_marbles - lost_marbles = remaining_marbles := by
  sorry

end josh_remaining_marbles_l293_293680


namespace possible_values_of_sum_l293_293408

theorem possible_values_of_sum
  (p q r : ℝ)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_system : q = p * (4 - p) ∧ r = q * (4 - q) ∧ p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end possible_values_of_sum_l293_293408


namespace x_intercept_is_correct_l293_293009

-- Define the original line equation
def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

-- Define the perpendicular line's y-intercept
def y_intercept (y : ℝ) : Prop := y = -3

-- Define the equation of the perpendicular line in slope-intercept form
def perpendicular_line (x y : ℝ) : Prop := y = (5 / 4) * x + -3

-- Prove that the x-intercept of the perpendicular line is 12/5
theorem x_intercept_is_correct : ∃ x : ℝ, x ≠ 0 ∧ (∃ y : ℝ, y = 0) ∧ (perpendicular_line x y) :=
sorry

end x_intercept_is_correct_l293_293009


namespace smallest_integer_x_divisibility_l293_293750

theorem smallest_integer_x_divisibility :
  ∃ x : ℤ, (2 * x + 2) % 33 = 0 ∧ (2 * x + 2) % 44 = 0 ∧ (2 * x + 2) % 55 = 0 ∧ (2 * x + 2) % 666 = 0 ∧ x = 36629 := 
sorry

end smallest_integer_x_divisibility_l293_293750


namespace total_teaching_hours_l293_293796

-- Define the durations of the classes
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2

def math_class_duration : ℕ := 1
def science_class_duration : ℚ := 1.5
def history_class_duration : ℕ := 2

-- Define Eduardo's teaching time
def eduardo_total_time : ℚ :=
  eduardo_math_classes * math_class_duration +
  eduardo_science_classes * science_class_duration +
  eduardo_history_classes * history_class_duration

-- Define Frankie's teaching time (double the classes of Eduardo)
def frankie_total_time : ℚ :=
  2 * (eduardo_math_classes * math_class_duration) +
  2 * (eduardo_science_classes * science_class_duration) +
  2 * (eduardo_history_classes * history_class_duration)

-- Define the total teaching time for both Eduardo and Frankie
def total_teaching_time : ℚ :=
  eduardo_total_time + frankie_total_time

-- Theorem statement that both their total teaching time is 39 hours
theorem total_teaching_hours : total_teaching_time = 39 :=
by
  -- skipping the proof using sorry
  sorry

end total_teaching_hours_l293_293796


namespace boxes_needed_l293_293003

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l293_293003


namespace hyperbola_focus_coordinates_l293_293308

open Real

theorem hyperbola_focus_coordinates :
  ∃ x y : ℝ, (2 * x^2 - y^2 + 8 * x + 4 * y - 28 = 0) ∧
           ((x = -2 - 4 * sqrt 3 ∧ y = 2) ∨ (x = -2 + 4 * sqrt 3 ∧ y = 2)) := by sorry

end hyperbola_focus_coordinates_l293_293308


namespace triangle_angle_ratio_l293_293482

theorem triangle_angle_ratio (a b c : ℝ) (h₁ : a + b + c = 180)
  (h₂ : b = 2 * a) (h₃ : c = 3 * a) : a = 30 ∧ b = 60 ∧ c = 90 :=
by
  sorry

end triangle_angle_ratio_l293_293482


namespace smallest_rel_prime_210_l293_293805

theorem smallest_rel_prime_210 : ∃ (y : ℕ), y > 1 ∧ Nat.gcd y 210 = 1 ∧ (∀ z : ℕ, z > 1 ∧ Nat.gcd z 210 = 1 → y ≤ z) ∧ y = 11 :=
by {
  sorry -- proof to be filled in
}

end smallest_rel_prime_210_l293_293805


namespace problem_l293_293812

theorem problem (x : ℝ) : (x^2 + 2 * x - 3 ≤ 0) → ¬(abs x > 3) :=
by sorry

end problem_l293_293812


namespace tarantulas_per_egg_sac_l293_293479

-- Condition: Each tarantula has 8 legs
def legs_per_tarantula : ℕ := 8

-- Condition: There are 32000 baby tarantula legs
def total_legs : ℕ := 32000

-- Condition: Number of egg sacs is one less than 5
def number_of_egg_sacs : ℕ := 5 - 1

-- Calculated: Number of tarantulas in total
def total_tarantulas : ℕ := total_legs / legs_per_tarantula

-- Proof Statement: Number of tarantulas per egg sac
theorem tarantulas_per_egg_sac : total_tarantulas / number_of_egg_sacs = 1000 := by
  sorry

end tarantulas_per_egg_sac_l293_293479


namespace find_n_for_quadratic_l293_293450

theorem find_n_for_quadratic (a b c m n p : ℕ) (h1 : a = 3) (h2 : b = -7) (h3 : c = 1)
  (h_eq : 3 * m + 7 * n + c = 0)
  (h_gcd : Int.gcd (Int.natAbs m) (Int.gcd (Int.natAbs n) (Int.natAbs p)) = 1) :
  n = 37 :=
by
  -- We assume the values for the purpose of building a valid theorem statement
  have ha : a = 3 := h1
  have hb : b = -7 := h2
  have hc : c = 1 := h3
    
  -- The standard form of the quadratic equation roots
  have h_roots : ∀ x, 3 * x * x - 7 * x + 1 = 0 ↔ x = (7 + ⟩ (37 : ℕ))
    := sorry
    
  -- Conclusion based on the shape of the roots and gcd condition
  have h_gcd' : Int.gcd 7 (Int.gcd 37 6) = 1 := sorry

  -- Finally, deduce the value of n
  exact eq.trans h_eq h_roots

end find_n_for_quadratic_l293_293450


namespace fraction_of_quarters_l293_293798

-- Conditions as definitions
def total_quarters : ℕ := 30
def states_between_1790_1809 : ℕ := 18

-- Goal theorem to prove 
theorem fraction_of_quarters : (states_between_1790_1809 : ℚ) / (total_quarters : ℚ) = 3 / 5 :=
by 
  sorry

end fraction_of_quarters_l293_293798


namespace first_shuffle_correct_l293_293030

def initial_order : Fin 13 → ℕ
| 0 := 2 | 1 := 3 | 2 := 4 | 3 := 5
| 4 := 6 | 5 := 7 | 6 := 8 | 7 := 9
| 8 := 10 | 9 := 11 | 10 := 12 | 11 := 13 | 12 := 14

def final_order : Fin 13 → ℕ
| 0 := 11 | 1 := 10 | 2 := 13 | 3 := 9
| 4 := 14 | 5 := 4 | 6 := 5 | 7 := 2
| 8 := 6 | 9 := 12 | 10 := 7 | 11 := 3 | 12 := 8

def expected_first_shuffle : Fin 13 → ℕ
| 0 := 10 | 1 := 2 | 2 := 5 | 3 := 13
| 4 := 12 | 5 := 8 | 6 := 4 | 7 := 3
| 8 := 11 | 9 := 6 | 10 := 14 | 11 := 9 | 12 := 7

theorem first_shuffle_correct :
  ∃ (π : Equiv.Perm (Fin 13)), 
    (∀ i : Fin 13, π (π (initial_order i)) = final_order i) ∧ 
    (∀ i : Fin 13, π (initial_order i) = expected_first_shuffle i) :=
sorry

end first_shuffle_correct_l293_293030


namespace nine_pow_2048_mod_50_l293_293749

theorem nine_pow_2048_mod_50 : (9^2048) % 50 = 21 := sorry

end nine_pow_2048_mod_50_l293_293749


namespace meaningful_sqrt_range_l293_293993

theorem meaningful_sqrt_range (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 :=
by
  sorry

end meaningful_sqrt_range_l293_293993


namespace seeds_germinated_percentage_l293_293161

theorem seeds_germinated_percentage (n1 n2 : ℕ) (p1 p2 : ℝ) (h1 : n1 = 300) (h2 : n2 = 200) (h3 : p1 = 0.25) (h4 : p2 = 0.30) :
  ( (n1 * p1 + n2 * p2) / (n1 + n2) ) * 100 = 27 :=
by
  sorry

end seeds_germinated_percentage_l293_293161


namespace point_A_in_second_quadrant_l293_293835

-- Define the conditions as functions
def x_coord : ℝ := -3
def y_coord : ℝ := 4

-- Define a set of quadrants for clarity
inductive Quadrant
| first
| second
| third
| fourth

-- Prove that if the point has specific coordinates, it lies in the specific quadrant
def point_in_quadrant (x: ℝ) (y: ℝ) : Quadrant :=
  if x < 0 ∧ y > 0 then Quadrant.second
  else if x > 0 ∧ y > 0 then Quadrant.first
  else if x < 0 ∧ y < 0 then Quadrant.third
  else Quadrant.fourth

-- The statement to prove:
theorem point_A_in_second_quadrant : point_in_quadrant x_coord y_coord = Quadrant.second :=
by 
  -- Proof would go here but is not required per instructions
  sorry

end point_A_in_second_quadrant_l293_293835


namespace year_weeks_span_l293_293070

theorem year_weeks_span (days_in_year : ℕ) (h1 : days_in_year = 365 ∨ days_in_year = 366) :
  ∃ W : ℕ, (W = 53 ∨ W = 54) ∧ (days_in_year = 365 → W = 53) ∧ (days_in_year = 366 → W = 53 ∨ W = 54) :=
by
  sorry

end year_weeks_span_l293_293070


namespace inequality_c_l293_293577

theorem inequality_c (x : ℝ) : x^2 + 1 + 1 / (x^2 + 1) ≥ 2 := sorry

end inequality_c_l293_293577


namespace side_ratio_triangle_square_pentagon_l293_293297

-- Define the conditions
def perimeter_triangle (t : ℝ) := 3 * t = 18
def perimeter_square (s : ℝ) := 4 * s = 16
def perimeter_pentagon (p : ℝ) := 5 * p = 20

-- Statement to be proved
theorem side_ratio_triangle_square_pentagon 
  (t s p : ℝ)
  (ht : perimeter_triangle t)
  (hs : perimeter_square s)
  (hp : perimeter_pentagon p) : 
  (t / s = 3 / 2) ∧ (t / p = 3 / 2) := 
sorry

end side_ratio_triangle_square_pentagon_l293_293297


namespace min_value_x_squared_plus_6x_l293_293892

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l293_293892


namespace car_speed_first_hour_l293_293737

theorem car_speed_first_hour (x : ℝ) (h : (79 = (x + 60) / 2)) : x = 98 :=
by {
  sorry
}

end car_speed_first_hour_l293_293737


namespace maya_total_pages_l293_293505

def books_first_week : ℕ := 5
def pages_per_book_first_week : ℕ := 300
def books_second_week := books_first_week * 2
def pages_per_book_second_week : ℕ := 350
def books_third_week := books_first_week * 3
def pages_per_book_third_week : ℕ := 400

def total_pages_first_week : ℕ := books_first_week * pages_per_book_first_week
def total_pages_second_week : ℕ := books_second_week * pages_per_book_second_week
def total_pages_third_week : ℕ := books_third_week * pages_per_book_third_week

def total_pages_maya_read : ℕ := total_pages_first_week + total_pages_second_week + total_pages_third_week

theorem maya_total_pages : total_pages_maya_read = 11000 := by
  sorry

end maya_total_pages_l293_293505


namespace p_is_necessary_but_not_sufficient_for_q_l293_293970

variable (x : ℝ)

def p : Prop := -1 ≤ x ∧ x ≤ 5
def q : Prop := (x - 5) * (x + 1) < 0

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) := 
sorry

end p_is_necessary_but_not_sufficient_for_q_l293_293970


namespace probability_greater_difficulty_probability_same_difficulty_l293_293997

/-- A datatype representing the difficulty levels of questions. -/
inductive Difficulty
| easy : Difficulty
| medium : Difficulty
| difficult : Difficulty

/-- A datatype representing the four questions with their difficulties. -/
inductive Question
| A1 : Question
| A2 : Question
| B : Question
| C : Question

/-- The function to get the difficulty of a question. -/
def difficulty (q : Question) : Difficulty :=
  match q with
  | Question.A1 => Difficulty.easy
  | Question.A2 => Difficulty.easy
  | Question.B  => Difficulty.medium
  | Question.C  => Difficulty.difficult

/-- The set of all possible pairings of questions selected by two students A and B. -/
def all_pairs : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A1, Question.B), (Question.A1, Question.C),
    (Question.A2, Question.A1), (Question.A2, Question.A2), (Question.A2, Question.B), (Question.A2, Question.C),
    (Question.B, Question.A1), (Question.B, Question.A2), (Question.B, Question.B), (Question.B, Question.C),
    (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B), (Question.C, Question.C) ]

/-- The event that the difficulty of the question selected by student A is greater than that selected by student B. -/
def event_N : List (Question × Question) :=
  [ (Question.B, Question.A1), (Question.B, Question.A2), (Question.C, Question.A1), (Question.C, Question.A2), (Question.C, Question.B) ]

/-- The event that the difficulties of the questions selected by both students are the same. -/
def event_M : List (Question × Question) :=
  [ (Question.A1, Question.A1), (Question.A1, Question.A2), (Question.A2, Question.A1), (Question.A2, Question.A2), 
    (Question.B, Question.B), (Question.C, Question.C) ]

/-- The probabilities of the events. -/
noncomputable def probability_event_N : ℚ := (event_N.length : ℚ) / (all_pairs.length : ℚ)
noncomputable def probability_event_M : ℚ := (event_M.length : ℚ) / (all_pairs.length : ℚ)

/-- The theorem statements -/
theorem probability_greater_difficulty : probability_event_N = 5 / 16 := sorry
theorem probability_same_difficulty : probability_event_M = 3 / 8 := sorry

end probability_greater_difficulty_probability_same_difficulty_l293_293997


namespace exists_saddle_point_probability_l293_293296

noncomputable def saddle_point_probability := (3 : ℝ) / 10

theorem exists_saddle_point_probability {A : ℕ → ℕ → ℝ}
  (h : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1 ∧ (∀ k l, (i ≠ k ∨ j ≠ l) → A i j ≠ A k l)) :
  (∃ (p : ℝ), p = saddle_point_probability) :=
by 
  sorry

end exists_saddle_point_probability_l293_293296


namespace find_quadratic_polynomial_l293_293958

def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_quadratic_polynomial : 
  ∃ a b c: ℝ, (∀ x : ℂ, quadratic_polynomial a b c x.re = 0 → (x = 3 + 4*I) ∨ (x = 3 - 4*I)) 
  ∧ (b = 8) 
  ∧ (a = -4/3) 
  ∧ (c = -50/3) :=
by
  sorry

end find_quadratic_polynomial_l293_293958


namespace chord_equation_l293_293658

variable {x y k b : ℝ}

-- Define the condition of the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 - 4 = 0

-- Define the condition that the point M(1, 1) is the midpoint
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- Define the line equation in terms of its slope k and y-intercept b
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

theorem chord_equation :
  (∃ (x₁ x₂ y₁ y₂ : ℝ), ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ midpoint_condition x₁ y₁ x₂ y₂) →
  (∃ (k b : ℝ), line k b x y ∧ k + b = 1 ∧ b = 1 - k) →
  y = -0.5 * x + 1.5 ↔ x + 2 * y - 3 = 0 :=
by
  sorry

end chord_equation_l293_293658


namespace overall_average_commission_rate_l293_293253

-- Define conditions for the commissions and transaction amounts
def C₁ := 0.25 / 100 * 100 + 0.25 / 100 * 105.25
def C₂ := 0.35 / 100 * 150 + 0.45 / 100 * 155.50
def C₃ := 0.30 / 100 * 80 + 0.40 / 100 * 83
def total_commission := C₁ + C₂ + C₃
def TA := 100 + 105.25 + 150 + 155.50 + 80 + 83

-- The proposition to prove
theorem overall_average_commission_rate : (total_commission / TA) * 100 = 0.3429 :=
  by
  sorry

end overall_average_commission_rate_l293_293253


namespace positive_difference_between_two_numbers_l293_293143

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l293_293143


namespace total_students_accommodated_l293_293573

structure BusConfig where
  columns : ℕ
  rows : ℕ
  broken_seats : ℕ

structure SplitBusConfig where
  columns : ℕ
  left_rows : ℕ
  right_rows : ℕ
  broken_seats : ℕ

structure ComplexBusConfig where
  columns : ℕ
  rows : ℕ
  special_rows_broken_seats : ℕ

def bus1 : BusConfig := { columns := 4, rows := 10, broken_seats := 2 }
def bus2 : BusConfig := { columns := 5, rows := 8, broken_seats := 4 }
def bus3 : BusConfig := { columns := 3, rows := 12, broken_seats := 3 }
def bus4 : SplitBusConfig := { columns := 4, left_rows := 6, right_rows := 8, broken_seats := 1 }
def bus5 : SplitBusConfig := { columns := 6, left_rows := 8, right_rows := 10, broken_seats := 5 }
def bus6 : ComplexBusConfig := { columns := 5, rows := 10, special_rows_broken_seats := 4 }

theorem total_students_accommodated :
  let seats_bus1 := (bus1.columns * bus1.rows) - bus1.broken_seats;
  let seats_bus2 := (bus2.columns * bus2.rows) - bus2.broken_seats;
  let seats_bus3 := (bus3.columns * bus3.rows) - bus3.broken_seats;
  let seats_bus4 := (bus4.columns * bus4.left_rows) + (bus4.columns * bus4.right_rows) - bus4.broken_seats;
  let seats_bus5 := (bus5.columns * bus5.left_rows) + (bus5.columns * bus5.right_rows) - bus5.broken_seats;
  let seats_bus6 := (bus6.columns * bus6.rows) - bus6.special_rows_broken_seats;
  seats_bus1 + seats_bus2 + seats_bus3 + seats_bus4 + seats_bus5 + seats_bus6 = 311 :=
sorry

end total_students_accommodated_l293_293573


namespace smallest_integer_solution_l293_293157

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end smallest_integer_solution_l293_293157


namespace axis_of_symmetry_of_function_l293_293149

theorem axis_of_symmetry_of_function 
  (f : ℝ → ℝ)
  (h : ∀ x, f x = 3 * Real.cos x - Real.sqrt 3 * Real.sin x)
  : ∃ k : ℤ, x = k * Real.pi - Real.pi / 6 ∧ x = Real.pi - Real.pi / 6 :=
sorry

end axis_of_symmetry_of_function_l293_293149


namespace right_triangle_angles_ratio_l293_293335

theorem right_triangle_angles_ratio (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3) :
  α = 67.5 ∧ β = 22.5 :=
sorry

end right_triangle_angles_ratio_l293_293335


namespace inequalities_indeterminate_l293_293095

variable (s x y z : ℝ)

theorem inequalities_indeterminate (h_s : s > 0) (h_ineq : s * x > z * y) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (¬ (x > z)) ∨ (¬ (-x > -z)) ∨ (¬ (s > z / x)) ∨ (¬ (s < y / x)) :=
by sorry

end inequalities_indeterminate_l293_293095


namespace positive_difference_between_two_numbers_l293_293144

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end positive_difference_between_two_numbers_l293_293144


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l293_293239

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l293_293239


namespace photo_album_slots_l293_293788

def photos_from_cristina : Nat := 7
def photos_from_john : Nat := 10
def photos_from_sarah : Nat := 9
def photos_from_clarissa : Nat := 14

theorem photo_album_slots :
  photos_from_cristina + photos_from_john + photos_from_sarah + photos_from_clarissa = 40 :=
by
  sorry

end photo_album_slots_l293_293788


namespace fan_airflow_weekly_l293_293763

def fan_airflow_per_second : ℕ := 10
def fan_work_minutes_per_day : ℕ := 10
def minutes_to_seconds (m : ℕ) : ℕ := m * 60
def days_per_week : ℕ := 7

theorem fan_airflow_weekly : 
  (fan_airflow_per_second * (minutes_to_seconds fan_work_minutes_per_day) * days_per_week) = 42000 := 
by
  sorry

end fan_airflow_weekly_l293_293763


namespace f_odd_f_increasing_on_2_infty_solve_inequality_f_l293_293462

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = -f x := by
  sorry

theorem f_increasing_on_2_infty (x₁ x₂ : ℝ) (hx₁ : 2 < x₁) (hx₂ : 2 < x₂) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

theorem solve_inequality_f (x : ℝ) (hx : -5 < x ∧ x < -1) : f (2*x^2 + 5*x + 8) + f (x - 3 - x^2) < 0 := by
  sorry

end f_odd_f_increasing_on_2_infty_solve_inequality_f_l293_293462


namespace pointA_in_second_quadrant_l293_293838

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l293_293838


namespace perfect_square_mod_3_l293_293855

theorem perfect_square_mod_3 (k : ℤ) (hk : ∃ m : ℤ, k = m^2) : k % 3 = 0 ∨ k % 3 = 1 :=
by
  sorry

end perfect_square_mod_3_l293_293855


namespace correct_tourism_model_l293_293219

noncomputable def tourism_model (x : ℕ) : ℝ :=
  80 * (Real.cos ((Real.pi / 6) * x + (2 * Real.pi / 3))) + 120

theorem correct_tourism_model :
  (∀ n : ℕ, tourism_model (n + 12) = tourism_model n) ∧
  (tourism_model 8 - tourism_model 2 = 160) ∧
  (tourism_model 2 = 40) :=
by
  sorry

end correct_tourism_model_l293_293219


namespace sin_minus_cos_eq_sqrt2_l293_293064

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l293_293064


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l293_293241

-- Definitions and statements for part (a)
def is_palindromic (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Provide an example number for part (a)
theorem example_palindromic_divisible_by_5 : ∃ n, 
  five_digit_number n ∧ is_palindromic n ∧ divisible_by_5 n :=
  ⟨51715, by sorry⟩

-- Definitions and statements for part (b)
def num_five_digit_palindromic_divisible_by_5 : ℕ :=
  100

theorem count_palindromic_divisible_by_5 : 
  num_five_digit_palindromic_divisible_by_5 = 100 :=
  by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l293_293241


namespace arithmetic_sequence_terms_count_l293_293677

theorem arithmetic_sequence_terms_count :
  ∃ n : ℕ, ∀ a d l, 
    a = 13 → 
    d = 3 → 
    l = 73 → 
    l = a + (n - 1) * d ∧ n = 21 :=
by
  sorry

end arithmetic_sequence_terms_count_l293_293677


namespace relatively_prime_subsequence_exists_l293_293856

theorem relatively_prime_subsequence_exists :
  ∃ (s : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → Nat.gcd (2^(s i) - 3) (2^(s j) - 3) = 1) :=
by
  sorry

end relatively_prime_subsequence_exists_l293_293856


namespace molecular_weight_of_6_moles_Al2_CO3_3_l293_293748

noncomputable def molecular_weight_Al2_CO3_3: ℝ :=
  let Al_weight := 26.98
  let C_weight := 12.01
  let O_weight := 16.00
  let CO3_weight := C_weight + 3 * O_weight
  let one_mole_weight := 2 * Al_weight + 3 * CO3_weight
  6 * one_mole_weight

theorem molecular_weight_of_6_moles_Al2_CO3_3 : 
  molecular_weight_Al2_CO3_3 = 1403.94 :=
by
  sorry

end molecular_weight_of_6_moles_Al2_CO3_3_l293_293748


namespace sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l293_293251

-- Definitions
def a (t : ℤ) := 4 * t
def b (t : ℤ) := 3 - 2 * t - t^2
def c (t : ℤ) := 3 + 2 * t - t^2

-- Theorem for sum of squares
theorem sum_of_squares_twice_square (t : ℤ) : 
  a t ^ 2 + b t ^ 2 + c t ^ 2 = 2 * ((3 + t^2) ^ 2) :=
by 
  sorry

-- Theorem for sum of fourth powers
theorem sum_of_fourth_powers_twice_fourth_power (t : ℤ) : 
  a t ^ 4 + b t ^ 4 + c t ^ 4 = 2 * ((3 + t^2) ^ 4) :=
by 
  sorry

end sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l293_293251


namespace solve_abs_equation_l293_293139

theorem solve_abs_equation (x : ℝ) : 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 :=
by
  sorry

end solve_abs_equation_l293_293139


namespace at_least_one_non_negative_l293_293454

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 := 
sorry

end at_least_one_non_negative_l293_293454


namespace convex_polygon_diagonals_25_convex_polygon_triangles_25_l293_293167

-- Define a convex polygon with 25 sides
def convex_polygon_sides : ℕ := 25

-- Define the number of diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the number of triangles that can be formed by choosing any three vertices from n vertices
def number_of_triangles (n : ℕ) : ℕ := n.choose 3

-- Theorem to prove the number of diagonals is 275 for a convex polygon with 25 sides
theorem convex_polygon_diagonals_25 : number_of_diagonals convex_polygon_sides = 275 :=
by sorry

-- Theorem to prove the number of triangles is 2300 for a convex polygon with 25 sides
theorem convex_polygon_triangles_25 : number_of_triangles convex_polygon_sides = 2300 :=
by sorry

end convex_polygon_diagonals_25_convex_polygon_triangles_25_l293_293167


namespace sum_of_reciprocal_squares_of_roots_l293_293259

theorem sum_of_reciprocal_squares_of_roots (a b c : ℝ) 
    (h_roots : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c) :
    a + b + c = 6 ∧ ab + bc + ca = 11 ∧ abc = 6 → 
    (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 := 
by
  sorry

end sum_of_reciprocal_squares_of_roots_l293_293259


namespace x_intercept_of_perpendicular_line_l293_293010

noncomputable def x_intercept_perpendicular (m₁ m₂ : ℚ) : ℚ :=
  let m_perpendicular := -1 / m₁ in
  let b := -3 in
  -b / m_perpendicular

theorem x_intercept_of_perpendicular_line :
  (4 * x_intercept_perpendicular (-4/5) (5/4) + 5 * 0) = 10 :=
by
  sorry

end x_intercept_of_perpendicular_line_l293_293010


namespace cos_180_degrees_l293_293627

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l293_293627


namespace evaluate_expression_l293_293953

theorem evaluate_expression (x : Real) (hx : x = -52.7) : 
  ⌈(⌊|x|⌋ + ⌈|x|⌉)⌉ = 105 := by
  sorry

end evaluate_expression_l293_293953


namespace min_inquiries_for_parity_l293_293682

-- Define the variables and predicates
variables (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n)

-- Define the main theorem we need to prove
theorem min_inquiries_for_parity (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n) : 
  ∃ k, (k = m + n - 4) := 
sorry

end min_inquiries_for_parity_l293_293682


namespace power_function_alpha_l293_293083

theorem power_function_alpha (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) (point_condition : f 8 = 2) : 
  α = 1 / 3 :=
by
  sorry

end power_function_alpha_l293_293083


namespace fixed_point_exists_l293_293532

-- Defining the function f
def f (a x : ℝ) : ℝ := a * x - 3 + 3

-- Stating that there exists a fixed point (3, 3a)
theorem fixed_point_exists (a : ℝ) : ∃ y : ℝ, f a 3 = y :=
by
  use (3 * a)
  simp [f]
  sorry

end fixed_point_exists_l293_293532


namespace total_pages_is_905_l293_293373

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l293_293373


namespace sad_children_count_l293_293123

-- Definitions of conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 18
def girls : ℕ := 42
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- Calculate the number of children who are either happy or sad
def happy_or_sad_children : ℕ := total_children - neither_happy_nor_sad_children

-- Prove that the number of sad children is 10
theorem sad_children_count : happy_or_sad_children - happy_children = 10 := by
  sorry

end sad_children_count_l293_293123


namespace contractor_net_earnings_l293_293563

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l293_293563


namespace sara_dozen_quarters_l293_293363

theorem sara_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (quarters_per_dozen : ℕ) 
  (h1 : dollars = 9) (h2 : quarters_per_dollar = 4) (h3 : quarters_per_dozen = 12) : 
  dollars * quarters_per_dollar / quarters_per_dozen = 3 := 
by 
  sorry

end sara_dozen_quarters_l293_293363


namespace brandon_textbooks_weight_l293_293112

-- Define the weights of Jon's textbooks
def weight_jon_book1 := 2
def weight_jon_book2 := 8
def weight_jon_book3 := 5
def weight_jon_book4 := 9

-- Calculate the total weight of Jon's textbooks
def total_weight_jon := weight_jon_book1 + weight_jon_book2 + weight_jon_book3 + weight_jon_book4

-- Define the condition where Jon's textbooks weigh three times as much as Brandon's textbooks
def jon_to_brandon_ratio := 3

-- Define the weight of Brandon's textbooks
def weight_brandon := total_weight_jon / jon_to_brandon_ratio

-- The goal is to prove that the weight of Brandon's textbooks is 8 pounds.
theorem brandon_textbooks_weight : weight_brandon = 8 := by
  sorry

end brandon_textbooks_weight_l293_293112


namespace find_a_l293_293383

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l293_293383


namespace sugar_inventory_l293_293342

theorem sugar_inventory :
  ∀ (initial : ℕ) (day2_use : ℕ) (day2_borrow : ℕ) (day3_buy : ℕ) (day4_buy : ℕ) (day5_use : ℕ) (day5_return : ℕ),
  initial = 65 →
  day2_use = 18 →
  day2_borrow = 5 →
  day3_buy = 30 →
  day4_buy = 20 →
  day5_use = 10 →
  day5_return = 3 →
  initial - day2_use - day2_borrow + day3_buy + day4_buy - day5_use + day5_return = 85 :=
by
  intros initial day2_use day2_borrow day3_buy day4_buy day5_use day5_return
  intro h_initial
  intro h_day2_use
  intro h_day2_borrow
  intro h_day3_buy
  intro h_day4_buy
  intro h_day5_use
  intro h_day5_return
  subst h_initial
  subst h_day2_use
  subst h_day2_borrow
  subst h_day3_buy
  subst h_day4_buy
  subst h_day5_use
  subst h_day5_return
  sorry

end sugar_inventory_l293_293342


namespace factor_roots_l293_293954

noncomputable def checkRoots (a b c t : ℚ) : Prop :=
  a * t^2 + b * t + c = 0

theorem factor_roots (t : ℚ) :
  checkRoots 8 17 (-10) t ↔ t = 5/8 ∨ t = -2 := by
sorry

end factor_roots_l293_293954


namespace triangle_inequality_l293_293045

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2)
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a < b + c) (h6 : b < a + c) (h7 : c < a + b) :
  a^2 + b^2 + c^2 + 2 * a * b * c < 2 := 
sorry

end triangle_inequality_l293_293045


namespace highest_growth_rate_at_K_div_2_l293_293675

variable {K : ℝ}

-- Define the population growth rate as a function of the population size.
def population_growth_rate (N : ℝ) : ℝ := sorry

-- Define the S-shaped curve condition of population growth.
axiom s_shaped_curve : ∃ N : ℝ, population_growth_rate N = 0 ∧ population_growth_rate (N/2) > population_growth_rate N

theorem highest_growth_rate_at_K_div_2 (N : ℝ) (hN : N = K/2) :
  population_growth_rate N > population_growth_rate K :=
by
  sorry

end highest_growth_rate_at_K_div_2_l293_293675


namespace bill_due_in_months_l293_293739

theorem bill_due_in_months
  (TD : ℝ) (FV : ℝ) (R_annual : ℝ) (m : ℝ) 
  (h₀ : TD = 270)
  (h₁ : FV = 2520)
  (h₂ : R_annual = 16) :
  m = 9 :=
by
  sorry

end bill_due_in_months_l293_293739


namespace sam_balloons_l293_293074

theorem sam_balloons (f d t S : ℝ) (h₁ : f = 10.0) (h₂ : d = 16.0) (h₃ : t = 40.0) (h₄ : f + S - d = t) : S = 46.0 := 
by 
  -- Replace "sorry" with a valid proof to solve this problem
  sorry

end sam_balloons_l293_293074


namespace octal_subtraction_correct_l293_293197

-- Define the octal numbers
def octal752 : ℕ := 7 * 8^2 + 5 * 8^1 + 2 * 8^0
def octal364 : ℕ := 3 * 8^2 + 6 * 8^1 + 4 * 8^0
def octal376 : ℕ := 3 * 8^2 + 7 * 8^1 + 6 * 8^0

-- Prove the octal number subtraction
theorem octal_subtraction_correct : octal752 - octal364 = octal376 := by
  sorry

end octal_subtraction_correct_l293_293197


namespace guess_probability_l293_293038

-- Definitions based on the problem conditions
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

def possible_attempts : ℕ := (5 * 4) -- A^2_5

def favorable_outcomes : ℕ := (4 * 2) -- C^1_4 * A^2_2

noncomputable def probability_correct_guess : ℝ :=
  (favorable_outcomes : ℝ) / (possible_attempts : ℝ)

-- Lean statement for the proof problem
theorem guess_probability : probability_correct_guess = 2 / 5 := by
  sorry

end guess_probability_l293_293038


namespace probability_of_prime_sum_l293_293546

def is_sum_of_numbers_prime_probability : ℚ :=
  let total_outcomes := 216
  let favorable_outcomes := 73
  favorable_outcomes / total_outcomes

theorem probability_of_prime_sum :
  is_sum_of_numbers_prime_probability = 73 / 216 :=
by
  sorry

end probability_of_prime_sum_l293_293546


namespace arithmetic_expression_equals_47_l293_293945

-- Define the arithmetic expression
def arithmetic_expression : ℕ :=
  2 + 5 * 3^2 - 4 + 6 * 2 / 3

-- The proof goal: arithmetic_expression equals 47
theorem arithmetic_expression_equals_47 : arithmetic_expression = 47 := 
by
  sorry

end arithmetic_expression_equals_47_l293_293945


namespace stay_nights_l293_293541

theorem stay_nights (cost_per_night : ℕ) (num_people : ℕ) (total_cost : ℕ) (n : ℕ) 
    (h1 : cost_per_night = 40) (h2 : num_people = 3) (h3 : total_cost = 360) (h4 : cost_per_night * num_people * n = total_cost) :
    n = 3 :=
sorry

end stay_nights_l293_293541


namespace trig_identity_l293_293757

open Real

theorem trig_identity : sin (20 * π / 180) * cos (10 * π / 180) - cos (160 * π / 180) * sin (170 * π / 180) = 1 / 2 := 
by
  sorry

end trig_identity_l293_293757


namespace space_station_cost_share_l293_293715

def total_cost : ℤ := 50 * 10^9
def people_count : ℤ := 500 * 10^6
def per_person_share (C N : ℤ) : ℤ := C / N

theorem space_station_cost_share :
  per_person_share total_cost people_count = 100 :=
by
  sorry

end space_station_cost_share_l293_293715


namespace count_more_blue_l293_293908

-- Definitions derived from the provided conditions
variables (total_people more_green both neither : ℕ)
variable (more_blue : ℕ)

-- Condition 1: There are 150 people in total
axiom total_people_def : total_people = 150

-- Condition 2: 90 people believe that teal is "more green"
axiom more_green_def : more_green = 90

-- Condition 3: 35 people believe it is both "more green" and "more blue"
axiom both_def : both = 35

-- Condition 4: 25 people think that teal is neither "more green" nor "more blue"
axiom neither_def : neither = 25


-- Theorem statement
theorem count_more_blue (total_people more_green both neither more_blue : ℕ) 
  (total_people_def : total_people = 150)
  (more_green_def : more_green = 90)
  (both_def : both = 35)
  (neither_def : neither = 25) :
  more_blue = 70 :=
by
  sorry

end count_more_blue_l293_293908


namespace nina_money_l293_293902

variable (C : ℝ)

theorem nina_money (h1: 6 * C = 8 * (C - 1.15)) : 6 * C = 27.6 := by
  have h2: C = 4.6 := sorry
  rw [h2]
  norm_num
  done

end nina_money_l293_293902


namespace no_real_solution_for_inequality_l293_293304

theorem no_real_solution_for_inequality :
  ∀ x : ℝ, ¬(3 * x^2 - x + 2 < 0) :=
by
  sorry

end no_real_solution_for_inequality_l293_293304


namespace ratio_B_C_l293_293174

variable (A B C : ℕ)
variable (h1 : A = B + 2)
variable (h2 : A + B + C = 37)
variable (h3 : B = 14)

theorem ratio_B_C : B / C = 2 := by
  sorry

end ratio_B_C_l293_293174


namespace evaluate_at_3_l293_293328

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 + x + 6

theorem evaluate_at_3 : g 3 = 135 := 
  by
  sorry

end evaluate_at_3_l293_293328


namespace limit_of_an_l293_293414

theorem limit_of_an (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n, a_n n = (4 * n - 3) / (2 * n + 1)) → 
  a = 2 → 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros ha hA ε hε
  sorry

end limit_of_an_l293_293414


namespace cos_180_eq_neg_one_l293_293595

theorem cos_180_eq_neg_one :
  Real.cos (Float.pi) = -1 := 
sorry

end cos_180_eq_neg_one_l293_293595


namespace cos_180_eq_neg_one_l293_293612

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l293_293612


namespace positive_difference_l293_293147

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l293_293147


namespace triangleProblem_correct_l293_293969

noncomputable def triangleProblem : Prop :=
  ∃ (a b c A B C : ℝ),
    A = 60 * Real.pi / 180 ∧
    b = 1 ∧
    (1 / 2) * b * c * Real.sin A = Real.sqrt 3 ∧
    Real.cos A = 1 / 2 ∧
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    (a / Real.sin A) = (b / Real.sin B) ∧ (b / Real.sin B) = (c / Real.sin C) ∧
    (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3

theorem triangleProblem_correct : triangleProblem :=
  sorry

end triangleProblem_correct_l293_293969


namespace cheesecake_factory_savings_l293_293359

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l293_293359


namespace john_hourly_wage_l293_293493

theorem john_hourly_wage (days_off: ℕ) (hours_per_day: ℕ) (weekly_wage: ℕ) 
  (days_off_eq: days_off = 3) (hours_per_day_eq: hours_per_day = 4) (weekly_wage_eq: weekly_wage = 160):
  (weekly_wage / ((7 - days_off) * hours_per_day) = 10) :=
by
  /-
  Given:
  days_off = 3
  hours_per_day = 4
  weekly_wage = 160

  To prove:
  weekly_wage / ((7 - days_off) * hours_per_day) = 10
  -/
  sorry

end john_hourly_wage_l293_293493


namespace binomial_square_l293_293939

theorem binomial_square (a b : ℝ) : (2 * a - 3 * b)^2 = 4 * a^2 - 12 * a * b + 9 * b^2 :=
by
  sorry

end binomial_square_l293_293939


namespace gcd_lcm_product_l293_293310

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l293_293310


namespace pentagon_area_l293_293512

theorem pentagon_area (P Q R S T : Point)
  (h1 : is_square P Q R S)
  (h2 : is_perpendicular P T R)
  (h3 : distance P T = 5)
  (h4 : distance T R = 12) :
  area_of_pentagon P T R S Q = 139 := by
  sorry

end pentagon_area_l293_293512


namespace solve_for_x_l293_293639

theorem solve_for_x (x : ℝ) (h : (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4)) : x = 257 / 16 := by
  sorry

end solve_for_x_l293_293639


namespace matchsticks_for_3_by_1996_grid_l293_293744

def total_matchsticks_needed (rows cols : ℕ) : ℕ :=
  (cols * (rows + 1)) + (rows * (cols + 1))

theorem matchsticks_for_3_by_1996_grid : total_matchsticks_needed 3 1996 = 13975 := by
  sorry

end matchsticks_for_3_by_1996_grid_l293_293744


namespace julius_wins_probability_l293_293681

noncomputable def probability_julius_wins (p_julius p_larry : ℚ) : ℚ :=
  (p_julius / (1 - p_larry ^ 2))

theorem julius_wins_probability :
  probability_julius_wins (2/3) (1/3) = 3/4 :=
by
  sorry

end julius_wins_probability_l293_293681


namespace B_completes_remaining_work_in_12_days_l293_293761

-- Definitions for conditions.
def work_rate_a := 1/15
def work_rate_b := 1/18
def days_worked_by_a := 5

-- Calculation of work done by A and the remaining work for B
def work_done_by_a := days_worked_by_a * work_rate_a
def remaining_work := 1 - work_done_by_a

-- Proof statement
theorem B_completes_remaining_work_in_12_days : 
  ∀ (work_rate_a work_rate_b : ℚ), 
    work_rate_a = 1/15 → 
    work_rate_b = 1/18 → 
    days_worked_by_a = 5 → 
    work_done_by_a = days_worked_by_a * work_rate_a → 
    remaining_work = 1 - work_done_by_a → 
    (remaining_work / work_rate_b) = 12 :=
by 
  intros 
  sorry

end B_completes_remaining_work_in_12_days_l293_293761


namespace divisible_by_units_digit_l293_293211

theorem divisible_by_units_digit :
  ∃ l : List ℕ, l = [21, 22, 24, 25] ∧ l.length = 4 := 
  sorry

end divisible_by_units_digit_l293_293211


namespace sum_of_perimeters_of_squares_l293_293876

theorem sum_of_perimeters_of_squares (x y : ℕ)
  (h1 : x^2 - y^2 = 19) : 4 * x + 4 * y = 76 := 
by
  sorry

end sum_of_perimeters_of_squares_l293_293876


namespace brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l293_293851

def march_13_2007_day_of_week : String := "Tuesday"

def days_until_brothers_birthday : Nat := 2000

def start_date := (2007, 3, 13)  -- (year, month, day)

def days_per_week := 7

def carlos_initial_age := 7

def day_of_week_after_n_days (start_day : String) (n : Nat) : String :=
  match n % 7 with
  | 0 => "Tuesday"
  | 1 => "Wednesday"
  | 2 => "Thursday"
  | 3 => "Friday"
  | 4 => "Saturday"
  | 5 => "Sunday"
  | 6 => "Monday"
  | _ => "Unknown" -- This case should never happen

def carlos_age_after_n_days (initial_age : Nat) (n : Nat) : Nat :=
  initial_age + n / 365

theorem brother_15th_birthday_day_of_week : 
  day_of_week_after_n_days march_13_2007_day_of_week days_until_brothers_birthday = "Sunday" := 
by sorry

theorem carlos_age_on_brothers_15th_birthday :
  carlos_age_after_n_days carlos_initial_age days_until_brothers_birthday = 12 :=
by sorry

end brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l293_293851


namespace simplify_fraction_l293_293699

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l293_293699


namespace simplify_expression_l293_293863

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (2 * x + 10) - (x + 3) * (3 * x - 2) = 3 * x^2 + 15 * x - 34 := 
by
  sorry

end simplify_expression_l293_293863


namespace inequality_solution_set_l293_293992

theorem inequality_solution_set 
  (m n : ℤ)
  (h1 : ∀ x : ℤ, mx - n > 0 → x < 1 / 3)
  (h2 : ∀ x : ℤ, (m + n) x < n - m) :
  ∀ x : ℤ, x > -1 / 2 := 
sorry

end inequality_solution_set_l293_293992


namespace sufficient_but_not_necessary_l293_293814

-- Definitions of conditions
def p (x : ℝ) : Prop := 1 / (x + 1) > 0
def q (x : ℝ) : Prop := (1/x > 0)

-- Main theorem statement
theorem sufficient_but_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
sorry

end sufficient_but_not_necessary_l293_293814


namespace max_value_fraction_sum_l293_293498

theorem max_value_fraction_sum (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 3) :
  (ab / (a + b + 1) + ac / (a + c + 1) + bc / (b + c + 1) ≤ 3 / 2) :=
sorry

end max_value_fraction_sum_l293_293498


namespace opposite_of_5_is_neg_5_l293_293388

def opposite_number (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_5_is_neg_5 : opposite_number 5 (-5) := by
  sorry

end opposite_of_5_is_neg_5_l293_293388


namespace coconuts_total_l293_293432

theorem coconuts_total (B_trips : Nat) (Ba_coconuts_per_trip : Nat) (Br_coconuts_per_trip : Nat) (combined_trips : Nat) (B_totals : B_trips = 12) (Ba_coconuts : Ba_coconuts_per_trip = 4) (Br_coconuts : Br_coconuts_per_trip = 8) : combined_trips * (Ba_coconuts_per_trip + Br_coconuts_per_trip) = 144 := 
by
  simp [B_totals, Ba_coconuts, Br_coconuts]
  sorry

end coconuts_total_l293_293432


namespace gcd_72_120_168_l293_293535

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
by
  sorry

end gcd_72_120_168_l293_293535


namespace exists_two_digit_number_l293_293273

theorem exists_two_digit_number :
  ∃ x y : ℕ, (1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ (10 * x + y = (x + y) * (x - y)) ∧ (10 * x + y = 48) :=
by
  sorry

end exists_two_digit_number_l293_293273


namespace find_k_l293_293207

theorem find_k (k : ℝ) :
  (∀ x, x^2 + k*x + 10 = 0 → (∃ r s : ℝ, x = r ∨ x = s) ∧ r + s = -k ∧ r * s = 10) ∧
  (∀ x, x^2 - k*x + 10 = 0 → (∃ r s : ℝ, x = r + 4 ∨ x = s + 4) ∧ (r + 4) + (s + 4) = k) → 
  k = 4 :=
by
  sorry

end find_k_l293_293207


namespace limit_of_sequence_limit_frac_seq_l293_293415

def N (ε : ℝ) : ℕ := ⌈((5 / ε) - 1) / 2⌉.toNat

theorem limit_of_sequence (ε : ℝ) (n : ℕ) (hn : n ≥ N ε) 
  (hε_pos : ε > 0) : 
  abs ((4 * n - 3) / (2 * n + 1) - 2) < ε :=
sorry

theorem limit_frac_seq : 
  tendsto (λ n, (4 * n - 3) / (2 * n + 1)) at_top (𝓝 2) :=
begin
  intros ε hε,
  use N ε,
  intros n hn,
  exact limit_of_sequence ε n hn hε,
end

end limit_of_sequence_limit_frac_seq_l293_293415


namespace students_taking_one_language_l293_293021

-- Definitions based on the conditions
def french_class_students : ℕ := 21
def spanish_class_students : ℕ := 21
def both_languages_students : ℕ := 6
def total_students : ℕ := french_class_students + spanish_class_students - both_languages_students

-- The theorem we want to prove
theorem students_taking_one_language :
    total_students = 36 :=
by
  -- Add the proof here
  sorry

end students_taking_one_language_l293_293021


namespace b_finishes_remaining_work_in_5_days_l293_293168

theorem b_finishes_remaining_work_in_5_days :
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  days_b_to_finish = 5 :=
by
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  show days_b_to_finish = 5
  sorry

end b_finishes_remaining_work_in_5_days_l293_293168


namespace salary_january_l293_293136

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8300)
  (h3 : May = 6500) :
  J = 5300 :=
by
  sorry

end salary_january_l293_293136


namespace total_fires_l293_293928

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end total_fires_l293_293928


namespace solve_m_l293_293668

theorem solve_m (m : ℝ) :
  (∃ x > 0, (2 * m - 4) ^ 2 = x ∧ (3 * m - 1) ^ 2 = x) →
  (m = -3 ∨ m = 1) :=
by 
  sorry

end solve_m_l293_293668


namespace cos_180_eq_neg_one_l293_293609

theorem cos_180_eq_neg_one : real.cos (real.pi) = -1 := 
by 
  sorry

end cos_180_eq_neg_one_l293_293609


namespace triangle_with_angle_ratio_is_right_l293_293483

theorem triangle_with_angle_ratio_is_right (
  (k : ℝ) 
  (h_ratio : (k + 2 * k + 3 * k = 180)) 
) : (30 ≤ k ∧ k ≤ 30) ∧ (2 * k = 2 * 30) ∧ (3 * k = 3 * 30) 
  ∧ (k = 30) ∧ (90 = 3 * k) :=
by {
  sorry
}

end triangle_with_angle_ratio_is_right_l293_293483


namespace apartment_building_floors_l293_293904

theorem apartment_building_floors (K E P : ℕ) (h1 : 1 < K) (h2 : K < E) (h3 : E < P) (h4 : K * E * P = 715) : 
  E = 11 :=
sorry

end apartment_building_floors_l293_293904


namespace q_r_share_difference_l293_293926

theorem q_r_share_difference
  (T : ℝ) -- Total amount of money
  (x : ℝ) -- Common multiple of shares
  (p_share q_share r_share s_share : ℝ) -- Shares before tax
  (p_tax q_tax r_tax s_tax : ℝ) -- Tax percentages
  (h_ratio : p_share = 3 * x ∧ q_share = 7 * x ∧ r_share = 12 * x ∧ s_share = 5 * x) -- Ratio condition
  (h_tax : p_tax = 0.10 ∧ q_tax = 0.15 ∧ r_tax = 0.20 ∧ s_tax = 0.25) -- Tax condition
  (h_difference_pq : q_share * (1 - q_tax) - p_share * (1 - p_tax) = 2400) -- Difference between p and q after tax
  : (r_share * (1 - r_tax) - q_share * (1 - q_tax)) = 2695.38 := sorry

end q_r_share_difference_l293_293926


namespace rose_needs_more_money_l293_293513

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l293_293513


namespace fraction_inequality_l293_293982

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l293_293982


namespace length_difference_squares_l293_293132

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end length_difference_squares_l293_293132


namespace smallest_possible_recording_l293_293170

theorem smallest_possible_recording :
  ∃ (A B C : ℤ), 
      (0 ≤ A ∧ A ≤ 10) ∧ 
      (0 ≤ B ∧ B ≤ 10) ∧ 
      (0 ≤ C ∧ C ≤ 10) ∧ 
      (A + B + C = 12) ∧ 
      (A + B + C) % 5 = 0 ∧ 
      A = 0 :=
by
  sorry

end smallest_possible_recording_l293_293170


namespace num_triangles_correct_num_lines_correct_l293_293850

-- Definition for the first proof problem: Number of triangles
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem num_triangles_correct :
  num_triangles 9 = 84 :=
by
  sorry

-- Definition for the second proof problem: Number of lines
def num_lines (n : ℕ) : ℕ := Nat.choose n 2

theorem num_lines_correct :
  num_lines 9 = 36 :=
by
  sorry

end num_triangles_correct_num_lines_correct_l293_293850


namespace nanometers_to_scientific_notation_l293_293227

theorem nanometers_to_scientific_notation :
  (246 : ℝ) * (10 ^ (-9 : ℝ)) = (2.46 : ℝ) * (10 ^ (-7 : ℝ)) :=
by
  sorry

end nanometers_to_scientific_notation_l293_293227


namespace initial_books_in_library_l293_293880

theorem initial_books_in_library 
  (initial_books : ℕ)
  (books_taken_out_Tuesday : ℕ := 120)
  (books_returned_Wednesday : ℕ := 35)
  (books_withdrawn_Thursday : ℕ := 15)
  (books_final_count : ℕ := 150)
  : initial_books - books_taken_out_Tuesday + books_returned_Wednesday - books_withdrawn_Thursday = books_final_count → initial_books = 250 :=
by
  intros h
  sorry

end initial_books_in_library_l293_293880


namespace find_5a_plus_5b_l293_293302

noncomputable def g (x : ℝ) : ℝ := 5 * x - 4
noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def f_inv (a b x : ℝ) : ℝ := g x + 3

theorem find_5a_plus_5b (a b : ℝ) (h_inverse : ∀ x, f_inv a b (f a b x) = x) : 5 * a + 5 * b = 2 :=
by
  sorry

end find_5a_plus_5b_l293_293302


namespace total_points_correct_l293_293034

-- Define the number of teams
def num_teams : ℕ := 16

-- Define the number of draws
def num_draws : ℕ := 30

-- Define the scoring system
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def loss_deduction_threshold : ℕ := 3
def points_deduction_per_threshold : ℕ := 1

-- Define the total number of games
def total_games : ℕ := num_teams * (num_teams - 1) / 2

-- Define the number of wins (non-draw games)
def num_wins : ℕ := total_games - num_draws

-- Define the total points from wins
def total_points_from_wins : ℕ := num_wins * points_for_win

-- Define the total points from draws
def total_points_from_draws : ℕ := num_draws * points_for_draw * 2

-- Define the total points (as no team lost more than twice, no deductions apply)
def total_points : ℕ := total_points_from_wins + total_points_from_draws

theorem total_points_correct :
  total_points = 330 := by
  sorry

end total_points_correct_l293_293034


namespace cistern_water_depth_l293_293422

theorem cistern_water_depth
  (length width : ℝ) 
  (wet_surface_area : ℝ)
  (h : ℝ) 
  (hl : length = 7)
  (hw : width = 4)
  (ha : wet_surface_area = 55.5)
  (h_eq : 28 + 22 * h = wet_surface_area) 
  : h = 1.25 := 
  by 
  sorry

end cistern_water_depth_l293_293422


namespace probability_one_shirt_two_shorts_one_sock_l293_293223

-- Define the number of each type of clothing
def shirts := 5
def shorts := 6
def socks := 7
def total_articles := shirts + shorts + socks
def selected_articles := 4

-- Calculate combinations
def total_combinations := Nat.choose total_articles selected_articles
def shirt_combinations := Nat.choose shirts 1
def short_combinations := Nat.choose shorts 2
def sock_combinations := Nat.choose socks 1

-- Calculate probability
def favorable_outcomes := shirt_combinations * short_combinations * sock_combinations
def required_probability := (favorable_outcomes : ℚ) / total_combinations

-- Prove the probability
theorem probability_one_shirt_two_shorts_one_sock :
  required_probability = 35 / 204 :=
by
  -- Sorry placeholder for the proof
  sorry

end probability_one_shirt_two_shorts_one_sock_l293_293223


namespace partition_exists_for_all_pos_k_l293_293843

open Finset

noncomputable def partition_sum_eq (k : ℕ) (hk : k > 0) : Prop :=
  ∃ (X Y : Finset ℕ), 
    (X ∪ Y = range (2^(k+1)) ∧ X ∩ Y = ∅) ∧ 
    ∀ (m : ℕ), m ∈ range(k+1) → ∑ x in X, x ^ m = ∑ y in Y, y ^ m

theorem partition_exists_for_all_pos_k : ∀ (k : ℕ), 0 < k → partition_sum_eq k := sorry

end partition_exists_for_all_pos_k_l293_293843


namespace percentage_of_men_attended_picnic_l293_293673

variable (E : ℝ) (W M P : ℝ)
variable (H1 : M = 0.5 * E)
variable (H2 : W = 0.5 * E)
variable (H3 : 0.4 * W = 0.2 * E)
variable (H4 : 0.3 * E = P * M + 0.2 * E)

theorem percentage_of_men_attended_picnic : P = 0.2 :=
by sorry

end percentage_of_men_attended_picnic_l293_293673


namespace more_apples_than_pears_l293_293542

-- Define the variables
def apples := 17
def pears := 9

-- Theorem: The number of apples minus the number of pears equals 8
theorem more_apples_than_pears : apples - pears = 8 :=
by
  sorry

end more_apples_than_pears_l293_293542


namespace height_of_pole_l293_293291

/-- A telephone pole is supported by a steel cable extending from the top of the pole to a point on the ground 3 meters from its base.
When Leah, who is 1.5 meters tall, stands 2.5 meters from the base of the pole towards the point where the cable is attached to the ground,
her head just touches the cable. Prove that the height of the pole is 9 meters. -/
theorem height_of_pole 
  (cable_length_from_base : ℝ)
  (leah_distance_from_base : ℝ)
  (leah_height : ℝ)
  : cable_length_from_base = 3 → leah_distance_from_base = 2.5 → leah_height = 1.5 → 
    (∃ height_of_pole : ℝ, height_of_pole = 9) := 
by
  intros h1 h2 h3
  sorry

end height_of_pole_l293_293291


namespace complement_union_A_B_l293_293166

open Set

variable {U : Type*} [Preorder U] [BoundedOrder U]

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  compl (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_A_B_l293_293166


namespace inequality_solution_l293_293404

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  sorry

end inequality_solution_l293_293404


namespace vector_c_correct_l293_293235

theorem vector_c_correct (a b c : ℤ × ℤ) (h_a : a = (1, -3)) (h_b : b = (-2, 4))
    (h_condition : 4 • a + (3 • b - 2 • a) + c = (0, 0)) :
    c = (4, -6) :=
by 
  -- The proof steps go here, but we'll skip them with 'sorry' for now.
  sorry

end vector_c_correct_l293_293235


namespace taylor_family_reunion_adults_l293_293579

def number_of_kids : ℕ := 45
def number_of_tables : ℕ := 14
def people_per_table : ℕ := 12
def total_people := number_of_tables * people_per_table

theorem taylor_family_reunion_adults : total_people - number_of_kids = 123 := by
  sorry

end taylor_family_reunion_adults_l293_293579


namespace probability_A1_selected_probability_neither_A2_B2_selected_l293_293039

-- Define the set of students
structure Student := (id : String) (gender : String)

def students : List Student :=
  [⟨"A1", "M"⟩, ⟨"A2", "M"⟩, ⟨"A3", "M"⟩, ⟨"A4", "M"⟩, ⟨"B1", "F"⟩, ⟨"B2", "F"⟩, ⟨"B3", "F"⟩]

-- Define the conditions
def males := students.filter (λ s => s.gender = "M")
def females := students.filter (λ s => s.gender = "F")

def possible_pairs : List (Student × Student) :=
  List.product males females

-- Prove the probability of selecting A1
theorem probability_A1_selected : (3 : ℚ) / (12 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by
  sorry

-- Prove the probability that neither A2 nor B2 are selected
theorem probability_neither_A2_B2_selected : (11 : ℚ) / (12 : ℚ) = (11 : ℚ) / (12 : ℚ) :=
by
  sorry

end probability_A1_selected_probability_neither_A2_B2_selected_l293_293039


namespace expression_evaluation_l293_293314

theorem expression_evaluation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 = 1 / y^2) :
  (x^2 - 4 / x^2) * (y^2 + 4 / y^2) = x^4 - 16 / x^4 :=
by
  sorry

end expression_evaluation_l293_293314


namespace divide_gray_area_l293_293832

-- The conditions
variables {A_rectangle A_square : ℝ} (h : 0 ≤ A_square ∧ A_square ≤ A_rectangle)

-- The main statement
theorem divide_gray_area : ∃ l : ℝ → ℝ → Prop, (∀ (x : ℝ), l x (A_rectangle / 2)) ∧ (∀ (y : ℝ), l (A_square / 2) y) ∧ (A_rectangle - A_square) / 2 = (A_rectangle - A_square) / 2 := by sorry

end divide_gray_area_l293_293832


namespace probability_correct_l293_293994

noncomputable def probability_point_between_lines : ℝ :=
  let intersection_x_l := 4    -- x-intercept of line l
  let intersection_x_m := 3    -- x-intercept of line m
  let area_under_l := (1 / 2) * intersection_x_l * 8 -- area under line l
  let area_under_m := (1 / 2) * intersection_x_m * 9 -- area under line m
  let area_between := area_under_l - area_under_m    -- area between lines
  (area_between / area_under_l : ℝ)

theorem probability_correct : probability_point_between_lines = 0.16 :=
by
  simp only [probability_point_between_lines]
  sorry

end probability_correct_l293_293994


namespace cos_180_eq_neg1_l293_293591

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l293_293591


namespace ribbon_difference_correct_l293_293026

theorem ribbon_difference_correct : 
  ∀ (L W H : ℕ), L = 22 → W = 22 → H = 11 → 
  let method1 := 2 * L + 2 * W + 4 * H + 24
      method2 := 2 * L + 4 * W + 2 * H + 24
  in method2 - method1 = 22 :=
begin
  intros L W H hL hW hH,
  let method1 := 2 * L + 2 * W + 4 * H + 24,
  let method2 := 2 * L + 4 * W + 2 * H + 24,
  calc
    method2 - method1 = (2 * L + 4 * W + 2 * H + 24) - (2 * L + 2 * W + 4 * H + 24) : by sorry
                    ... = 22 : by sorry,
end

end ribbon_difference_correct_l293_293026


namespace original_gain_percentage_is_5_l293_293915

def costPrice : ℝ := 200
def newCostPrice : ℝ := costPrice * 0.95
def desiredProfitRatio : ℝ := 0.10
def newSellingPrice : ℝ := newCostPrice * (1 + desiredProfitRatio)
def originalSellingPrice : ℝ := newSellingPrice + 1

theorem original_gain_percentage_is_5 :
  ((originalSellingPrice - costPrice) / costPrice) * 100 = 5 :=
by 
  sorry

end original_gain_percentage_is_5_l293_293915


namespace is_isosceles_right_triangle_l293_293079

theorem is_isosceles_right_triangle 
  {a b c : ℝ}
  (h : |c^2 - a^2 - b^2| + (a - b)^2 = 0) : 
  a = b ∧ c^2 = a^2 + b^2 :=
sorry

end is_isosceles_right_triangle_l293_293079


namespace total_marks_is_275_l293_293717

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end total_marks_is_275_l293_293717


namespace rolling_circle_trace_eq_envelope_l293_293172

-- Definitions for the geometrical setup
variable {a : ℝ} (C : ℝ → ℝ → Prop)

-- The main statement to prove
theorem rolling_circle_trace_eq_envelope (hC : ∀ t : ℝ, C (a * t) a) :
  ∃ P : ℝ × ℝ → Prop, ∀ t : ℝ, C (a/2 * t + a/2 * Real.sin t) (a/2 + a/2 * Real.cos t) :=
by
  sorry

end rolling_circle_trace_eq_envelope_l293_293172


namespace tangent_lines_l293_293315

noncomputable def curve1 (x : ℝ) : ℝ := 2 * x ^ 2 - 5
noncomputable def curve2 (x : ℝ) : ℝ := x ^ 2 - 3 * x + 5

theorem tangent_lines :
  (∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = -20 * x - 55 ∨ y = -13 * x - 20 ∨ y = 8 * x - 13 ∨ y = x + 1) ∧ 
    (
      (m₁ = 4 * 2 ∧ b₁ = 3) ∨ 
      (m₁ = 2 * -5 - 3 ∧ b₁ = 45) ∨
      (m₂ = 4 * -5 ∧ b₂ = 45) ∨
      (m₂ = 2 * 2 - 3 ∧ b₂ = 3)
    )) :=
sorry

end tangent_lines_l293_293315


namespace bob_stickers_l293_293301

variables {B T D : ℕ}

theorem bob_stickers (h1 : D = 72) (h2 : T = 3 * B) (h3 : D = 2 * T) : B = 12 :=
by
  sorry

end bob_stickers_l293_293301


namespace average_age_of_coaches_l293_293530

variables 
  (total_members : ℕ) (avg_age_total : ℕ) 
  (num_girls : ℕ) (num_boys : ℕ) (num_coaches : ℕ) 
  (avg_age_girls : ℕ) (avg_age_boys : ℕ)

theorem average_age_of_coaches 
  (h1 : total_members = 50) 
  (h2 : avg_age_total = 18)
  (h3 : num_girls = 25) 
  (h4 : num_boys = 20) 
  (h5 : num_coaches = 5)
  (h6 : avg_age_girls = 16)
  (h7 : avg_age_boys = 17) : 
  (900 - (num_girls * avg_age_girls + num_boys * avg_age_boys)) / num_coaches = 32 :=
by
  sorry

end average_age_of_coaches_l293_293530


namespace range_of_abscissa_l293_293664

/--
Given three points A, F1, F2 in the Cartesian plane and a point P satisfying the given conditions,
prove that the range of the abscissa of point P is [0, 3].

Conditions:
- A = (1, 0)
- F1 = (-2, 0)
- F2 = (2, 0)
- \| overrightarrow{PF1} \| + \| overrightarrow{PF2} \| = 6
- \| overrightarrow{PA} \| ≤ sqrt(6)
-/
theorem range_of_abscissa :
  ∀ (P : ℝ × ℝ),
    (|P.1 + 2| + |P.1 - 2| = 6) →
    ((P.1 - 1)^2 + P.2^2 ≤ 6) →
    (0 ≤ P.1 ∧ P.1 ≤ 3) :=
by
  intros P H1 H2
  sorry

end range_of_abscissa_l293_293664


namespace roller_coaster_cost_l293_293795

variable (ferris_wheel_rides : Nat) (log_ride_rides : Nat) (rc_rides : Nat)
variable (ferris_wheel_cost : Nat) (log_ride_cost : Nat)
variable (initial_tickets : Nat) (additional_tickets : Nat)
variable (total_needed_tickets : Nat)

theorem roller_coaster_cost :
  ferris_wheel_rides = 2 →
  log_ride_rides = 7 →
  rc_rides = 3 →
  ferris_wheel_cost = 2 →
  log_ride_cost = 1 →
  initial_tickets = 20 →
  additional_tickets = 6 →
  total_needed_tickets = initial_tickets + additional_tickets →
  let total_ride_costs := ferris_wheel_rides * ferris_wheel_cost + log_ride_rides * log_ride_cost
  let rc_cost := (total_needed_tickets - total_ride_costs) / rc_rides
  rc_cost = 5 := by
  sorry

end roller_coaster_cost_l293_293795


namespace length_of_train_l293_293551

-- We define the conditions
def crosses_platform_1 (L : ℝ) : Prop := 
  let v := (L + 100) / 15
  v = (L + 100) / 15

def crosses_platform_2 (L : ℝ) : Prop := 
  let v := (L + 250) / 20
  v = (L + 250) / 20

-- We state the main theorem we need to prove
theorem length_of_train :
  ∃ L : ℝ, crosses_platform_1 L ∧ crosses_platform_2 L ∧ (L = 350) :=
sorry

end length_of_train_l293_293551


namespace min_mag_of_z3_l293_293820

noncomputable def min_mag_z3 (z1 z2 z3 : ℂ) : ℝ :=
  if (|z1| = 1 ∧ |z2| = 1 ∧ |z1 + z2 + z3| = 1 ∧ (∀ t:ℂ, z1 / z2 = t ∧ t.im = t)) 
  then (Real.sqrt 2 - 1)
  else 0

theorem min_mag_of_z3 (z1 z2 z3 : ℂ) (hz1 : |z1| = 1) (hz2 : |z2| = 1)
  (hz_sum : |z1 + z2 + z3| = 1) (hz1hz2_imag : (z1 / z2).re = 0) :
  |z3| = Real.sqrt 2 - 1 :=
by
  sorry

end min_mag_of_z3_l293_293820


namespace taxable_income_l293_293848

theorem taxable_income (tax_paid : ℚ) (state_tax_rate : ℚ) (months_resident : ℚ) (total_months : ℚ) (T : ℚ) :
  tax_paid = 1275 ∧ state_tax_rate = 0.04 ∧ months_resident = 9 ∧ total_months = 12 → 
  T = 42500 :=
by
  intros h
  sorry

end taxable_income_l293_293848


namespace area_of_fourth_rectangle_l293_293768

theorem area_of_fourth_rectangle (a b c d : ℕ) (h1 : a = 18) (h2 : b = 27) (h3 : c = 12) :
d = 93 :=
by
  -- Problem reduces to showing that d equals 93 using the given h1, h2, h3
  sorry

end area_of_fourth_rectangle_l293_293768


namespace tiled_board_remainder_l293_293280

def num_ways_to_tile_9x1 : Nat := -- hypothetical function to calculate the number of ways
  sorry

def N : Nat :=
  num_ways_to_tile_9x1 -- placeholder for N, should be computed using correct formula

theorem tiled_board_remainder : N % 1000 = 561 :=
  sorry

end tiled_board_remainder_l293_293280


namespace chosen_number_l293_293766

theorem chosen_number (x : ℝ) (h1 : x / 9 - 100 = 10) : x = 990 :=
  sorry

end chosen_number_l293_293766


namespace total_doctors_and_nurses_l293_293396

theorem total_doctors_and_nurses
    (ratio_doctors_nurses : ℕ -> ℕ -> Prop)
    (num_nurses : ℕ)
    (h₁ : ratio_doctors_nurses 2 3)
    (h₂ : num_nurses = 150) :
    ∃ num_doctors total_doctors_nurses, 
    (total_doctors_nurses = num_doctors + num_nurses) 
    ∧ (num_doctors / num_nurses = 2 / 3) 
    ∧ total_doctors_nurses = 250 := 
by
  sorry

end total_doctors_and_nurses_l293_293396


namespace johns_spent_amount_l293_293772

def original_price : ℝ := 2000
def increase_rate : ℝ := 0.02

theorem johns_spent_amount : 
  let increased_amount := original_price * increase_rate in
  let john_total := original_price + increased_amount in
  john_total = 2040 :=
by
  sorry

end johns_spent_amount_l293_293772


namespace irrational_number_l293_293188

theorem irrational_number (a b c d : ℝ) (h₁ : 0.55555 = a) (h₂ : a ∈ ℚ)
  (h₃ : b = π / 2) (h₄ : c = 22 / 3) (h₅ : c ∈ ℚ) (h₆ : d = 3.121121121112) (h₇ : d ∈ ℚ) :
  irrational b :=
by sorry

end irrational_number_l293_293188


namespace find_k_in_geometric_sequence_l293_293106

theorem find_k_in_geometric_sequence (a : ℕ → ℕ) (k : ℕ)
  (h1 : ∀ n, a n = a 2 * 3^(n-2))
  (h2 : a 2 = 3)
  (h3 : a 3 = 9)
  (h4 : a k = 243) :
  k = 6 :=
sorry

end find_k_in_geometric_sequence_l293_293106


namespace g_at_4_l293_293329

def g (x : ℝ) : ℝ := 5 * x + 6

theorem g_at_4 : g 4 = 26 :=
by
  sorry

end g_at_4_l293_293329


namespace part_a_part_b_l293_293240

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem part_a : ∃ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n :=
by { use 51715, split, { exact ⟨by norm_num, by norm_num⟩ }, split, { norm_num }, { norm_num } }

theorem part_b : ∃ c : ℕ, (∀ n : ℕ, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n ↔ n ∈ list.range c) ∧ c = 100 :=
by sorry

end part_a_part_b_l293_293240


namespace part_one_part_two_l293_293662

def f (x m : ℝ) := abs (x + m) + abs (2 * x - 1)

theorem part_one (x : ℝ) (h : f x 1 ≥ 3) : x ≤ -1 ∨ x ≥ 1 :=
by sorry

theorem part_two (m x : ℝ) (hm_pos : m > 0) (hx_range : x ∈ Icc m (2 * m^2)) :
  (1 / 2) * f x m ≤ abs (x + 1) → 1 / 2 < m ∧ m ≤ 1 :=
by sorry

end part_one_part_two_l293_293662


namespace factorize_polynomial_l293_293799

theorem factorize_polynomial (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2 * y) ^ 2 := 
by sorry

end factorize_polynomial_l293_293799


namespace sum_of_three_consecutive_odd_integers_l293_293403

theorem sum_of_three_consecutive_odd_integers (n : ℤ) 
  (h1 : n + (n + 4) = 130) 
  (h2 : n % 2 = 1) : 
  n + (n + 2) + (n + 4) = 195 := 
by
  sorry

end sum_of_three_consecutive_odd_integers_l293_293403


namespace children_working_initially_l293_293555

theorem children_working_initially (W C : ℝ) (n : ℕ) 
  (h1 : 10 * W = 1 / 5) 
  (h2 : n * C = 1 / 10) 
  (h3 : 5 * W + 10 * C = 1 / 5) : 
  n = 10 :=
by
  sorry

end children_working_initially_l293_293555


namespace find_a_l293_293382

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l293_293382


namespace sum_of_solutions_l293_293402

theorem sum_of_solutions : 
  (∀ x : ℝ, (3 * x) / 15 = 4 / x) → (0 + 4 = 4) :=
by
  sorry

end sum_of_solutions_l293_293402


namespace cos_180_eq_neg1_l293_293589

noncomputable def cos_of_180_deg : ℝ := 
  let p0 : ℝ × ℝ := (1, 0)
  let p180 : ℝ × ℝ := (-1, 0)
  if h : ∠ (0 : ℝ, 0, p0, p180) = real.angle.pi then -1 else sorry

theorem cos_180_eq_neg1 : cos 180 = cos_of_180_deg := sorry

end cos_180_eq_neg1_l293_293589


namespace bethany_total_hours_l293_293937

-- Define the hours Bethany rode on each set of days
def hours_mon_wed_fri : ℕ := 3  -- 1 hour each on Monday, Wednesday, and Friday
def hours_tue_thu : ℕ := 1  -- 30 min each on Tuesday and Thursday
def hours_sat : ℕ := 2  -- 2 hours on Saturday

-- Define the total hours per week
def total_hours_per_week : ℕ := hours_mon_wed_fri + hours_tue_thu + hours_sat

-- Define the total hours in 2 weeks
def total_hours_in_2_weeks : ℕ := total_hours_per_week * 2

-- Prove that the total hours in 2 weeks is 12
theorem bethany_total_hours : total_hours_in_2_weeks = 12 :=
by
  -- Replace the definitions with their values and check the equality
  rw [total_hours_in_2_weeks, total_hours_per_week, hours_mon_wed_fri, hours_tue_thu, hours_sat]
  simp
  norm_num
  sorry

end bethany_total_hours_l293_293937


namespace line_equation_through_M_P_Q_l293_293456

-- Given that M is the midpoint between P and Q, we should have:
-- M = (1, -2)
-- P = (2, 0)
-- Q = (0, -4)
-- We need to prove that the line passing through these points has the equation 2x - y - 4 = 0

theorem line_equation_through_M_P_Q :
  ∀ (x y : ℝ), (1 - 2 = (2 * (x - 1)) ∧ 0 - 2 = (2 * (0 - (-2)))) ->
  (x - y - 4 = 0) := 
by
  sorry

end line_equation_through_M_P_Q_l293_293456


namespace neither_chemistry_nor_biology_l293_293220

variable (club_size chemistry_students biology_students both_students neither_students : ℕ)

def students_in_club : Prop :=
  club_size = 75

def students_taking_chemistry : Prop :=
  chemistry_students = 40

def students_taking_biology : Prop :=
  biology_students = 35

def students_taking_both : Prop :=
  both_students = 25

theorem neither_chemistry_nor_biology :
  students_in_club club_size ∧ 
  students_taking_chemistry chemistry_students ∧
  students_taking_biology biology_students ∧
  students_taking_both both_students →
  neither_students = 75 - ((chemistry_students - both_students) + (biology_students - both_students) + both_students) :=
by
  intros
  sorry

end neither_chemistry_nor_biology_l293_293220


namespace total_pages_correct_l293_293371

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l293_293371


namespace initial_cargo_l293_293040

theorem initial_cargo (initial_cargo additional_cargo total_cargo : ℕ) 
  (h1 : additional_cargo = 8723) 
  (h2 : total_cargo = 14696) 
  (h3 : initial_cargo + additional_cargo = total_cargo) : 
  initial_cargo = 5973 := 
by 
  -- Start with the assumptions and directly obtain the calculation as required
  sorry

end initial_cargo_l293_293040


namespace sqrt_nine_eq_three_l293_293164

theorem sqrt_nine_eq_three : Real.sqrt 9 = 3 :=
by
  sorry

end sqrt_nine_eq_three_l293_293164


namespace find_measure_angle_AOD_l293_293829

-- Definitions of angles in the problem
def angle_COA := 150
def angle_BOD := 120

-- Definition of the relationship between angles
def angle_AOD_eq_four_times_angle_BOC (x : ℝ) : Prop :=
  4 * x = 360

-- Proof Problem Lean Statement
theorem find_measure_angle_AOD (x : ℝ) (h1 : 180 - 30 = angle_COA) (h2 : 180 - 60 = angle_BOD) (h3 : angle_AOD_eq_four_times_angle_BOC x) : 
  4 * x = 360 :=
  by 
  -- Insert necessary steps here
  sorry

end find_measure_angle_AOD_l293_293829


namespace greatest_visible_unit_cubes_from_single_point_l293_293925

-- Define the size of the cube
def cube_size : ℕ := 9

-- The total number of unit cubes in the 9x9x9 cube
def total_unit_cubes (n : ℕ) : ℕ := n^3

-- The greatest number of unit cubes visible from a single point
def visible_unit_cubes (n : ℕ) : ℕ := 3 * n^2 - 3 * (n - 1) + 1

-- The given cube size is 9
def given_cube_size : ℕ := cube_size

-- The correct answer for the greatest number of visible unit cubes from a single point
def correct_visible_cubes : ℕ := 220

-- Theorem stating the visibility calculation for a 9x9x9 cube
theorem greatest_visible_unit_cubes_from_single_point :
  visible_unit_cubes cube_size = correct_visible_cubes := by
  sorry

end greatest_visible_unit_cubes_from_single_point_l293_293925


namespace conical_pile_volume_l293_293912

noncomputable def volume_of_cone (d : ℝ) (h : ℝ) : ℝ :=
  (Real.pi * (d / 2) ^ 2 * h) / 3

theorem conical_pile_volume :
  let diameter := 10
  let height := 0.60 * diameter
  volume_of_cone diameter height = 50 * Real.pi :=
by
  sorry

end conical_pile_volume_l293_293912


namespace cities_with_fewer_than_200000_residents_l293_293869

def percentage_of_cities_with_fewer_than_50000 : ℕ := 20
def percentage_of_cities_with_50000_to_199999 : ℕ := 65

theorem cities_with_fewer_than_200000_residents :
  percentage_of_cities_with_fewer_than_50000 + percentage_of_cities_with_50000_to_199999 = 85 :=
by
  sorry

end cities_with_fewer_than_200000_residents_l293_293869


namespace d_not_unique_minimum_l293_293458

noncomputable def d (n : ℕ) (x : Fin n → ℝ) (t : ℝ) : ℝ :=
  (Finset.min' (Finset.univ.image (λ i => abs (x i - t))) sorry + 
  Finset.max' (Finset.univ.image (λ i => abs (x i - t))) sorry) / 2

theorem d_not_unique_minimum (n : ℕ) (x : Fin n → ℝ) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ d n x t1 = d n x t2 := sorry

end d_not_unique_minimum_l293_293458


namespace preimages_of_one_under_f_l293_293209

theorem preimages_of_one_under_f :
  {x : ℝ | (x^3 - x + 1 = 1)} = {-1, 0, 1} := by
  sorry

end preimages_of_one_under_f_l293_293209


namespace gcd_6724_13104_l293_293641

theorem gcd_6724_13104 : Int.gcd 6724 13104 = 8 := 
sorry

end gcd_6724_13104_l293_293641


namespace combined_savings_l293_293360

def salary_per_hour : ℝ := 10
def daily_hours : ℝ := 10
def weekly_days : ℝ := 5
def robby_saving_ratio : ℝ := 2 / 5
def jaylene_saving_ratio : ℝ := 3 / 5
def miranda_saving_ratio : ℝ := 1 / 2
def weeks : ℝ := 4

theorem combined_savings 
  (sph : ℝ := salary_per_hour)
  (dh : ℝ := daily_hours)
  (wd : ℝ := weekly_days)
  (rr : ℝ := robby_saving_ratio)
  (jr : ℝ := jaylene_saving_ratio)
  (mr : ℝ := miranda_saving_ratio)
  (wk : ℝ := weeks) :
  (rr * (wk * wd * (dh * sph)) + jr * (wk * wd * (dh * sph)) + mr * (wk * wd * (dh * sph))) = 3000 :=
by
  sorry

end combined_savings_l293_293360


namespace correct_addition_result_l293_293017

theorem correct_addition_result (x : ℚ) (h : x - 13/5 = 9/7) : x + 13/5 = 227/35 := 
by sorry

end correct_addition_result_l293_293017


namespace max_ab_condition_l293_293204

theorem max_ab_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 4 = 0)
  (line_check : ∀ x y : ℝ, (x = 1 ∧ y = -2) → 2*a*x - b*y - 2 = 0) : ab ≤ 1/4 :=
by
  sorry

end max_ab_condition_l293_293204


namespace time_to_meet_l293_293793

-- Definitions based on conditions
def motorboat_speed_Serezha : ℝ := 20 -- km/h
def crossing_time_Serezha : ℝ := 0.5 -- hours (30 minutes)
def running_speed_Dima : ℝ := 6 -- km/h
def running_time_Dima : ℝ := 0.25 -- hours (15 minutes)
def combined_speed : ℝ := running_speed_Dima + running_speed_Dima -- equal speeds running towards each other
def distance_meet : ℝ := (running_speed_Dima * running_time_Dima) -- The distance they need to cover towards each other

-- Prove the time for them to meet
theorem time_to_meet : (distance_meet / combined_speed) = (7.5 / 60) :=
by
  sorry

end time_to_meet_l293_293793


namespace production_increase_percentage_l293_293913

variable (T : ℝ) -- Initial production
variable (T1 T2 T5 : ℝ) -- Productions at different years
variable (x : ℝ) -- Unknown percentage increase for last three years

-- Conditions
def condition1 : Prop := T1 = T * 1.06
def condition2 : Prop := T2 = T1 * 1.08
def condition3 : Prop := T5 = T * (1.1 ^ 5)

-- Statement to prove
theorem production_increase_percentage :
  condition1 T T1 →
  condition2 T1 T2 →
  (T5 = T2 * (1 + x / 100) ^ 3) →
  x = 12.1 :=
by
  sorry

end production_increase_percentage_l293_293913


namespace Maggie_takes_75_percent_l293_293638

def Debby's_portion : ℚ := 0.25
def Maggie's_share : ℚ := 4500
def Total_amount : ℚ := 6000
def Maggie's_portion : ℚ := Maggie's_share / Total_amount

theorem Maggie_takes_75_percent : Maggie's_portion = 0.75 :=
by
  sorry

end Maggie_takes_75_percent_l293_293638


namespace find_treasure_island_l293_293108

-- Define the types for the three islands
inductive Island : Type
| A | B | C

-- Define the possible inhabitants of island A
inductive Inhabitant : Type
| Knight  -- always tells the truth
| Liar    -- always lies
| Normal  -- might tell the truth or lie

-- Define the conditions
def no_treasure_on_A : Prop := ¬ ∃ (x : Island), x = Island.A ∧ (x = Island.A)
def normal_people_on_A_two_treasures : Prop := ∀ (h : Inhabitant), h = Inhabitant.Normal → (∃ (x y : Island), x ≠ y ∧ (x ≠ Island.A ∧ y ≠ Island.A))

-- The question to ask
def question_to_ask (h : Inhabitant) : Prop :=
  (h = Inhabitant.Knight) ↔ (∃ (x : Island), (x = Island.B) ∧ (¬ ∃ (y : Island), (y = Island.A) ∧ (y = Island.A)))

-- The theorem statement
theorem find_treasure_island (inh : Inhabitant) :
  no_treasure_on_A ∧ normal_people_on_A_two_treasures →
  (question_to_ask inh → (∃ (x : Island), x = Island.B)) ∧ (¬ question_to_ask inh → (∃ (x : Island), x = Island.C)) :=
by
  intro h
  sorry

end find_treasure_island_l293_293108


namespace smaller_odd_number_l293_293960

theorem smaller_odd_number (n : ℤ) (h : n + (n + 2) = 48) : n = 23 :=
by
  sorry

end smaller_odd_number_l293_293960


namespace cos_180_eq_neg_one_l293_293597

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l293_293597


namespace hours_on_task2_l293_293276

theorem hours_on_task2
    (total_hours_per_week : ℕ) 
    (work_days_per_week : ℕ) 
    (hours_per_day_task1 : ℕ) 
    (hours_reduction_task1 : ℕ)
    (h_total_hours : total_hours_per_week = 40)
    (h_work_days : work_days_per_week = 5)
    (h_hours_task1 : hours_per_day_task1 = 5)
    (h_hours_reduction : hours_reduction_task1 = 5)
    : (total_hours_per_week / 2 / work_days_per_week) = 4 :=
by
  -- Skipping proof with sorry
  sorry

end hours_on_task2_l293_293276


namespace cost_of_gasoline_l293_293242

def odometer_initial : ℝ := 85120
def odometer_final : ℝ := 85150
def fuel_efficiency : ℝ := 30
def price_per_gallon : ℝ := 4.25

theorem cost_of_gasoline : 
  ((odometer_final - odometer_initial) / fuel_efficiency) * price_per_gallon = 4.25 := 
by 
  sorry

end cost_of_gasoline_l293_293242


namespace smallest_digit_divisible_by_9_l293_293961

theorem smallest_digit_divisible_by_9 :
  ∃ (d : ℕ), (25 + d) % 9 = 0 ∧ (∀ e : ℕ, (25 + e) % 9 = 0 → e ≥ d) :=
by
  sorry

end smallest_digit_divisible_by_9_l293_293961


namespace inverse_of_congruence_implies_equal_area_l293_293874

-- Definitions to capture conditions and relationships
def congruent_triangles (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with congruency of two triangles
  sorry

def equal_areas (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with equal areas of two triangles
  sorry

-- Statement to prove the inverse proposition
theorem inverse_of_congruence_implies_equal_area :
  (∀ T1 T2 : Triangle, congruent_triangles T1 T2 → equal_areas T1 T2) →
  (∀ T1 T2 : Triangle, equal_areas T1 T2 → congruent_triangles T1 T2) :=
  sorry

end inverse_of_congruence_implies_equal_area_l293_293874


namespace greatest_possible_length_l293_293578

theorem greatest_possible_length (a b c : ℕ) (h1 : a = 28) (h2 : b = 45) (h3 : c = 63) : 
  Nat.gcd (Nat.gcd a b) c = 7 :=
by
  sorry

end greatest_possible_length_l293_293578


namespace total_hours_over_two_weeks_l293_293933

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l293_293933


namespace binomial_expansion_five_l293_293803

open Finset

theorem binomial_expansion_five (a b : ℝ) : 
  (a + b)^5 = a^5 + 5 * a^4 * b + 10 * a^3 * b^2 + 10 * a^2 * b^3 + 5 * a * b^4 + b^5 := 
by sorry

end binomial_expansion_five_l293_293803


namespace cos_180_eq_neg1_l293_293636

theorem cos_180_eq_neg1 : real.cos (real.pi) = -1 := by
  sorry

end cos_180_eq_neg1_l293_293636


namespace red_marbles_count_l293_293420

theorem red_marbles_count (R : ℕ) (h1 : 48 - R > 0) (h2 : ((48 - R) / 48 : ℚ) * ((48 - R) / 48) = 9 / 16) : R = 12 :=
sorry

end red_marbles_count_l293_293420


namespace eval_frac_equal_two_l293_293684

noncomputable def eval_frac (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : ℂ :=
  (a^8 + b^8) / (a^2 + b^2)^4

theorem eval_frac_equal_two (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : eval_frac a b h1 h2 h3 = 2 :=
by {
  sorry
}

end eval_frac_equal_two_l293_293684


namespace cos_180_eq_minus_1_l293_293629

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l293_293629


namespace students_per_group_l293_293151

def total_students : ℕ := 30
def number_of_groups : ℕ := 6

theorem students_per_group :
  total_students / number_of_groups = 5 :=
by
  sorry

end students_per_group_l293_293151


namespace volleyballTeam_starters_l293_293124

noncomputable def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  let remainingPlayers := totalPlayers - quadruplets
  let chooseQuadruplet := quadruplets
  let chooseRemaining := Nat.choose remainingPlayers (starters - 1)
  chooseQuadruplet * chooseRemaining

theorem volleyballTeam_starters :
  chooseStarters 16 4 6 = 3168 :=
by
  sorry

end volleyballTeam_starters_l293_293124


namespace marys_income_percent_of_juans_income_l293_293162

variables (M T J : ℝ)

theorem marys_income_percent_of_juans_income (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marys_income_percent_of_juans_income_l293_293162


namespace second_differences_of_cubes_l293_293014

-- Define the first difference for cubes of consecutive natural numbers
def first_difference (n : ℕ) : ℕ :=
  ((n + 1) ^ 3) - (n ^ 3)

-- Define the second difference for the first differences
def second_difference (n : ℕ) : ℕ :=
  first_difference (n + 1) - first_difference n

-- Proof statement: Prove that second differences are equal to 6n + 6
theorem second_differences_of_cubes (n : ℕ) : second_difference n = 6 * n + 6 :=
  sorry

end second_differences_of_cubes_l293_293014


namespace sum_reciprocals_geom_seq_l293_293649

theorem sum_reciprocals_geom_seq (a₁ q : ℝ) (h_pos_a₁ : 0 < a₁) (h_pos_q : 0 < q)
    (h_sum : a₁ + a₁ * q + a₁ * q^2 + a₁ * q^3 = 9)
    (h_prod : a₁^4 * q^6 = 81 / 4) :
    (1 / a₁) + (1 / (a₁ * q)) + (1 / (a₁ * q^2)) + (1 / (a₁ * q^3)) = 2 :=
by
  sorry

end sum_reciprocals_geom_seq_l293_293649


namespace JaneReadingSpeed_l293_293109

theorem JaneReadingSpeed (total_pages read_second_half_speed total_days pages_first_half days_first_half_speed : ℕ)
  (h1 : total_pages = 500)
  (h2 : read_second_half_speed = 5)
  (h3 : total_days = 75)
  (h4 : pages_first_half = 250)
  (h5 : days_first_half_speed = pages_first_half / (total_days - (pages_first_half / read_second_half_speed))) :
  days_first_half_speed = 10 := by
  sorry

end JaneReadingSpeed_l293_293109


namespace find_Natisfy_condition_l293_293801

-- Define the original number
def N : Nat := 2173913043478260869565

-- Define the function to move the first digit of a number to the end
def move_first_digit_to_end (n : Nat) : Nat := sorry

-- The proof statement
theorem find_Natisfy_condition : 
  let new_num1 := N * 4
  let new_num2 := new_num1 / 5
  move_first_digit_to_end N = new_num2 
:=
  sorry

end find_Natisfy_condition_l293_293801


namespace sum_first_five_terms_arithmetic_sequence_l293_293393

theorem sum_first_five_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 5 * d = 10)
  (h2 : a + 6 * d = 15)
  (h3 : a + 7 * d = 20) :
  5 * (2 * a + (5 - 1) * d) / 2 = -25 := by
  sorry

end sum_first_five_terms_arithmetic_sequence_l293_293393


namespace ms_perez_class_total_students_l293_293726

/-- Half the students in Ms. Perez's class collected 12 cans each, two students didn't collect any cans,
    and the remaining 13 students collected 4 cans each. The total number of cans collected is 232. 
    Prove that the total number of students in Ms. Perez's class is 30. -/
theorem ms_perez_class_total_students (S : ℕ) :
  (S / 2) * 12 + 13 * 4 + 2 * 0 = 232 →
  S = S / 2 + 13 + 2 →
  S = 30 :=
by {
  sorry
}

end ms_perez_class_total_students_l293_293726


namespace smallest_positive_integer_l293_293548

theorem smallest_positive_integer :
  ∃ x : ℕ,
    x % 5 = 4 ∧
    x % 7 = 5 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    (∀ y : ℕ, (y % 5 = 4 ∧ y % 7 = 5 ∧ y % 11 = 9 ∧ y % 13 = 11) → y ≥ x) ∧ x = 999 :=
by
  sorry

end smallest_positive_integer_l293_293548


namespace cos_180_degrees_l293_293626

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l293_293626


namespace school_students_l293_293263

theorem school_students (T S : ℕ) (h1 : T = 6 * S - 78) (h2 : T - S = 2222) : T = 2682 :=
by
  sorry

end school_students_l293_293263


namespace contractor_total_amount_l293_293566

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l293_293566


namespace find_amount_l293_293956

theorem find_amount (amount : ℝ) (h : 0.25 * amount = 75) : amount = 300 :=
sorry

end find_amount_l293_293956


namespace grassy_plot_width_l293_293179

theorem grassy_plot_width (L : ℝ) (P : ℝ) (C : ℝ) (cost_per_sqm : ℝ) (W : ℝ) : 
  L = 110 →
  P = 2.5 →
  C = 510 →
  cost_per_sqm = 0.6 →
  (115 * (W + 5) - 110 * W = C / cost_per_sqm) →
  W = 55 :=
by
  intros hL hP hC hcost_per_sqm harea
  sorry

end grassy_plot_width_l293_293179


namespace evaluate_expression_l293_293018

theorem evaluate_expression :
  71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 72 + 70 * Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l293_293018


namespace tom_needs_more_boxes_l293_293000

theorem tom_needs_more_boxes
    (living_room_length : ℕ)
    (living_room_width : ℕ)
    (box_coverage : ℕ)
    (already_installed : ℕ) :
    living_room_length = 16 →
    living_room_width = 20 →
    box_coverage = 10 →
    already_installed = 250 →
    (living_room_length * living_room_width - already_installed) / box_coverage = 7 :=
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    sorry

end tom_needs_more_boxes_l293_293000


namespace simplified_value_l293_293015

-- Define the given expression
def expr := (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3)

-- State the theorem
theorem simplified_value : expr = 10^1.7 :=
by
  sorry -- Proof omitted

end simplified_value_l293_293015


namespace average_velocity_mass_flow_rate_available_horsepower_l293_293990

/-- Average velocity of water flowing out of the sluice gate. -/
theorem average_velocity (g h₁ h₂ : ℝ) (h1_5m : h₁ = 5) (h2_5_4m : h₂ = 5.4) (g_9_81 : g = 9.81) :
    (1 / 2) * (Real.sqrt (2 * g * h₁) + Real.sqrt (2 * g * h₂)) = 10.1 :=
by
  sorry

/-- Mass flow rate of water per second when given average velocity and opening dimensions. -/
theorem mass_flow_rate (v A : ℝ) (v_10_1 : v = 10.1) (A_0_6 : A = 0.4 * 1.5) (rho : ℝ) (rho_1000 : rho = 1000) :
    ρ * A * v = 6060 :=
by
  sorry

/-- Available horsepower through turbines given mass flow rate and average velocity. -/
theorem available_horsepower (m v : ℝ) (m_6060 : m = 6060) (v_10_1 : v = 10.1 ) (hp : ℝ)
    (hp_735_5 : hp = 735.5 ) :
    (1 / 2) * m * v^2 / hp = 420 :=
by
  sorry

end average_velocity_mass_flow_rate_available_horsepower_l293_293990


namespace fraction_sent_for_production_twice_l293_293051

variable {x : ℝ} (hx : x > 0)

theorem fraction_sent_for_production_twice :
  let initial_sulfur := (1.5 / 100 : ℝ)
  let first_sulfur_addition := (0.5 / 100 : ℝ)
  let second_sulfur_addition := (2 / 100 : ℝ) 
  (initial_sulfur - initial_sulfur * x + first_sulfur_addition * x -
    ((initial_sulfur - initial_sulfur * x + first_sulfur_addition * x) * x) + 
    second_sulfur_addition * x = initial_sulfur) → x = 1 / 2 :=
sorry

end fraction_sent_for_production_twice_l293_293051


namespace number_of_boys_l293_293764

theorem number_of_boys (B G : ℕ) 
    (h1 : B + G = 345) 
    (h2 : G = B + 69) : B = 138 :=
by
  sorry

end number_of_boys_l293_293764


namespace annular_region_area_l293_293547

noncomputable def area_annulus (r1 r2 : ℝ) : ℝ :=
  (Real.pi * r2 ^ 2) - (Real.pi * r1 ^ 2)

theorem annular_region_area :
  area_annulus 4 7 = 33 * Real.pi :=
by 
  sorry

end annular_region_area_l293_293547


namespace proposition_D_l293_293107

-- Definitions extracted from the conditions
variables {a b : ℝ} (c d : ℝ)

-- Proposition D to be proven
theorem proposition_D (ha : a < b) (hb : b < 0) : a^2 > b^2 := sorry

end proposition_D_l293_293107


namespace cos_180_eq_neg1_l293_293619

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end cos_180_eq_neg1_l293_293619


namespace solve_for_x_l293_293103

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end solve_for_x_l293_293103


namespace three_digit_sum_of_factorials_l293_293751

theorem three_digit_sum_of_factorials : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n = 145) ∧ 
  (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ 
    1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ 1 ≤ d3 ∧ d3 < 10 ∧ 
    (d1 * d1.factorial + d2 * d2.factorial + d3 * d3.factorial = n)) :=
  by
  sorry

end three_digit_sum_of_factorials_l293_293751


namespace min_value_x_squared_plus_6x_l293_293893

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l293_293893


namespace teresa_total_marks_l293_293719

theorem teresa_total_marks :
  let science_marks := 70
  let music_marks := 80
  let social_studies_marks := 85
  let physics_marks := 1 / 2 * music_marks
  science_marks + music_marks + social_studies_marks + physics_marks = 275 :=
by
  sorry

end teresa_total_marks_l293_293719


namespace find_m_l293_293377

-- Conditions given
def ellipse (x y m : ℝ) : Prop := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- The theorem to prove
theorem find_m (m : ℝ) (h₁ : ellipse 1 1 m) (h₂ : eccentricity 2) : m = 3 ∨ m = 5 :=
  sorry

end find_m_l293_293377


namespace problem_inequality_l293_293809

theorem problem_inequality (a b : ℝ) (hab : 1 / a + 1 / b = 1) : 
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
by
  sorry

end problem_inequality_l293_293809


namespace ratio_surfer_malibu_santa_monica_l293_293265

theorem ratio_surfer_malibu_santa_monica (M S : ℕ) (hS : S = 20) (hTotal : M + S = 60) : M / S = 2 :=
by 
  sorry

end ratio_surfer_malibu_santa_monica_l293_293265


namespace combined_mpg_l293_293697

theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℕ) 
  (h1 : ray_mpg = 50) (h2 : tom_mpg = 8) 
  (h3 : ray_miles = 100) (h4 : tom_miles = 200) : 
  (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 100 / 9 :=
by
  sorry

end combined_mpg_l293_293697


namespace choose_15_3_eq_455_l293_293091

theorem choose_15_3_eq_455 : Nat.choose 15 3 = 455 := by
  sorry

end choose_15_3_eq_455_l293_293091


namespace range_of_x_l293_293325

theorem range_of_x
  (x : ℝ)
  (h1 : ∀ m, -1 ≤ m ∧ m ≤ 4 → m * (x^2 - 1) - 1 - 8 * x < 0) :
  0 < x ∧ x < 5 / 2 :=
sorry

end range_of_x_l293_293325


namespace trigonometric_eq_solution_count_l293_293523

theorem trigonometric_eq_solution_count :
  ∃ B : Finset ℤ, B.card = 250 ∧ ∀ x ∈ B, 2000 ≤ x ∧ x ≤ 3000 ∧ 
  2 * Real.sqrt 2 * Real.sin (Real.pi * x / 4)^3 = Real.sin (Real.pi / 4 * (1 + x)) :=
sorry

end trigonometric_eq_solution_count_l293_293523


namespace like_terms_implies_m_minus_n_l293_293096

/-- If 4x^(2m+2)y^(n-1) and -3x^(3m+1)y^(3n-5) are like terms, then m - n = -1. -/
theorem like_terms_implies_m_minus_n
  (m n : ℤ)
  (h1 : 2 * m + 2 = 3 * m + 1)
  (h2 : n - 1 = 3 * n - 5) :
  m - n = -1 :=
by
  sorry

end like_terms_implies_m_minus_n_l293_293096


namespace find_f_2012_l293_293659

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

theorem find_f_2012 (a b : ℝ) (h : f (1 / 2012) a b = 5) : f 2012 a b = -1 :=
by
  sorry

end find_f_2012_l293_293659


namespace find_principal_l293_293411

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end find_principal_l293_293411


namespace tiles_cover_the_floor_l293_293771

theorem tiles_cover_the_floor
  (n : ℕ)
  (h : 2 * n - 1 = 101)
  : n ^ 2 = 2601 := sorry

end tiles_cover_the_floor_l293_293771


namespace good_function_count_l293_293119

theorem good_function_count :
  let p := 2017
  let F_p := Zmod p
  ∃ (α : F_p), (α ≠ 0) →
  (∀ x y : ℤ, (f(x) * f(y) = f(x + y) + (α ^ y) * f(x - y))) →
  (∃ f : ℤ → F_p, f(0) = 2 ∧ f (n + 2016) = f(n)) →
  ∃ n : ℕ, n = 1327392 :=
sorry

end good_function_count_l293_293119


namespace initial_amount_of_A_l293_293292

variable (a b c : ℕ)

-- Conditions
axiom condition1 : a - b - c = 32
axiom condition2 : b + c = 48
axiom condition3 : a + b + c = 128

-- The goal is to prove that A had 80 cents initially.
theorem initial_amount_of_A : a = 80 :=
by
  -- We need to skip the proof here
  sorry

end initial_amount_of_A_l293_293292


namespace simplify_fraction_l293_293703

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l293_293703


namespace solve_eq_l293_293378

theorem solve_eq (x a b : ℝ) (h₁ : x^2 + 10 * x = 34) (h₂ : a = 59) (h₃ : b = 5) :
  a + b = 64 :=
by {
  -- insert proof here, eventually leading to a + b = 64
  sorry
}

end solve_eq_l293_293378


namespace roots_of_polynomial_l293_293128

theorem roots_of_polynomial :
  ∀ x : ℝ, x * (x + 2)^2 * (3 - x) * (5 + x) = 0 ↔ (x = 0 ∨ x = -2 ∨ x = 3 ∨ x = -5) :=
by
  sorry

end roots_of_polynomial_l293_293128


namespace inequality_has_exactly_one_solution_l293_293195

-- Definitions based on the conditions
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3 * a

-- The main theorem that encodes the proof problem
theorem inequality_has_exactly_one_solution (a : ℝ) : 
  (∃! x : ℝ, |f x a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end inequality_has_exactly_one_solution_l293_293195


namespace number_of_stones_l293_293277

theorem number_of_stones (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (stone_breadth_dm : ℕ)
  (hall_length_dm_eq : hall_length_m * 10 = 360)
  (hall_breadth_dm_eq : hall_breadth_m * 10 = 150)
  (stone_length_eq : stone_length_dm = 6)
  (stone_breadth_eq : stone_breadth_dm = 5) :
  ((hall_length_m * 10) * (hall_breadth_m * 10)) / (stone_length_dm * stone_breadth_dm) = 1800 :=
by
  sorry

end number_of_stones_l293_293277


namespace geometric_sequence_common_ratio_l293_293203

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_cond : (a 0 * (1 + q + q^2)) / (a 0 * q^2) = 3) : q = 1 :=
by
  sorry

end geometric_sequence_common_ratio_l293_293203


namespace total_sample_variance_l293_293222

/-- In a survey of the heights (in cm) of high school students at Shuren High School:

 - 20 boys were selected with an average height of 174 cm and a variance of 12.
 - 30 girls were selected with an average height of 164 cm and a variance of 30.

We need to prove that the variance of the total sample is 46.8. -/
theorem total_sample_variance :
  let boys_count := 20
  let girls_count := 30
  let boys_avg := 174
  let girls_avg := 164
  let boys_var := 12
  let girls_var := 30
  let total_count := boys_count + girls_count
  let overall_avg := (boys_avg * boys_count + girls_avg * girls_count) / total_count
  let total_var := 
    (boys_count * (boys_var + (boys_avg - overall_avg)^2) / total_count)
    + (girls_count * (girls_var + (girls_avg - overall_avg)^2) / total_count)
  total_var = 46.8 := by
    sorry

end total_sample_variance_l293_293222


namespace no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l293_293898

theorem no_solution_for_x_y_z (a : ℕ) : 
  ¬ ∃ (x y z : ℚ), x^2 + y^2 + z^2 = 8 * a + 7 :=
by
  sorry

theorem seven_n_plus_eight_is_perfect_square (n : ℕ) :
  ∃ x : ℕ, 7^n + 8 = x^2 ↔ n = 0 :=
by
  sorry

end no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l293_293898


namespace right_triangle_divisibility_l293_293686

theorem right_triangle_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (a % 3 = 0 ∨ b % 3 = 0) ∧ (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) :=
by
  -- skipping the proof
  sorry

end right_triangle_divisibility_l293_293686


namespace man_speed_in_still_water_l293_293177

theorem man_speed_in_still_water (c_speed : ℝ) (distance_m : ℝ) (time_sec : ℝ) (downstream_distance_km : ℝ) (downstream_time_hr : ℝ) :
    c_speed = 3 →
    distance_m = 15 →
    time_sec = 2.9997600191984644 →
    downstream_distance_km = distance_m / 1000 →
    downstream_time_hr = time_sec / 3600 →
    (downstream_distance_km / downstream_time_hr) - c_speed = 15 :=
by
  intros hc hd ht hdownstream_distance hdownstream_time 
  sorry

end man_speed_in_still_water_l293_293177


namespace probability_of_two_boys_given_one_boy_l293_293582

-- Define the events and probabilities
def P_BB : ℚ := 1/4
def P_BG : ℚ := 1/4
def P_GB : ℚ := 1/4
def P_GG : ℚ := 1/4

def P_at_least_one_boy : ℚ := 1 - P_GG

def P_two_boys_given_at_least_one_boy : ℚ := P_BB / P_at_least_one_boy

-- Statement to be proven
theorem probability_of_two_boys_given_one_boy : P_two_boys_given_at_least_one_boy = 1/3 :=
by sorry

end probability_of_two_boys_given_one_boy_l293_293582


namespace product_of_integers_cubes_sum_to_35_l293_293412

-- Define the conditions
def integers_sum_of_cubes (a b : ℤ) : Prop :=
  a^3 + b^3 = 35

-- Define the theorem that the product of integers whose cubes sum to 35 is 6
theorem product_of_integers_cubes_sum_to_35 :
  ∃ a b : ℤ, integers_sum_of_cubes a b ∧ a * b = 6 :=
by
  sorry

end product_of_integers_cubes_sum_to_35_l293_293412


namespace combined_savings_l293_293361

def salary_per_hour : ℝ := 10
def daily_hours : ℝ := 10
def weekly_days : ℝ := 5
def robby_saving_ratio : ℝ := 2 / 5
def jaylene_saving_ratio : ℝ := 3 / 5
def miranda_saving_ratio : ℝ := 1 / 2
def weeks : ℝ := 4

theorem combined_savings 
  (sph : ℝ := salary_per_hour)
  (dh : ℝ := daily_hours)
  (wd : ℝ := weekly_days)
  (rr : ℝ := robby_saving_ratio)
  (jr : ℝ := jaylene_saving_ratio)
  (mr : ℝ := miranda_saving_ratio)
  (wk : ℝ := weeks) :
  (rr * (wk * wd * (dh * sph)) + jr * (wk * wd * (dh * sph)) + mr * (wk * wd * (dh * sph))) = 3000 :=
by
  sorry

end combined_savings_l293_293361


namespace solve_for_n_l293_293946

theorem solve_for_n (n : ℤ) (h : (5/4 : ℚ) * n + (5/4 : ℚ) = n) : n = -5 := by
    sorry

end solve_for_n_l293_293946


namespace no_such_real_x_exists_l293_293698

theorem no_such_real_x_exists :
  ¬ ∃ (x : ℝ), ⌊ x ⌋ + ⌊ 2 * x ⌋ + ⌊ 4 * x ⌋ + ⌊ 8 * x ⌋ + ⌊ 16 * x ⌋ + ⌊ 32 * x ⌋ = 12345 := 
sorry

end no_such_real_x_exists_l293_293698


namespace symmetric_slope_angle_l293_293467

-- Define the problem conditions in Lean
def slope_angle (θ : Real) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Statement of the theorem in Lean
theorem symmetric_slope_angle (θ : Real) (h : slope_angle θ) :
  θ = 0 ∨ θ = Real.pi - θ :=
sorry

end symmetric_slope_angle_l293_293467


namespace positive_three_digit_integers_divisible_by_12_and_7_l293_293475

theorem positive_three_digit_integers_divisible_by_12_and_7 : 
  ∃ n : ℕ, n = 11 ∧ ∀ k : ℕ, (k ∣ 12) ∧ (k ∣ 7) ∧ (100 ≤ k) ∧ (k < 1000) :=
by
  sorry

end positive_three_digit_integers_divisible_by_12_and_7_l293_293475


namespace total_fires_l293_293929

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end total_fires_l293_293929


namespace problem1_problem2_l293_293973

-- Problem 1
theorem problem1 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5 / 3 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) (h_quad : π < α ∧ α < 3 * π / 2) :
  Real.cos (-π + α) + Real.cos (π / 2 + α) = 3 * Real.sqrt 5 / 5 :=
by sorry

end problem1_problem2_l293_293973


namespace find_x_value_l293_293060

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l293_293060


namespace positive_difference_l293_293146

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l293_293146


namespace four_digit_integer_existence_l293_293721

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end four_digit_integer_existence_l293_293721


namespace Theresa_game_scores_l293_293674

theorem Theresa_game_scores 
  (h_sum_10 : 9 + 5 + 4 + 7 + 6 + 2 + 4 + 8 + 3 + 7 = 55)
  (h_p11 : ∀ p11 : ℕ, p11 < 10 → (55 + p11) % 11 = 0)
  (h_p12 : ∀ p11 p12 : ℕ, p11 < 10 → p12 < 10 → ((55 + p11 + p12) % 12 = 0)) :
  ∃ p11 p12 : ℕ, p11 < 10 ∧ p12 < 10 ∧ (55 + p11) % 11 = 0 ∧ (55 + p11 + p12) % 12 = 0 ∧ p11 * p12 = 0 :=
by
  sorry

end Theresa_game_scores_l293_293674


namespace total_dogs_l293_293789

def number_of_boxes : ℕ := 15
def dogs_per_box : ℕ := 8

theorem total_dogs : number_of_boxes * dogs_per_box = 120 := by
  sorry

end total_dogs_l293_293789


namespace charity_event_assignment_l293_293200

theorem charity_event_assignment (students : Finset ℕ) (h_students : students.card = 5) :
  ∃ (num_ways : ℕ), num_ways = 60 :=
by
  let select_two_for_friday := Nat.choose 5 2
  let remaining_students_after_friday := 5 - 2
  let select_one_for_saturday := Nat.choose remaining_students_after_friday 1
  let remaining_students_after_saturday := remaining_students_after_friday - 1
  let select_one_for_sunday := Nat.choose remaining_students_after_saturday 1
  let total_ways := select_two_for_friday * select_one_for_saturday * select_one_for_sunday
  use total_ways
  sorry

end charity_event_assignment_l293_293200


namespace problem1_solution_set_problem2_range_of_a_l293_293974

section Problem1

def f1 (x : ℝ) : ℝ := |x - 4| + |x - 2|

theorem problem1_solution_set (a : ℝ) (h : a = 2) :
  { x : ℝ | f1 x > 10 } = { x : ℝ | x > 8 ∨ x < -2 } := sorry

end Problem1


section Problem2

def f2 (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem problem2_range_of_a (f_geq : ∀ x : ℝ, f2 x a ≥ 1) :
  a ≥ 5 ∨ a ≤ 3 := sorry

end Problem2

end problem1_solution_set_problem2_range_of_a_l293_293974


namespace trip_time_total_l293_293110

noncomputable def wrong_direction_time : ℝ := 75 / 60
noncomputable def return_time : ℝ := 75 / 45
noncomputable def normal_trip_time : ℝ := 250 / 45

theorem trip_time_total :
  wrong_direction_time + return_time + normal_trip_time = 8.48 := by
  sorry

end trip_time_total_l293_293110


namespace arithmetic_sequence_general_term_sum_of_inverse_l293_293318

noncomputable def a_n (n : ℕ) : ℝ := 4 * n + 1

noncomputable def S_n (n : ℕ) : ℝ := n * (a_n 1 + a_n n) / 2

noncomputable def T_n (n : ℕ) : ℝ := ∑ k in Finset.range n, (1 / (S_n (k + 1) - (k + 1)))

theorem arithmetic_sequence_general_term (n : ℕ) : a_n n = 4 * n + 1 := 
by sorry

theorem sum_of_inverse (n : ℕ) : T_n n = n / (2 * (n + 1)) := 
by sorry

end arithmetic_sequence_general_term_sum_of_inverse_l293_293318


namespace solve_for_C_l293_293194

-- Given constants and assumptions
def SumOfDigitsFirst (A B : ℕ) := 8 + 4 + A + 5 + 3 + B + 2 + 1
def SumOfDigitsSecond (A B C : ℕ) := 5 + 2 + 7 + A + B + 6 + 0 + C

theorem solve_for_C (A B C : ℕ) 
  (h1 : (SumOfDigitsFirst A B % 9) = 0)
  (h2 : (SumOfDigitsSecond A B C % 9) = 0) 
  : C = 3 :=
sorry

end solve_for_C_l293_293194


namespace single_point_graph_value_of_d_l293_293527

theorem single_point_graph_value_of_d (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + d = 0 → x = -2 ∧ y = 3) ↔ d = 21 := 
by 
  sorry

end single_point_graph_value_of_d_l293_293527


namespace slices_leftover_l293_293184

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end slices_leftover_l293_293184


namespace arcsin_one_half_l293_293786

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l293_293786


namespace total_pay_XY_l293_293266

-- Assuming X's pay is 120% of Y's pay and Y's pay is 268.1818181818182,
-- Prove that the total pay to X and Y is 590.00.
theorem total_pay_XY (Y_pay : ℝ) (X_pay : ℝ) (total_pay : ℝ) :
  Y_pay = 268.1818181818182 →
  X_pay = 1.2 * Y_pay →
  total_pay = X_pay + Y_pay →
  total_pay = 590.00 :=
by
  intros hY hX hT
  sorry

end total_pay_XY_l293_293266


namespace problem1_simplified_problem2_simplified_l293_293417

-- Definition and statement for the first problem
def problem1_expression (x y : ℝ) : ℝ := 
  -3 * x * y - 3 * x^2 + 4 * x * y + 2 * x^2

theorem problem1_simplified (x y : ℝ) : 
  problem1_expression x y = x * y - x^2 := 
by
  sorry

-- Definition and statement for the second problem
def problem2_expression (a b : ℝ) : ℝ := 
  3 * (a^2 - 2 * a * b) - 5 * (a^2 + 4 * a * b)

theorem problem2_simplified (a b : ℝ) : 
  problem2_expression a b = -2 * a^2 - 26 * a * b :=
by
  sorry

end problem1_simplified_problem2_simplified_l293_293417


namespace correct_cd_value_l293_293405

noncomputable def repeating_decimal (c d : ℕ) : ℝ :=
  1 + c / 10.0 + d / 100.0 + (c * 10 + d) / 990.0

theorem correct_cd_value (c d : ℕ) (h : (c = 9) ∧ (d = 9)) : 90 * (repeating_decimal 9 9 - (1 + 9 / 10.0 + 9 / 100.0)) = 0.9 :=
by
  sorry

end correct_cd_value_l293_293405


namespace races_to_champion_l293_293334

theorem races_to_champion (num_sprinters : ℕ) (sprinters_per_race : ℕ) (advancing_per_race : ℕ)
  (eliminated_per_race : ℕ) (initial_races : ℕ) (total_races : ℕ):
  num_sprinters = 360 ∧ sprinters_per_race = 8 ∧ advancing_per_race = 2 ∧ 
  eliminated_per_race = 6 ∧ initial_races = 45 ∧ total_races = 62 →
  initial_races + (initial_races / sprinters_per_race +
  ((initial_races / sprinters_per_race) / sprinters_per_race +
  (((initial_races / sprinters_per_race) / sprinters_per_race) / sprinters_per_race + 1))) = total_races :=
sorry

end races_to_champion_l293_293334


namespace total_marks_is_275_l293_293716

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end total_marks_is_275_l293_293716


namespace mistaken_multiplication_l293_293284

theorem mistaken_multiplication (x : ℕ) : 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  (a * b - a * x = incorrect_result) ↔ (x = 34) := 
by 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  sorry

end mistaken_multiplication_l293_293284


namespace equidistant_point_l293_293767

theorem equidistant_point (x y : ℝ) :
  (abs x = abs y) → (abs x = abs (x + y - 3) / (Real.sqrt 2)) → x = 1.5 :=
by {
  -- proof omitted
  sorry
}

end equidistant_point_l293_293767


namespace cos_180_eq_minus_1_l293_293630

theorem cos_180_eq_minus_1 :
  ∃ (p : ℝ × ℝ), p = (1, 0) ∧ (rotate_point p 180 = (-1, 0)) ∧ (cos_of_point (rotate_point p 180) = -1) := 
sorry

-- Additional definitions required corresponding to the conditions:
noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ := 
  (-- implementation of rotation, which is rotated x and y coordinate).
  sorry

noncomputable def cos_of_point (p : ℝ × ℝ) : ℝ :=
  p.1 -- x-coordinate of point p
  sorry

end cos_180_eq_minus_1_l293_293630


namespace loss_per_meter_is_five_l293_293041

def cost_price_per_meter : ℝ := 50
def total_meters_sold : ℝ := 400
def selling_price : ℝ := 18000

noncomputable def total_cost_price : ℝ := cost_price_per_meter * total_meters_sold
noncomputable def total_loss : ℝ := total_cost_price - selling_price
noncomputable def loss_per_meter : ℝ := total_loss / total_meters_sold

theorem loss_per_meter_is_five : loss_per_meter = 5 :=
by sorry

end loss_per_meter_is_five_l293_293041


namespace total_frisbees_l293_293429

-- Let x be the number of $3 frisbees and y be the number of $4 frisbees.
variables (x y : ℕ)

-- Condition 1: Total sales amount is 200 dollars.
def condition1 : Prop := 3 * x + 4 * y = 200

-- Condition 2: At least 8 $4 frisbees were sold.
def condition2 : Prop := y >= 8

-- Prove that the total number of frisbees sold is 64.
theorem total_frisbees (h1 : condition1 x y) (h2 : condition2 y) : x + y = 64 :=
by
  sorry

end total_frisbees_l293_293429


namespace total_local_percentage_approx_52_74_l293_293900

-- We provide the conditions as definitions
def total_arts_students : ℕ := 400
def local_arts_percentage : ℝ := 0.50
def total_science_students : ℕ := 100
def local_science_percentage : ℝ := 0.25
def total_commerce_students : ℕ := 120
def local_commerce_percentage : ℝ := 0.85

-- Calculate the expected total percentage of local students
noncomputable def calculated_total_local_percentage : ℝ :=
  let local_arts_students := local_arts_percentage * total_arts_students
  let local_science_students := local_science_percentage * total_science_students
  let local_commerce_students := local_commerce_percentage * total_commerce_students
  let total_local_students := local_arts_students + local_science_students + local_commerce_students
  let total_students := total_arts_students + total_science_students + total_commerce_students
  (total_local_students / total_students) * 100

-- State what we need to prove
theorem total_local_percentage_approx_52_74 :
  abs (calculated_total_local_percentage - 52.74) < 1 :=
sorry

end total_local_percentage_approx_52_74_l293_293900


namespace cubic_polynomial_solution_l293_293424

theorem cubic_polynomial_solution 
  (p : ℚ → ℚ) 
  (h1 : p 1 = 1)
  (h2 : p 2 = 1 / 4)
  (h3 : p 3 = 1 / 9)
  (h4 : p 4 = 1 / 16)
  (h6 : p 6 = 1 / 36)
  (h0 : p 0 = -1 / 25) : 
  p 5 = 20668 / 216000 :=
sorry

end cubic_polynomial_solution_l293_293424


namespace sqrt_square_l293_293163

theorem sqrt_square (x : ℝ) (h_nonneg : 0 ≤ x) : (Real.sqrt x)^2 = x :=
by
  sorry

example : (Real.sqrt 25)^2 = 25 :=
by
  exact sqrt_square 25 (by norm_num)

end sqrt_square_l293_293163


namespace cost_price_of_watch_l293_293552

theorem cost_price_of_watch 
  (CP : ℝ)
  (h1 : 0.88 * CP = SP_loss)
  (h2 : 1.04 * CP = SP_gain)
  (h3 : SP_gain - SP_loss = 140) :
  CP = 875 := 
sorry

end cost_price_of_watch_l293_293552


namespace no_positive_integer_solutions_l293_293949

theorem no_positive_integer_solutions :
  ∀ (A : ℕ), 1 ≤ A ∧ A ≤ 9 → ¬∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = A * 10 + A ∧ x + y = 10 * A + 1 := by
  sorry

end no_positive_integer_solutions_l293_293949


namespace part1_part2_l293_293088

-- Define the universal set R
def R := ℝ

-- Define set A
def A (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0

-- Define set B parameterized by a
def B (x a : ℝ) : Prop := (x - (a + 5)) / (x - a) > 0

-- Prove (1): A ∩ B when a = -2
theorem part1 : { x : ℝ | A x } ∩ { x : ℝ | B x (-2) } = { x : ℝ | 3 < x ∧ x ≤ 4 } :=
by
  sorry

-- Prove (2): The range of a such that A ⊆ B
theorem part2 : { a : ℝ | ∀ x, A x → B x a } = { a : ℝ | a < -6 ∨ a > 4 } :=
by
  sorry

end part1_part2_l293_293088


namespace total_cost_after_discounts_l293_293289

-- Definition of the cost function with applicable discounts
def pencil_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

def pen_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

-- The statement to be proved
theorem total_cost_after_discounts :
  let pencil_price := 2.50
  let pen_price := 3.50
  let pencil_count := 38
  let pen_count := 56
  let pencil_discount_threshold := 30
  let pencil_discount_rate := 0.10
  let pen_discount_threshold := 50
  let pen_discount_rate := 0.15
  let total_cost := pencil_cost pencil_price pencil_count pencil_discount_threshold pencil_discount_rate
                   + pen_cost pen_price pen_count pen_discount_threshold pen_discount_rate
  total_cost = 252.10 := 
by 
  sorry

end total_cost_after_discounts_l293_293289


namespace card_selection_l293_293823

noncomputable def count_ways := 438400

theorem card_selection :
  let decks := 2
  let total_cards := 52 * decks
  let suits := 4
  let non_royal_count := 10 * decks
  let royal_count := 3 * decks
  let non_royal_options := non_royal_count * decks
  let royal_options := royal_count * decks
  1 * (non_royal_options)^4 + (suits.choose 1) * royal_options * (non_royal_options)^3 + (suits.choose 2) * (royal_options)^2 * (non_royal_options)^2 = count_ways :=
sorry

end card_selection_l293_293823


namespace C_alone_work_days_l293_293558

theorem C_alone_work_days (A_work_days B_work_days combined_work_days : ℝ) 
  (A_work_rate B_work_rate C_work_rate combined_work_rate : ℝ)
  (hA : A_work_days = 6)
  (hB : B_work_days = 5)
  (hCombined : combined_work_days = 2)
  (hA_work_rate : A_work_rate = 1 / A_work_days)
  (hB_work_rate : B_work_rate = 1 / B_work_days)
  (hCombined_work_rate : combined_work_rate = 1 / combined_work_days)
  (work_rate_eq : A_work_rate + B_work_rate + C_work_rate = combined_work_rate):
  (1 / C_work_rate) = 7.5 :=
by
  sorry

end C_alone_work_days_l293_293558


namespace cos_pi_div_four_minus_alpha_l293_293317

theorem cos_pi_div_four_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) : 
    Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 :=
sorry

end cos_pi_div_four_minus_alpha_l293_293317


namespace sin_minus_cos_eq_sqrt2_l293_293066

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l293_293066


namespace find_last_two_digits_l293_293487

noncomputable def tenth_digit (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ) : ℕ :=
d7 + d8

noncomputable def ninth_digit (d1 d2 d3 d4 d5 d6 d7 : ℕ) : ℕ :=
d6 + d7

theorem find_last_two_digits :
  ∃ d9 d10 : ℕ, d9 = ninth_digit 1 1 2 3 5 8 13 ∧ d10 = tenth_digit 1 1 2 3 5 8 13 21 :=
by
  sorry

end find_last_two_digits_l293_293487


namespace pq_solution_l293_293871

theorem pq_solution :
  ∃ (p q : ℤ), (20 * x ^ 2 - 110 * x - 120 = (5 * x + p) * (4 * x + q))
    ∧ (5 * q + 4 * p = -110) ∧ (p * q = -120)
    ∧ (p + 2 * q = -8) :=
by
  sorry

end pq_solution_l293_293871


namespace polynomial_identity_l293_293905

theorem polynomial_identity
  (z1 z2 : ℂ)
  (h1 : z1 + z2 = -6)
  (h2 : z1 * z2 = 11)
  : (1 + z1^2 * z2) * (1 + z1 * z2^2) = 1266 := 
by 
  sorry

end polynomial_identity_l293_293905


namespace cos_180_degrees_l293_293625

theorem cos_180_degrees :
  real.cos (real.pi) = -1 :=
by
  sorry

end cos_180_degrees_l293_293625


namespace number_of_weavers_l293_293525

theorem number_of_weavers (W : ℕ) 
  (h1 : ∀ t : ℕ, t = 4 → 4 = W * (1 * t)) 
  (h2 : ∀ t : ℕ, t = 16 → 64 = 16 * (1 / (W:ℝ) * t)) : 
  W = 4 := 
by {
  sorry
}

end number_of_weavers_l293_293525


namespace problem_A_problem_B_problem_C_problem_D_l293_293660

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

theorem problem_A : ∀ x: ℝ, 0 < x ∧ x < 1 → f x < 0 := 
by sorry

theorem problem_B : ∃! (x : ℝ), ∃ c : ℝ, deriv f x = 0 := 
by sorry

theorem problem_C : ∀ (x : ℝ), ∃ c : ℝ, deriv f x = 0 → ¬∃ d : ℝ, d ≠ c ∧ deriv f d = 0 := 
by sorry

theorem problem_D : ¬ ∃ x₀ : ℝ, f x₀ = 1 / Real.exp 1 := 
by sorry

end problem_A_problem_B_problem_C_problem_D_l293_293660


namespace min_value_x_squared_plus_6x_l293_293895

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l293_293895


namespace sum_of_interior_edges_l293_293288

noncomputable def interior_edge_sum (outer_length : ℝ) (wood_width : ℝ) (frame_area : ℝ) : ℝ := 
  let outer_width := (frame_area + 3 * (outer_length - 2 * wood_width) * 4) / outer_length
  let inner_length := outer_length - 2 * wood_width
  let inner_width := outer_width - 2 * wood_width
  2 * inner_length + 2 * inner_width

theorem sum_of_interior_edges :
  interior_edge_sum 7 2 34 = 9 := by
  sorry

end sum_of_interior_edges_l293_293288


namespace rectangle_area_l293_293428

def radius : ℝ := 10
def width : ℝ := 2 * radius
def length : ℝ := 3 * width
def area_of_rectangle : ℝ := length * width

theorem rectangle_area : area_of_rectangle = 1200 :=
  by sorry

end rectangle_area_l293_293428


namespace product_of_prs_eq_60_l293_293330

theorem product_of_prs_eq_60 (p r s : ℕ) (h1 : 3 ^ p + 3 ^ 5 = 270) (h2 : 2 ^ r + 46 = 94) (h3 : 6 ^ s + 5 ^ 4 = 1560) :
  p * r * s = 60 :=
  sorry

end product_of_prs_eq_60_l293_293330


namespace third_sec_second_chap_more_than_first_sec_third_chap_l293_293029

-- Define the page lengths for each section in each chapter
def first_chapter : List ℕ := [20, 10, 30]
def second_chapter : List ℕ := [5, 12, 8, 22]
def third_chapter : List ℕ := [7, 11]

-- Define the specific sections of interest
def third_section_second_chapter := second_chapter[2]  -- 8
def first_section_third_chapter := third_chapter[0]   -- 7

-- The theorem we want to prove
theorem third_sec_second_chap_more_than_first_sec_third_chap :
  third_section_second_chapter - first_section_third_chapter = 1 :=
by
  -- Sorry is used here to skip the proof.
  sorry

end third_sec_second_chap_more_than_first_sec_third_chap_l293_293029


namespace seventh_graders_count_l293_293121

theorem seventh_graders_count (x n : ℕ) (hx : n = x * (11 * x - 1))
  (hpoints : 5.5 * n = (11 * x) * (11 * x - 1) / 2) :
  x = 1 :=
by
  sorry

end seventh_graders_count_l293_293121


namespace find_k_of_geometric_mean_l293_293685

-- Let {a_n} be an arithmetic sequence with common difference d and a_1 = 9d.
-- Prove that if a_k is the geometric mean of a_1 and a_{2k}, then k = 4.
theorem find_k_of_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : ∀ n, a n = 9 * d + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a k ^ 2 = a 1 * a (2 * k)) : k = 4 :=
sorry

end find_k_of_geometric_mean_l293_293685


namespace area_on_larger_sphere_l293_293918

-- Define the variables representing the radii and the given area on the smaller sphere
variable (r1 r2 : ℝ) (area1 : ℝ)

-- Given conditions
def conditions : Prop :=
  r1 = 4 ∧ r2 = 6 ∧ area1 = 37

-- Define the statement that we need to prove
theorem area_on_larger_sphere (h : conditions r1 r2 area1) : 
  let area2 := area1 * (r2^2 / r1^2) in
  area2 = 83.25 :=
by
  -- Insert the proof here
  sorry

end area_on_larger_sphere_l293_293918


namespace prob_board_251_l293_293052

noncomputable def probability_boarding_bus_251 (r1 r2 : ℕ) : ℚ :=
  let interval_152 := r1
  let interval_251 := r2
  let total_area := interval_152 * interval_251
  let triangle_area := 1 / 2 * interval_152 * interval_152
  triangle_area / total_area

theorem prob_board_251 : probability_boarding_bus_251 5 7 = 5 / 14 := by
  sorry

end prob_board_251_l293_293052


namespace total_number_of_seats_l293_293264

def number_of_trains : ℕ := 3
def cars_per_train : ℕ := 12
def seats_per_car : ℕ := 24

theorem total_number_of_seats :
  number_of_trains * cars_per_train * seats_per_car = 864 := by
  sorry

end total_number_of_seats_l293_293264


namespace prove_inequality_l293_293978

-- Given conditions
variables {a b : ℝ}
variable {x : ℝ}
variable h : 0 < a
variable k : 0 < b
variable l : ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (abs(x + a) + 2 * abs(x - 1) > x^2 - b + 1)

-- To prove (a + 1/2)^2 + (b + 1/2)^2 > 2
theorem prove_inequality (h : 0 < a) (k : 0 < b) (l : ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (abs(x + a) + 2 * abs(x - 1) > x^2 - b + 1)) :
  (a + 1/2)^2 + (b + 1/2)^2 > 2 :=
sorry

end prove_inequality_l293_293978


namespace sets_difference_M_star_N_l293_293496

def M (y : ℝ) : Prop := y ≤ 2

def N (y : ℝ) : Prop := 0 ≤ y ∧ y ≤ 3

def M_star_N (y : ℝ) : Prop := y < 0

theorem sets_difference_M_star_N : {y : ℝ | M y ∧ ¬ N y} = {y : ℝ | M_star_N y} :=
by {
  sorry
}

end sets_difference_M_star_N_l293_293496


namespace rose_needs_more_money_l293_293514

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l293_293514


namespace square_side_length_difference_l293_293131

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end square_side_length_difference_l293_293131


namespace pencils_left_l293_293776

def ashton_boxes : Nat := 3
def pencils_per_box : Nat := 14
def pencils_given_to_brother : Nat := 6
def pencils_given_to_friends : Nat := 12

theorem pencils_left (h₁ : ashton_boxes = 3) 
                     (h₂ : pencils_per_box = 14)
                     (h₃ : pencils_given_to_brother = 6)
                     (h₄ : pencils_given_to_friends = 12) :
  (ashton_boxes * pencils_per_box - pencils_given_to_brother - pencils_given_to_friends) = 24 :=
by
  sorry

end pencils_left_l293_293776


namespace total_tomato_seeds_l293_293847

theorem total_tomato_seeds (morn_mike morn_morning ted_morning sarah_morning : ℕ)
    (aft_mike aft_ted aft_sarah : ℕ)
    (H1 : morn_mike = 50)
    (H2 : ted_morning = 2 * morn_mike)
    (H3 : sarah_morning = morn_mike + 30)
    (H4 : aft_mike = 60)
    (H5 : aft_ted = aft_mike - 20)
    (H6 : aft_sarah = sarah_morning + 20) :
    morn_mike + aft_mike + ted_morning + aft_ted + sarah_morning + aft_sarah = 430 :=
by
  rw [H1, H2, H3, H4, H5, H6]
  sorry

end total_tomato_seeds_l293_293847


namespace sum_expression_l293_293942

theorem sum_expression : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end sum_expression_l293_293942


namespace positiveDifferenceEquation_l293_293140

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l293_293140


namespace residue_neg_1234_mod_31_l293_293443

theorem residue_neg_1234_mod_31 : -1234 % 31 = 6 := 
by sorry

end residue_neg_1234_mod_31_l293_293443


namespace black_stones_count_l293_293996

theorem black_stones_count (T W B : ℕ) (hT : T = 48) (hW1 : 4 * W = 37 * 2 + 26) (hB : B = T - W) : B = 23 :=
by
  sorry

end black_stones_count_l293_293996


namespace similar_area_ratios_l293_293267

theorem similar_area_ratios (a₁ a₂ s₁ s₂ : ℝ) (h₁ : a₁ = s₁^2) (h₂ : a₂ = s₂^2) (h₃ : a₁ / a₂ = 1 / 9) (h₄ : s₁ = 4) : s₂ = 12 :=
by
  sorry

end similar_area_ratios_l293_293267


namespace fraction_changed_value_l293_293672

theorem fraction_changed_value:
  ∀ (num denom : ℝ), num / denom = 0.75 →
  (num + 0.15 * num) / (denom - 0.08 * denom) = 0.9375 :=
by
  intros num denom h_fraction
  sorry

end fraction_changed_value_l293_293672


namespace acute_angle_is_three_pi_over_eight_l293_293743

noncomputable def acute_angle_concentric_circles : Real :=
  let r₁ := 4
  let r₂ := 3
  let r₃ := 2
  let total_area := (r₁ * r₁ * Real.pi) + (r₂ * r₂ * Real.pi) + (r₃ * r₃ * Real.pi)
  let unshaded_area := 5 * (total_area / 8)
  let shaded_area := (3 / 5) * unshaded_area
  let theta := shaded_area / total_area * 2 * Real.pi
  theta

theorem acute_angle_is_three_pi_over_eight :
  acute_angle_concentric_circles = (3 * Real.pi / 8) :=
by
  sorry

end acute_angle_is_three_pi_over_eight_l293_293743


namespace number_of_77s_l293_293433

theorem number_of_77s (a b : ℕ) :
  (∃ a : ℕ, 1015 = a + 3 * 77 ∧ a + 21 = 10)
  ∧ (∃ b : ℕ, 2023 = b + 6 * 77 + 2 * 777 ∧ b = 7)
  → 6 = 6 := 
by
    sorry

end number_of_77s_l293_293433


namespace probability_diamond_then_ace_l293_293398

theorem probability_diamond_then_ace :
  let total_cards := 104
  let diamonds := 26
  let aces := 8
  let remaining_cards_after_first_draw := total_cards - 1
  let ace_of_diamonds_prob := (2 : ℚ) / total_cards
  let any_ace_after_ace_of_diamonds := (7 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_ace_of_diamonds_then_any_ace := ace_of_diamonds_prob * any_ace_after_ace_of_diamonds
  let diamond_not_ace_prob := (24 : ℚ) / total_cards
  let any_ace_after_diamond_not_ace := (8 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_diamond_not_ace_then_any_ace := diamond_not_ace_prob * any_ace_after_diamond_not_ace
  let total_prob := combined_prob_ace_of_diamonds_then_any_ace + combined_prob_diamond_not_ace_then_any_ace
  total_prob = (31 : ℚ) / 5308 :=
by
  sorry

end probability_diamond_then_ace_l293_293398


namespace find_q_l293_293536

noncomputable def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h₁ : -p / 3 = q) (h₂ : q = 1 + p + q + 5) (h₃ : d = 5) : q = 2 :=
by
  sorry

end find_q_l293_293536


namespace tim_balloons_proof_l293_293948

-- Define the number of balloons Dan has
def dan_balloons : ℕ := 29

-- Define the relationship between Tim's and Dan's balloons
def balloons_ratio : ℕ := 7

-- Define the number of balloons Tim has
def tim_balloons : ℕ := balloons_ratio * dan_balloons

-- Prove that the number of balloons Tim has is 203
theorem tim_balloons_proof : tim_balloons = 203 :=
sorry

end tim_balloons_proof_l293_293948


namespace dennis_years_taught_l293_293747

theorem dennis_years_taught (A V D : ℕ) (h1 : V + A + D = 75) (h2 : V = A + 9) (h3 : V = D - 9) : D = 34 :=
sorry

end dennis_years_taught_l293_293747


namespace count_four_digit_numbers_ending_25_l293_293472

theorem count_four_digit_numbers_ending_25 : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 100]) → ∃ n : ℕ, n = 100 :=
by
  sorry

end count_four_digit_numbers_ending_25_l293_293472


namespace ninth_graders_only_science_not_history_l293_293049

-- Conditions
def total_students : ℕ := 120
def students_science : ℕ := 85
def students_history : ℕ := 75

-- Statement: Determine the number of students enrolled only in the science class
theorem ninth_graders_only_science_not_history : 
  (students_science - (students_science + students_history - total_students)) = 45 := by
  sorry

end ninth_graders_only_science_not_history_l293_293049


namespace part_a_part_b_l293_293237

-- Define what it means to be palindromic
def is_palindromic (n : ℕ) : Prop := 
  let s := n.to_string 
  s = s.reverse

-- Define what it means to be five digits
def is_five_digits (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

-- Define what it means to be divisible by 5
def is_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 = 0

-- Proof problem for part (a)
theorem part_a : 
  ∃ n, is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n ∧ n = 51715 := 
by 
  -- Proof placeholder
  sorry

-- Proof problem for part (b)
theorem part_b : 
  {n : ℕ // is_palindromic n ∧ is_five_digits n ∧ is_divisible_by_5 n}.card = 100 := 
by 
  -- Proof placeholder
  sorry

end part_a_part_b_l293_293237


namespace total_pages_l293_293370

theorem total_pages (history_pages geography_additional math_factor science_factor : ℕ) 
  (h1 : history_pages = 160)
  (h2 : geography_additional = 70)
  (h3 : math_factor = 2)
  (h4 : science_factor = 2) 
  : let geography_pages := history_pages + geography_additional in
    let sum_history_geography := history_pages + geography_pages in
    let math_pages := sum_history_geography / math_factor in
    let science_pages := history_pages * science_factor in
    history_pages + geography_pages + math_pages + science_pages = 905 :=
by
  sorry

end total_pages_l293_293370


namespace length_of_bridge_l293_293572

-- Definitions based on the conditions
def walking_speed_kmph : ℝ := 10 -- speed in km/hr
def time_minutes : ℝ := 24 -- crossing time in minutes
def conversion_factor_km_to_m : ℝ := 1000
def conversion_factor_hr_to_min : ℝ := 60

-- The main statement to prove
theorem length_of_bridge :
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  walking_speed_m_per_min * time_minutes = 4000 := 
by
  let walking_speed_m_per_min := walking_speed_kmph * conversion_factor_km_to_m / conversion_factor_hr_to_min;
  sorry

end length_of_bridge_l293_293572
