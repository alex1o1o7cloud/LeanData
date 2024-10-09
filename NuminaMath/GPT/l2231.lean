import Mathlib

namespace second_player_wins_l2231_223153

theorem second_player_wins 
  (pile1 : ℕ) (pile2 : ℕ) (pile3 : ℕ)
  (h1 : pile1 = 10) (h2 : pile2 = 15) (h3 : pile3 = 20) :
  (pile1 - 1) + (pile2 - 1) + (pile3 - 1) % 2 = 0 :=
by
  sorry

end second_player_wins_l2231_223153


namespace extreme_value_at_3_increasing_on_interval_l2231_223136

def f (a : ℝ) (x : ℝ) : ℝ := 2*x^3 - 3*(a+1)*x^2 + 6*a*x + 8

theorem extreme_value_at_3 (a : ℝ) : (∃ x, x = 3 ∧ 6*x^2 - 6*(a+1)*x + 6*a = 0) → a = 3 :=
by
  sorry

theorem increasing_on_interval (a : ℝ) : (∀ x, x < 0 → 6*(x-a)*(x-1) > 0) → 0 ≤ a :=
by
  sorry

end extreme_value_at_3_increasing_on_interval_l2231_223136


namespace tennis_tournament_l2231_223120

theorem tennis_tournament (n : ℕ) (w m : ℕ) 
  (total_matches : ℕ)
  (women_wins men_wins : ℕ) :
  n + 2 * n = 3 * n →
  total_matches = (3 * n * (3 * n - 1)) / 2 →
  women_wins + men_wins = total_matches →
  women_wins / men_wins = 7 / 5 →
  n = 3 :=
by sorry

end tennis_tournament_l2231_223120


namespace at_least_one_hit_l2231_223187

-- Introduce the predicates
variable (p q : Prop)

-- State the theorem
theorem at_least_one_hit : (¬ (¬ p ∧ ¬ q)) = (p ∨ q) :=
by
  sorry

end at_least_one_hit_l2231_223187


namespace initial_amount_correct_l2231_223133

noncomputable def initial_amount (A R T : ℝ) : ℝ :=
  A / (1 + (R * T) / 100)

theorem initial_amount_correct :
  initial_amount 2000 3.571428571428571 4 = 1750 :=
by
  sorry

end initial_amount_correct_l2231_223133


namespace max_f1_l2231_223182

-- Define the function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b 

-- Define the condition 
def condition (a : ℝ) (b : ℝ) : Prop := f 0 a b = 4

-- State the theorem
theorem max_f1 (a b: ℝ) (h: condition a b) : 
  ∃ b_max, b_max = 1 ∧ ∀ b, f 1 a b ≤ 7 := 
sorry

end max_f1_l2231_223182


namespace find_y_l2231_223147

/-- 
  Given: The sum of angles around a point is 360 degrees, 
  and those angles are: 6y, 3y, 4y, and 2y.
  Prove: y = 24 
-/ 
theorem find_y (y : ℕ) (h : 6 * y + 3 * y + 4 * y + 2 * y = 360) : y = 24 :=
sorry

end find_y_l2231_223147


namespace minimum_value_of_f_l2231_223174

def f (x : ℝ) : ℝ := 5 * x^2 - 20 * x + 1357

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) := 
by 
  use 1337
  sorry

end minimum_value_of_f_l2231_223174


namespace random_event_is_crane_among_chickens_l2231_223115

-- Definitions of the idioms as events
def coveringTheSkyWithOneHand : Prop := false
def fumingFromAllSevenOrifices : Prop := false
def stridingLikeAMeteor : Prop := false
def standingOutLikeACraneAmongChickens : Prop := ¬false

-- The theorem stating that Standing out like a crane among chickens is a random event
theorem random_event_is_crane_among_chickens :
  ¬coveringTheSkyWithOneHand ∧ ¬fumingFromAllSevenOrifices ∧ ¬stridingLikeAMeteor → standingOutLikeACraneAmongChickens :=
by 
  sorry

end random_event_is_crane_among_chickens_l2231_223115


namespace twenty_four_multiples_of_4_l2231_223166

theorem twenty_four_multiples_of_4 {n : ℕ} : (n = 104) ↔ (∃ k : ℕ, k = 24 ∧ ∀ m : ℕ, (12 ≤ m ∧ m ≤ n) → ∃ t : ℕ, m = 12 + 4 * t ∧ 1 ≤ t ∧ t ≤ 24) := 
by
  sorry

end twenty_four_multiples_of_4_l2231_223166


namespace sam_money_left_l2231_223158

/- Definitions -/

def initial_dimes : ℕ := 38
def initial_quarters : ℕ := 12
def initial_nickels : ℕ := 25
def initial_pennies : ℕ := 30

def price_per_candy_bar_dimes : ℕ := 4
def price_per_candy_bar_nickels : ℕ := 2
def candy_bars_bought : ℕ := 5

def price_per_lollipop_nickels : ℕ := 6
def price_per_lollipop_pennies : ℕ := 10
def lollipops_bought : ℕ := 2

def price_per_bag_of_chips_quarters : ℕ := 1
def price_per_bag_of_chips_dimes : ℕ := 3
def price_per_bag_of_chips_pennies : ℕ := 5
def bags_of_chips_bought : ℕ := 3

/- Proof problem statement -/

theorem sam_money_left : 
  (initial_dimes * 10 + initial_quarters * 25 + initial_nickels * 5 + initial_pennies * 1) - 
  (
    candy_bars_bought * (price_per_candy_bar_dimes * 10 + price_per_candy_bar_nickels * 5) + 
    lollipops_bought * (price_per_lollipop_nickels * 5 + price_per_lollipop_pennies * 1) +
    bags_of_chips_bought * (price_per_bag_of_chips_quarters * 25 + price_per_bag_of_chips_dimes * 10 + price_per_bag_of_chips_pennies * 1)
  ) = 325 := 
sorry

end sam_money_left_l2231_223158


namespace track_width_l2231_223189

theorem track_width (r : ℝ) (h1 : 4 * π * r - 2 * π * r = 16 * π) (h2 : 2 * r = r + r) : 2 * r - r = 8 :=
by
  sorry

end track_width_l2231_223189


namespace shots_cost_l2231_223164

-- Define the conditions
def golden_retriever_pregnant_dogs : ℕ := 3
def golden_retriever_puppies_per_dog : ℕ := 4
def golden_retriever_shots_per_puppy : ℕ := 2
def golden_retriever_cost_per_shot : ℕ := 5

def german_shepherd_pregnant_dogs : ℕ := 2
def german_shepherd_puppies_per_dog : ℕ := 5
def german_shepherd_shots_per_puppy : ℕ := 3
def german_shepherd_cost_per_shot : ℕ := 8

def bulldog_pregnant_dogs : ℕ := 4
def bulldog_puppies_per_dog : ℕ := 3
def bulldog_shots_per_puppy : ℕ := 4
def bulldog_cost_per_shot : ℕ := 10

-- Define the total cost calculation
def total_puppies (dogs_per_breed puppies_per_dog : ℕ) : ℕ :=
  dogs_per_breed * puppies_per_dog

def total_shot_cost (puppies shots_per_puppy cost_per_shot : ℕ) : ℕ :=
  puppies * shots_per_puppy * cost_per_shot

def total_cost : ℕ :=
  let golden_retriever_puppies := total_puppies golden_retriever_pregnant_dogs golden_retriever_puppies_per_dog
  let german_shepherd_puppies := total_puppies german_shepherd_pregnant_dogs german_shepherd_puppies_per_dog
  let bulldog_puppies := total_puppies bulldog_pregnant_dogs bulldog_puppies_per_dog
  let golden_retriever_cost := total_shot_cost golden_retriever_puppies golden_retriever_shots_per_puppy golden_retriever_cost_per_shot
  let german_shepherd_cost := total_shot_cost german_shepherd_puppies german_shepherd_shots_per_puppy german_shepherd_cost_per_shot
  let bulldog_cost := total_shot_cost bulldog_puppies bulldog_shots_per_puppy bulldog_cost_per_shot
  golden_retriever_cost + german_shepherd_cost + bulldog_cost

-- Statement of the problem
theorem shots_cost (total_cost : ℕ) : total_cost = 840 := by
  -- Proof would go here
  sorry

end shots_cost_l2231_223164


namespace books_in_series_l2231_223146

-- Define the number of movies
def M := 14

-- Define that the number of books is one more than the number of movies
def B := M + 1

-- Theorem statement to prove that the number of books is 15
theorem books_in_series : B = 15 :=
by
  sorry

end books_in_series_l2231_223146


namespace boat_length_in_steps_l2231_223116

theorem boat_length_in_steps (L E S : ℝ) 
  (h1 : 250 * E = L + 250 * S) 
  (h2 : 50 * E = L - 50 * S) :
  L = 83 * E :=
by sorry

end boat_length_in_steps_l2231_223116


namespace range_of_t_l2231_223131

noncomputable def a_n (t : ℝ) (n : ℕ) : ℝ :=
  if n > 8 then ((1 / 3) - t) * (n:ℝ) + 2 else t ^ (n - 7)

theorem range_of_t (t : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → a_n t n > a_n t (n + 1)) →
  (1/2 < t ∧ t < 1) :=
by
  intros h
  -- The proof would go here.
  sorry

end range_of_t_l2231_223131


namespace max_marked_points_l2231_223185

theorem max_marked_points (segments : ℕ) (ratio : ℚ) (h_segments : segments = 10) (h_ratio : ratio = 3 / 4) : 
  ∃ n, n ≤ (segments * 2 / 2) ∧ n = 10 :=
by
  sorry

end max_marked_points_l2231_223185


namespace appropriate_term_for_assessment_l2231_223103

-- Definitions
def price : Type := String
def value : Type := String
def cost : Type := String
def expense : Type := String

-- Context for assessment of the project
def assessment_context : Type := Π (word : String), word ∈ ["price", "value", "cost", "expense"] → Prop

-- Main Lean statement
theorem appropriate_term_for_assessment (word : String) (h : word ∈ ["price", "value", "cost", "expense"]) :
  word = "value" :=
sorry

end appropriate_term_for_assessment_l2231_223103


namespace total_votes_election_l2231_223156

theorem total_votes_election (V : ℝ)
    (h1 : 0.55 * 0.8 * V + 2520 = 0.8 * V)
    (h2 : 0.36 > 0) :
    V = 7000 :=
  by
  sorry

end total_votes_election_l2231_223156


namespace minimum_employment_age_l2231_223124

/-- This structure represents the conditions of the problem -/
structure EmploymentConditions where
  jane_current_age : ℕ  -- Jane's current age
  years_until_dara_half_age : ℕ  -- Years until Dara is half Jane's age
  years_until_dara_min_age : ℕ  -- Years until Dara reaches minimum employment age

/-- The proof problem statement -/
theorem minimum_employment_age (conds : EmploymentConditions)
  (h_jane : conds.jane_current_age = 28)
  (h_half_age : conds.years_until_dara_half_age = 6)
  (h_min_age : conds.years_until_dara_min_age = 14) :
  let jane_in_six := conds.jane_current_age + conds.years_until_dara_half_age
  let dara_in_six := jane_in_six / 2
  let dara_now := dara_in_six - conds.years_until_dara_half_age
  let M := dara_now + conds.years_until_dara_min_age
  M = 25 :=
by
  sorry

end minimum_employment_age_l2231_223124


namespace savings_on_discounted_milk_l2231_223117

theorem savings_on_discounted_milk :
  let num_gallons := 8
  let price_per_gallon := 3.20
  let discount_rate := 0.25
  let discount_per_gallon := price_per_gallon * discount_rate
  let discounted_price_per_gallon := price_per_gallon - discount_per_gallon
  let total_cost_without_discount := num_gallons * price_per_gallon
  let total_cost_with_discount := num_gallons * discounted_price_per_gallon
  let savings := total_cost_without_discount - total_cost_with_discount
  savings = 6.40 :=
by
  sorry

end savings_on_discounted_milk_l2231_223117


namespace length_GH_l2231_223109

theorem length_GH (AB BC : ℝ) (hAB : AB = 10) (hBC : BC = 5) (DG DH GH : ℝ)
  (hDG : DG = DH) (hArea_DGH : 1 / 2 * DG * DH = 1 / 5 * (AB * BC)) :
  GH = 2 * Real.sqrt 10 :=
by
  sorry

end length_GH_l2231_223109


namespace andrey_gifts_l2231_223140

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l2231_223140


namespace circle_equation_l2231_223176

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let radius : ℝ := dist center point
  dist center point = 1 → 
  (x - 1)^2 + y^2 = radius^2 :=
by
  intros
  sorry

end circle_equation_l2231_223176


namespace abc_product_l2231_223122

/-- Given a b c + a b + b c + a c + a + b + c = 164 -/
theorem abc_product :
  ∃ (a b c : ℕ), a * b * c + a * b + b * c + a * c + a + b + c = 164 ∧ a * b * c = 80 :=
by
  sorry

end abc_product_l2231_223122


namespace find_smallest_z_l2231_223151

theorem find_smallest_z (x y z : ℤ) (h1 : 7 < x) (h2 : x < 9) (h3 : x < y) (h4 : y < z) 
  (h5 : y - x = 7) : z = 16 :=
by
  sorry

end find_smallest_z_l2231_223151


namespace part1_part2_l2231_223184

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x^2 ≤ 5 * x - 4
def q (x a : ℝ) : Prop := x^2 - (a + 2) * x + 2 * a ≤ 0

-- Theorem statement for part (1)
theorem part1 (x : ℝ) (h : p x) : 1 ≤ x ∧ x ≤ 4 := 
by sorry

-- Theorem statement for part (2)
theorem part2 (a : ℝ) : 
  (∀ x, p x → q x a) ∧ (∃ x, p x) ∧ ¬ (∀ x, q x a → p x) → 1 ≤ a ∧ a ≤ 4 := 
by sorry

end part1_part2_l2231_223184


namespace find_x_l2231_223163

theorem find_x :
  ∃ x : Real, abs (x - 0.052) < 1e-3 ∧
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + x^2 + 0.0035^2) = 100 :=
by
  sorry

end find_x_l2231_223163


namespace total_batteries_correct_l2231_223149

-- Definitions of the number of batteries used in each category
def batteries_flashlight : ℕ := 2
def batteries_toys : ℕ := 15
def batteries_controllers : ℕ := 2

-- The total number of batteries used by Tom
def total_batteries : ℕ := batteries_flashlight + batteries_toys + batteries_controllers

-- The proof statement that needs to be proven
theorem total_batteries_correct : total_batteries = 19 := by
  sorry

end total_batteries_correct_l2231_223149


namespace cube_convex_hull_half_volume_l2231_223155

theorem cube_convex_hull_half_volume : 
  ∃ a : ℝ, 0 <= a ∧ a <= 1 ∧ 4 * (a^3) / 6 + 4 * ((1 - a)^3) / 6 = 1 / 2 :=
by
  sorry

end cube_convex_hull_half_volume_l2231_223155


namespace sum_of_sequence_l2231_223129

theorem sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, a n = (-1 : ℤ)^(n+1) * (2*n - 1)) →
  (S 0 = 0) →
  (∀ n, S (n+1) = S n + a (n+1)) →
  (∀ n, S (n+1) = (-1 : ℤ)^(n+1) * (n+1)) :=
by
  intros h_a h_S0 h_S
  sorry

end sum_of_sequence_l2231_223129


namespace three_rays_with_common_point_l2231_223167

theorem three_rays_with_common_point (x y : ℝ) :
  (∃ (common : ℝ), ((5 = x - 1 ∧ y + 3 ≤ 5) ∨ 
                     (5 = y + 3 ∧ x - 1 ≤ 5) ∨ 
                     (x - 1 = y + 3 ∧ 5 ≤ x - 1 ∧ 5 ≤ y + 3)) 
  ↔ ((x = 6 ∧ y ≤ 2) ∨ (y = 2 ∧ x ≤ 6) ∨ (y = x - 4 ∧ x ≥ 6))) :=
sorry

end three_rays_with_common_point_l2231_223167


namespace equivalent_single_discount_rate_l2231_223172

-- Definitions based on conditions
def original_price : ℝ := 120
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.15
def combined_discount_rate : ℝ := 0.3625  -- This is the expected result

-- The proof problem statement
theorem equivalent_single_discount_rate :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 
  (original_price * (1 - combined_discount_rate)) := 
sorry

end equivalent_single_discount_rate_l2231_223172


namespace no_real_solution_l2231_223181

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end no_real_solution_l2231_223181


namespace ginger_distance_l2231_223148

theorem ginger_distance : 
  ∀ (d : ℝ), (d / 4 - d / 6 = 1 / 16) → (d = 3 / 4) := 
by 
  intro d h
  sorry

end ginger_distance_l2231_223148


namespace total_wheels_in_garage_l2231_223198

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l2231_223198


namespace proof_problem_l2231_223175

-- Definitions
def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Conditions
def condition_A : Prop := is_factor 4 24
def condition_B : Prop := is_divisor 19 152 ∧ ¬ is_divisor 19 96
def condition_E : Prop := is_factor 6 180

-- Proof problem statement
theorem proof_problem : condition_A ∧ condition_B ∧ condition_E :=
by sorry

end proof_problem_l2231_223175


namespace distance_to_weekend_class_l2231_223192

theorem distance_to_weekend_class:
  ∃ d v : ℝ, (d = v * (1 / 2)) ∧ (d = (v + 10) * (3 / 10)) → d = 7.5 :=
by
  sorry

end distance_to_weekend_class_l2231_223192


namespace find_v2_poly_l2231_223108

theorem find_v2_poly (x : ℤ) (v0 v1 v2 : ℤ) 
  (h1 : x = -4)
  (h2 : v0 = 1) 
  (h3 : v1 = v0 * x)
  (h4 : v2 = v1 * x + 6) :
  v2 = 22 :=
by
  -- To be filled with proof (example problem requirement specifies proof is not needed)
  sorry

end find_v2_poly_l2231_223108


namespace rin_craters_difference_l2231_223100

theorem rin_craters_difference (d da r : ℕ) (h1 : d = 35) (h2 : da = d - 10) (h3 : r = 75) :
  r - (d + da) = 15 :=
by
  sorry

end rin_craters_difference_l2231_223100


namespace depth_of_first_hole_l2231_223141

theorem depth_of_first_hole :
  (45 * 8 * (80 * 6 * 40) / (45 * 8) : ℝ) = 53.33 := by
  -- This is where you would provide the proof, but it will be skipped with 'sorry'
  sorry

end depth_of_first_hole_l2231_223141


namespace fewer_VIP_tickets_sold_l2231_223128

variable (V G : ℕ)

-- Definitions: total number of tickets sold and the total revenue from tickets sold
def total_tickets : Prop := V + G = 320
def total_revenue : Prop := 45 * V + 20 * G = 7500

-- Definition of the number of fewer VIP tickets than general admission tickets
def fewer_VIP_tickets : Prop := G - V = 232

-- The theorem to be proven
theorem fewer_VIP_tickets_sold (h1 : total_tickets V G) (h2 : total_revenue V G) : fewer_VIP_tickets V G :=
sorry

end fewer_VIP_tickets_sold_l2231_223128


namespace axis_of_symmetry_l2231_223183

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (5 - x)) : ∀ x : ℝ, f x = f (2 * 2.5 - x) :=
by
  sorry

end axis_of_symmetry_l2231_223183


namespace percentage_both_correct_l2231_223195

theorem percentage_both_correct (p1 p2 pn : ℝ) (h1 : p1 = 0.85) (h2 : p2 = 0.80) (h3 : pn = 0.05) :
  ∃ x, x = 0.70 ∧ x = p1 + p2 - 1 + pn := by
  sorry

end percentage_both_correct_l2231_223195


namespace find_certain_number_l2231_223113

theorem find_certain_number (x : ℝ) (h : ((7 * (x + 5)) / 5) - 5 = 33) : x = 22 :=
by
  sorry

end find_certain_number_l2231_223113


namespace find_num_of_boys_l2231_223130

-- Define the constants for number of girls and total number of kids
def num_of_girls : ℕ := 3
def total_kids : ℕ := 9

-- The theorem stating the number of boys based on the given conditions
theorem find_num_of_boys (g t : ℕ) (h1 : g = num_of_girls) (h2 : t = total_kids) :
  t - g = 6 :=
by
  sorry

end find_num_of_boys_l2231_223130


namespace tegwen_family_total_children_l2231_223111

variable (Tegwen : Type)

-- Variables representing the number of girls and boys
variable (g b : ℕ)

-- Conditions from the problem
variable (h1 : b = g - 1)
variable (h2 : g = (3/2:ℚ) * (b - 1))

-- Proposition that the total number of children is 11
theorem tegwen_family_total_children : g + b = 11 := by
  sorry

end tegwen_family_total_children_l2231_223111


namespace hyperbola_asymptotes_l2231_223161

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (e : ℝ) (he : e = Real.sqrt 3) (h_eq : e = Real.sqrt ((a^2 + b^2) / a^2)) :
  (∀ x : ℝ, y = x * Real.sqrt 2) :=
by
  sorry

end hyperbola_asymptotes_l2231_223161


namespace initial_ratio_of_liquids_l2231_223160

theorem initial_ratio_of_liquids (A B : ℕ) (H1 : A = 21)
  (H2 : 9 * A = 7 * (B + 9)) :
  A / B = 7 / 6 :=
sorry

end initial_ratio_of_liquids_l2231_223160


namespace men_l2231_223110

-- Given conditions
variable (W M : ℕ)
variable (B : ℕ) [DecidableEq ℕ] -- number of boys
variable (total_earnings : ℕ)

def earnings : ℕ := 5 * M + W * M + 8 * W

-- Total earnings of men, women, and boys is Rs. 150.
def conditions : Prop := 
  5 * M = W * M ∧ 
  W * M = 8 * W ∧ 
  earnings = total_earnings

-- Prove men's wages (total wages for 5 men) is Rs. 50.
theorem men's_wages (hm : total_earnings = 150) (hb : W = 8) : 
  5 * M = 50 :=
by
  sorry

end men_l2231_223110


namespace product_eq_1280_l2231_223126

axiom eq1 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48
axiom eq2 (a b c d : ℝ) : 4 * d + 2 * c = 2 * b
axiom eq3 (a b c d : ℝ) : 4 * b + 2 * c = 2 * a
axiom eq4 (a b c d : ℝ) : c - 2 = d
axiom eq5 (a b c d : ℝ) : d + b = 10

theorem product_eq_1280 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48 → 4 * d + 2 * c = 2 * b → 4 * b + 2 * c = 2 * a → c - 2 = d → d + b = 10 → a * b * c * d = 1280 :=
by 
  intro h1 h2 h3 h4 h5
  -- we put the proof here
  sorry

end product_eq_1280_l2231_223126


namespace smallest_positive_natural_number_l2231_223190

theorem smallest_positive_natural_number (a b c d e : ℕ) 
    (h1 : a = 3) (h2 : b = 5) (h3 : c = 6) (h4 : d = 18) (h5 : e = 23) :
    ∃ (x y : ℕ), x = (e - a) / b - d / c ∨ x = e - d + b - c - a ∧ x = 1 := by
  sorry

end smallest_positive_natural_number_l2231_223190


namespace part1_solution_set_of_inequality_part2_range_of_m_l2231_223121

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1_solution_set_of_inequality :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} :=
by
  sorry

theorem part2_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x > 6 * m ^ 2 - 4 * m) ↔ -1/3 < m ∧ m < 1 :=
by
  sorry

end part1_solution_set_of_inequality_part2_range_of_m_l2231_223121


namespace max_articles_produced_l2231_223119

variables (a b c d p q r s z : ℝ)
variables (h1 : d = (a^2 * b * c) / z)
variables (h2 : p * q * r ≤ s)

theorem max_articles_produced : 
  p * q * r * (a / z) = s * (a / z) :=
by
  sorry

end max_articles_produced_l2231_223119


namespace simplify_and_evaluate_expression_l2231_223193

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) : 
  (1 - (3 / (m + 3))) / (m / (m^2 + 6 * m + 9)) = Real.sqrt 2 := 
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l2231_223193


namespace player_A_advantage_l2231_223168

theorem player_A_advantage (B A : ℤ) (rolls : ℕ) (h : rolls = 36) 
  (game_conditions : ∀ (x : ℕ), (x % 2 = 1 → A = A + x ∧ B = B - x) ∧ 
                      (x % 2 = 0 ∧ x ≠ 2 → A = A - x ∧ B = B + x) ∧ 
                      (x = 2 → A = A ∧ B = B)) : 
  (36 * (1 / 18 : ℚ) = 2) :=
by {
  -- Mathematical proof will be filled here
  sorry
}

end player_A_advantage_l2231_223168


namespace z_squared_in_second_quadrant_l2231_223159
open Complex Real

noncomputable def z : ℂ := exp (π * I / 3)

theorem z_squared_in_second_quadrant : (z^2).re < 0 ∧ (z^2).im > 0 :=
by
  sorry

end z_squared_in_second_quadrant_l2231_223159


namespace train_length_approx_l2231_223104

noncomputable def length_of_train (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_seconds

theorem train_length_approx (speed_km_hr time_seconds : ℝ) (h_speed : speed_km_hr = 120) (h_time : time_seconds = 4) :
  length_of_train speed_km_hr time_seconds = 133.32 :=
by
  sorry

end train_length_approx_l2231_223104


namespace slice_of_bread_area_l2231_223101

theorem slice_of_bread_area (total_area : ℝ) (number_of_parts : ℕ) (h1 : total_area = 59.6) (h2 : number_of_parts = 4) : 
  total_area / number_of_parts = 14.9 :=
by
  rw [h1, h2]
  norm_num


end slice_of_bread_area_l2231_223101


namespace solve_for_t_l2231_223178

variable (f : ℝ → ℝ)
variable (x t : ℝ)

-- Conditions
def cond1 : Prop := ∀ x, f ((1 / 2) * x - 1) = 2 * x + 3
def cond2 : Prop := f t = 4

-- Theorem statement
theorem solve_for_t (h1 : cond1 f) (h2 : cond2 f t) : t = -3 / 4 := by
  sorry

end solve_for_t_l2231_223178


namespace sqrt_nested_eq_five_l2231_223191

theorem sqrt_nested_eq_five {x : ℝ} (h : x = Real.sqrt (15 + x)) : x = 5 :=
sorry

end sqrt_nested_eq_five_l2231_223191


namespace smallest_year_with_digit_sum_16_l2231_223177

def sum_of_digits (n : Nat) : Nat :=
  let digits : List Nat := n.digits 10
  digits.foldl (· + ·) 0

theorem smallest_year_with_digit_sum_16 :
  ∃ (y : Nat), 2010 < y ∧ sum_of_digits y = 16 ∧
  (∀ (z : Nat), 2010 < z ∧ sum_of_digits z = 16 → z ≥ y) → y = 2059 :=
by
  sorry

end smallest_year_with_digit_sum_16_l2231_223177


namespace piper_gym_sessions_l2231_223145

-- Define the conditions and the final statement as a theorem
theorem piper_gym_sessions (session_count : ℕ) (week_days : ℕ) (start_day : ℕ) 
  (alternate_day : ℕ) (skip_day : ℕ): (session_count = 35) ∧ (week_days = 7) ∧ 
  (start_day = 1) ∧ (alternate_day = 2) ∧ (skip_day = 7) → 
  (start_day + ((session_count - 1) / 3) * week_days + ((session_count - 1) % 3) * alternate_day) % week_days = 3 := 
by 
  sorry

end piper_gym_sessions_l2231_223145


namespace siblings_pizza_order_l2231_223179

theorem siblings_pizza_order :
  let Alex := 1 / 6
  let Beth := 2 / 5
  let Cyril := 1 / 3
  let Dan := 1 - (Alex + Beth + Cyril)
  Dan > Alex ∧ Alex > Cyril ∧ Cyril > Beth := sorry

end siblings_pizza_order_l2231_223179


namespace sin_cos_15_eq_1_over_4_l2231_223157

theorem sin_cos_15_eq_1_over_4 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := 
by
  sorry

end sin_cos_15_eq_1_over_4_l2231_223157


namespace total_legs_l2231_223173

def animals_legs (dogs : Nat) (birds : Nat) (insects : Nat) : Nat :=
  (dogs * 4) + (birds * 2) + (insects * 6)

theorem total_legs :
  animals_legs 3 2 2 = 22 := by
  sorry

end total_legs_l2231_223173


namespace k_interval_l2231_223106

noncomputable def f (x k : ℝ) : ℝ := x^2 + (1 - k) * x - k

theorem k_interval (k : ℝ) :
  (∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x k = 0) ↔ (2 < k ∧ k < 3) :=
by
  sorry

end k_interval_l2231_223106


namespace sister_granola_bars_l2231_223188

-- Definitions based on conditions
def total_bars := 20
def chocolate_chip_bars := 8
def oat_honey_bars := 6
def peanut_butter_bars := 6

def greg_set_aside_chocolate := 3
def greg_set_aside_oat_honey := 2
def greg_set_aside_peanut_butter := 2

def final_chocolate_chip := chocolate_chip_bars - greg_set_aside_chocolate - 2  -- 2 traded away
def final_oat_honey := oat_honey_bars - greg_set_aside_oat_honey - 4           -- 4 traded away
def final_peanut_butter := peanut_butter_bars - greg_set_aside_peanut_butter

-- Final distribution to sisters
def older_sister_chocolate := 2.5 -- 2 whole bars + 1/2 bar
def younger_sister_peanut := 2.5  -- 2 whole bars + 1/2 bar

theorem sister_granola_bars :
  older_sister_chocolate = 2.5 ∧ younger_sister_peanut = 2.5 :=
by
  sorry

end sister_granola_bars_l2231_223188


namespace six_digit_ababab_divisible_by_101_l2231_223186

theorem six_digit_ababab_divisible_by_101 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) :
  ∃ k : ℕ, 101 * k = 101010 * a + 10101 * b :=
sorry

end six_digit_ababab_divisible_by_101_l2231_223186


namespace triangle_area_is_24_l2231_223171

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end triangle_area_is_24_l2231_223171


namespace cubes_sum_formula_l2231_223165

theorem cubes_sum_formula (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) : a^3 + b^3 = 238 := 
by 
  sorry

end cubes_sum_formula_l2231_223165


namespace major_axis_length_l2231_223127

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (hr : r = 2) 
  (h_minor : minor_axis = 2 * r)
  (h_major : major_axis = 1.25 * minor_axis) :
  major_axis = 5 :=
by
  sorry

end major_axis_length_l2231_223127


namespace complex_inverse_identity_l2231_223114

theorem complex_inverse_identity : ∀ (i : ℂ), i^2 = -1 → (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by
  -- Let's introduce the variables and the condition.
  intro i h

  -- Sorry is used to signify the proof is omitted.
  sorry

end complex_inverse_identity_l2231_223114


namespace full_time_employees_l2231_223125

theorem full_time_employees (total_employees part_time_employees number_full_time_employees : ℕ)
  (h1 : total_employees = 65134)
  (h2 : part_time_employees = 2041)
  (h3 : number_full_time_employees = total_employees - part_time_employees)
  : number_full_time_employees = 63093 :=
by {
  sorry
}

end full_time_employees_l2231_223125


namespace tony_additional_degrees_l2231_223170

-- Definitions for the conditions
def total_years : ℕ := 14
def science_degree_years : ℕ := 4
def physics_degree_years : ℕ := 2
def additional_degree_years : ℤ := total_years - (science_degree_years + physics_degree_years)
def each_additional_degree_years : ℕ := 4
def additional_degrees : ℤ := additional_degree_years / each_additional_degree_years

-- Theorem stating the problem and the answer
theorem tony_additional_degrees : additional_degrees = 2 :=
 by
     sorry

end tony_additional_degrees_l2231_223170


namespace abs_eq_abs_iff_eq_frac_l2231_223197

theorem abs_eq_abs_iff_eq_frac {x : ℚ} :
  |x - 3| = |x - 4| → x = 7 / 2 :=
by
  intro h
  sorry

end abs_eq_abs_iff_eq_frac_l2231_223197


namespace minimum_value_of_expression_l2231_223135

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2 / 5) ^ (1 / 5) := by
sorry

end minimum_value_of_expression_l2231_223135


namespace problem_statement_l2231_223199

open Real

theorem problem_statement (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1/a) + (1/b) = 1) (hn_pos : 0 < n) :
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry -- proof to be provided

end problem_statement_l2231_223199


namespace arithmetic_problem_l2231_223112

theorem arithmetic_problem : 
  (888.88 - 555.55 + 111.11) * 2 = 888.88 := 
sorry

end arithmetic_problem_l2231_223112


namespace saucepan_capacity_l2231_223196

-- Define the conditions
variable (x : ℝ)
variable (h : 0.28 * x = 35)

-- State the theorem
theorem saucepan_capacity : x = 125 :=
by
  sorry

end saucepan_capacity_l2231_223196


namespace henry_change_l2231_223123

theorem henry_change (n : ℕ) (p m : ℝ) (h_n : n = 4) (h_p : p = 0.75) (h_m : m = 10) : 
  m - (n * p) = 7 := 
by 
  sorry

end henry_change_l2231_223123


namespace fish_fishermen_problem_l2231_223105

theorem fish_fishermen_problem (h: ℕ) (r: ℕ) (w_h: ℕ) (w_r: ℕ) (claimed_weight: ℕ) (total_real_weight: ℕ) 
  (total_fishermen: ℕ) :
  -- conditions
  (claimed_weight = 60) →
  (total_real_weight = 120) →
  (total_fishermen = 10) →
  (w_h = 30) →
  (w_r < 60 / 7) →
  (h + r = total_fishermen) →
  (2 * w_h * h + r * claimed_weight = claimed_weight * total_fishermen) →
  -- prove the number of regular fishermen
  (r = 7 ∨ r = 8) :=
sorry

end fish_fishermen_problem_l2231_223105


namespace integer_values_m_l2231_223144

theorem integer_values_m (m x y : ℤ) (h1 : x - 2 * y = m) (h2 : 2 * x + 3 * y = 2 * m - 3)
    (h3 : 3 * x + y ≥ 0) (h4 : x + 5 * y < 0) : m = 1 ∨ m = 2 :=
by
  sorry

end integer_values_m_l2231_223144


namespace parabola_vertex_l2231_223169

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end parabola_vertex_l2231_223169


namespace gym_class_total_students_l2231_223102

theorem gym_class_total_students (group1_members group2_members : ℕ) 
  (h1 : group1_members = 34) (h2 : group2_members = 37) :
  group1_members + group2_members = 71 :=
by
  sorry

end gym_class_total_students_l2231_223102


namespace star_compound_l2231_223132

noncomputable def star (A B : ℝ) : ℝ := (A + B) / 4

theorem star_compound : star (star 3 11) 6 = 2.375 := by
  sorry

end star_compound_l2231_223132


namespace equalize_nuts_l2231_223143

open Nat

noncomputable def transfer (p1 p2 p3 : ℕ) : Prop :=
  ∃ (m1 m2 m3 : ℕ), 
    m1 ≤ p1 ∧ m1 ≤ p2 ∧ 
    m2 ≤ (p2 + m1) ∧ m2 ≤ p3 ∧ 
    m3 ≤ (p3 + m2) ∧ m3 ≤ (p1 - m1) ∧
    (p1 - m1 + m3 = 16) ∧ 
    (p2 + m1 - m2 = 16) ∧ 
    (p3 + m2 - m3 = 16)

theorem equalize_nuts : transfer 22 14 12 := 
  sorry

end equalize_nuts_l2231_223143


namespace triangle_area_sqrt2_div2_find_a_c_l2231_223180

  -- Problem 1
  -- Prove the area of triangle ABC is sqrt(2)/2
  theorem triangle_area_sqrt2_div2 {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : Real.arcsin (1) = Real.pi / 2) : 
    (1 / 2) * 1 * Real.sqrt 2 = Real.sqrt 2 / 2 := sorry

  -- Problem 2
  -- Prove a = sqrt(7) and c = 2
  theorem find_a_c {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : (1 / 2) * a * Real.sin (Real.arcsin (Real.sqrt 3 / a)) = Real.sqrt 3 / 2) : 
    a = Real.sqrt 7 ∧ c = 2 := sorry

  
end triangle_area_sqrt2_div2_find_a_c_l2231_223180


namespace y_exceeds_x_by_35_percent_l2231_223107

theorem y_exceeds_x_by_35_percent {x y : ℝ} (h : x = 0.65 * y) : ((y - x) / x) * 100 = 35 :=
by
  sorry

end y_exceeds_x_by_35_percent_l2231_223107


namespace rhombus_area_l2231_223150

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 11) (h2 : d2 = 16) : (d1 * d2) / 2 = 88 :=
by {
  -- substitution and proof are omitted, proof body would be provided here
  sorry
}

end rhombus_area_l2231_223150


namespace number_of_ways_to_choose_positions_l2231_223118

-- Definition of the problem conditions
def number_of_people : ℕ := 8

-- Statement of the proof problem
theorem number_of_ways_to_choose_positions : 
  (number_of_people) * (number_of_people - 1) * (number_of_people - 2) = 336 := by
  -- skipping the proof itself
  sorry

end number_of_ways_to_choose_positions_l2231_223118


namespace negation_abs_lt_one_l2231_223154

theorem negation_abs_lt_one (x : ℝ) : (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_abs_lt_one_l2231_223154


namespace parabola_directrix_l2231_223152

theorem parabola_directrix (x y : ℝ) (h : y = 2 * x^2) : y = - (1 / 8) :=
sorry

end parabola_directrix_l2231_223152


namespace arnold_protein_intake_l2231_223138

theorem arnold_protein_intake :
  (∀ p q s : ℕ,  p = 18 / 2 ∧ q = 21 ∧ s = 56 → (p + q + s = 86)) := by
  sorry

end arnold_protein_intake_l2231_223138


namespace identify_quadratic_l2231_223142

def is_quadratic (eq : String) : Prop :=
  eq = "x^2 - 2x + 1 = 0"

theorem identify_quadratic :
  is_quadratic "x^2 - 2x + 1 = 0" :=
by
  sorry

end identify_quadratic_l2231_223142


namespace employees_in_room_l2231_223137

-- Define variables
variables (E : ℝ) (M : ℝ) (L : ℝ)

-- Given conditions
def condition1 : Prop := M = 0.99 * E
def condition2 : Prop := (M - L) / E = 0.98
def condition3 : Prop := L = 99.99999999999991

-- Prove statement
theorem employees_in_room (h1 : condition1 E M) (h2 : condition2 E M L) (h3 : condition3 L) : E = 10000 :=
by
  sorry

end employees_in_room_l2231_223137


namespace max_a_value_l2231_223139

theorem max_a_value (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 3| ≥ 2 * a) : a ≤ 1 :=
sorry

end max_a_value_l2231_223139


namespace find_k_l2231_223134

def vec2 := ℝ × ℝ

-- Definitions
def i : vec2 := (1, 0)
def j : vec2 := (0, 1)
def a : vec2 := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : vec2 := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Dot product definition for 2D vectors
def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by
  sorry

end find_k_l2231_223134


namespace hedge_cost_and_blocks_l2231_223194

-- Define the costs of each type of block
def costA : Nat := 2
def costB : Nat := 3
def costC : Nat := 4

-- Define the number of each type of block per section
def blocksPerSectionA : Nat := 20
def blocksPerSectionB : Nat := 10
def blocksPerSectionC : Nat := 5

-- Define the number of sections
def sections : Nat := 8

-- Define the total cost calculation
def totalCost : Nat := sections * (blocksPerSectionA * costA + blocksPerSectionB * costB + blocksPerSectionC * costC)

-- Define the total number of each type of block used
def totalBlocksA : Nat := sections * blocksPerSectionA
def totalBlocksB : Nat := sections * blocksPerSectionB
def totalBlocksC : Nat := sections * blocksPerSectionC

-- State the theorem
theorem hedge_cost_and_blocks :
  totalCost = 720 ∧ totalBlocksA = 160 ∧ totalBlocksB = 80 ∧ totalBlocksC = 40 := by
  sorry

end hedge_cost_and_blocks_l2231_223194


namespace journey_total_distance_l2231_223162

/--
Given:
- A person covers 3/5 of their journey by train.
- A person covers 7/20 of their journey by bus.
- A person covers 3/10 of their journey by bicycle.
- A person covers 1/50 of their journey by taxi.
- The rest of the journey (4.25 km) is covered by walking.

Prove:
  D = 15.74 km
where D is the total distance of the journey.
-/
theorem journey_total_distance :
  ∀ (D : ℝ), 3/5 * D + 7/20 * D + 3/10 * D + 1/50 * D + 4.25 = D → D = 15.74 :=
by
  intro D
  sorry

end journey_total_distance_l2231_223162
