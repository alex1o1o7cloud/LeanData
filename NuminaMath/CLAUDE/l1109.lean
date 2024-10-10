import Mathlib

namespace original_number_proof_l1109_110948

theorem original_number_proof (x : ℝ) : 3 * ((2 * x)^2 + 5) = 129 → x = Real.sqrt 9.5 := by
  sorry

end original_number_proof_l1109_110948


namespace factorization_equality_l1109_110962

theorem factorization_equality (a b : ℝ) : a * b^2 + 10 * a * b + 25 * a = a * (b + 5)^2 := by
  sorry

end factorization_equality_l1109_110962


namespace removed_triangles_area_l1109_110978

theorem removed_triangles_area (s : ℝ) (x : ℝ) : 
  s = 16 → 
  (s - 2*x)^2 + (s - 2*x)^2 = s^2 →
  2 * x^2 = 768 - 512 * Real.sqrt 2 := by
  sorry

end removed_triangles_area_l1109_110978


namespace cafeteria_apples_l1109_110987

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 30

/-- The number of pies made -/
def number_of_pies : ℕ := 7

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 8

/-- The total number of apples in the cafeteria initially -/
def total_apples : ℕ := apples_to_students + number_of_pies * apples_per_pie

theorem cafeteria_apples : total_apples = 86 := by sorry

end cafeteria_apples_l1109_110987


namespace john_notebook_duration_l1109_110969

/-- The number of days notebooks last given specific conditions -/
def notebook_duration (
  num_notebooks : ℕ
  ) (pages_per_notebook : ℕ
  ) (pages_per_weekday : ℕ
  ) (pages_per_weekend_day : ℕ
  ) : ℕ :=
  let total_pages := num_notebooks * pages_per_notebook
  let pages_per_week := 5 * pages_per_weekday + 2 * pages_per_weekend_day
  let full_weeks := total_pages / pages_per_week
  let remaining_pages := total_pages % pages_per_week
  let full_days := full_weeks * 7
  let extra_days := 
    if remaining_pages ≤ 5 * pages_per_weekday
    then remaining_pages / pages_per_weekday
    else 5 + (remaining_pages - 5 * pages_per_weekday + pages_per_weekend_day - 1) / pages_per_weekend_day
  full_days + extra_days

theorem john_notebook_duration :
  notebook_duration 5 40 4 6 = 43 := by
  sorry

end john_notebook_duration_l1109_110969


namespace adoption_fee_is_correct_l1109_110909

/-- The adoption fee for an untrained seeing-eye dog. -/
def adoption_fee : ℝ := 150

/-- The weekly training cost for a seeing-eye dog. -/
def weekly_training_cost : ℝ := 250

/-- The number of weeks of training required. -/
def training_weeks : ℕ := 12

/-- The total cost of certification. -/
def certification_cost : ℝ := 3000

/-- The percentage of certification cost covered by insurance. -/
def insurance_coverage : ℝ := 0.9

/-- The total out-of-pocket cost for John. -/
def total_out_of_pocket : ℝ := 3450

/-- Theorem stating that the adoption fee is correct given the conditions. -/
theorem adoption_fee_is_correct : 
  adoption_fee + (weekly_training_cost * training_weeks) + 
  (certification_cost * (1 - insurance_coverage)) = total_out_of_pocket :=
by sorry

end adoption_fee_is_correct_l1109_110909


namespace power_function_through_point_l1109_110981

/-- A power function passing through (3, √3) evaluates to 1/2 at x = 1/4 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x^α) →  -- f is a power function
  f 3 = Real.sqrt 3 →     -- f passes through (3, √3)
  f (1/4) = 1/2 := by
sorry

end power_function_through_point_l1109_110981


namespace two_hour_charge_l1109_110908

/-- Represents the pricing structure for therapy sessions -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hourDifference : firstHourCharge = additionalHourCharge + 25

/-- Calculates the total charge for a given number of therapy hours -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating the total charge for 2 hours of therapy -/
theorem two_hour_charge (pricing : TherapyPricing) 
  (h : totalCharge pricing 5 = 250) : totalCharge pricing 2 = 115 := by
  sorry

end two_hour_charge_l1109_110908


namespace students_just_passed_l1109_110942

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 28 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent ≤ 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 54 :=
by sorry

end students_just_passed_l1109_110942


namespace mary_fruit_expenses_l1109_110940

/-- The total cost of fruits Mary bought -/
def total_cost : ℚ := 34.72

/-- The cost of berries Mary bought -/
def berries_cost : ℚ := 11.08

/-- The cost of apples Mary bought -/
def apples_cost : ℚ := 14.33

/-- The cost of peaches Mary bought -/
def peaches_cost : ℚ := 9.31

/-- Theorem stating that the total cost is the sum of individual fruit costs -/
theorem mary_fruit_expenses : 
  total_cost = berries_cost + apples_cost + peaches_cost := by
  sorry

end mary_fruit_expenses_l1109_110940


namespace netflix_binge_watching_l1109_110947

theorem netflix_binge_watching (episode_length : ℕ) (daily_watch_time : ℕ) (days_to_finish : ℕ) : 
  episode_length = 20 →
  daily_watch_time = 120 →
  days_to_finish = 15 →
  (daily_watch_time * days_to_finish) / episode_length = 90 :=
by
  sorry

end netflix_binge_watching_l1109_110947


namespace f_domain_f_property_f_one_eq_zero_l1109_110930

/-- A function f with the given properties -/
def f : ℝ → ℝ :=
  sorry

theorem f_domain (x : ℝ) : x ≠ 0 → f x ≠ 0 :=
  sorry

theorem f_property (x₁ x₂ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) :
  f (x₁ * x₂) = f x₁ + f x₂ :=
  sorry

theorem f_one_eq_zero : f 1 = 0 :=
  sorry

end f_domain_f_property_f_one_eq_zero_l1109_110930


namespace at_least_two_first_class_products_l1109_110984

def total_products : ℕ := 9
def first_class : ℕ := 4
def second_class : ℕ := 3
def third_class : ℕ := 2
def products_to_draw : ℕ := 4

theorem at_least_two_first_class_products :
  (Nat.choose first_class 2 * Nat.choose (total_products - first_class) 2 +
   Nat.choose first_class 3 * Nat.choose (total_products - first_class) 1 +
   Nat.choose first_class 4 * Nat.choose (total_products - first_class) 0) =
  (Nat.choose total_products products_to_draw -
   Nat.choose (second_class + third_class) products_to_draw -
   (Nat.choose first_class 1 * Nat.choose (second_class + third_class) 3)) :=
by sorry

end at_least_two_first_class_products_l1109_110984


namespace three_X_five_equals_two_l1109_110927

def X (a b : ℝ) : ℝ := b + 8 * a - a^3

theorem three_X_five_equals_two : X 3 5 = 2 := by
  sorry

end three_X_five_equals_two_l1109_110927


namespace apples_needed_for_pies_l1109_110951

theorem apples_needed_for_pies (pies_to_bake : ℕ) (apples_per_pie : ℕ) (apples_on_hand : ℕ) : 
  pies_to_bake * apples_per_pie - apples_on_hand = 110 :=
by
  sorry

#check apples_needed_for_pies 15 10 40

end apples_needed_for_pies_l1109_110951


namespace complement_union_theorem_l1109_110904

universe u

def U : Set ℕ := {1, 2, 3, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_theorem :
  (U \ M) ∪ N = {3, 4, 5} := by sorry

end complement_union_theorem_l1109_110904


namespace inequality_preservation_l1109_110906

theorem inequality_preservation (a b c : ℝ) (h : a > b) (h' : b > 0) :
  a + c > b + c := by
sorry

end inequality_preservation_l1109_110906


namespace cricket_team_avg_age_l1109_110929

-- Define the team and its properties
structure CricketTeam where
  captain_age : ℝ
  wicket_keeper_age : ℝ
  num_bowlers : ℕ
  num_batsmen : ℕ
  team_avg_age : ℝ
  bowlers_avg_age : ℝ
  batsmen_avg_age : ℝ

-- Define the conditions
def team_conditions (team : CricketTeam) : Prop :=
  team.captain_age = 28 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.num_bowlers = 5 ∧
  team.num_batsmen = 4 ∧
  team.bowlers_avg_age = team.team_avg_age - 2 ∧
  team.batsmen_avg_age = team.team_avg_age + 3

-- Theorem statement
theorem cricket_team_avg_age (team : CricketTeam) :
  team_conditions team →
  team.team_avg_age = 30.5 := by
  sorry

end cricket_team_avg_age_l1109_110929


namespace chess_match_schedules_count_l1109_110922

/-- Represents a chess match schedule between two schools -/
structure ChessMatchSchedule where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat
  /-- Total number of games in the match -/
  total_games : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Condition: Each player plays against each player from the other school -/
  player_matchup : players_per_school * players_per_school = total_games
  /-- Condition: Games are evenly distributed across rounds -/
  round_distribution : total_games = games_per_round * total_rounds

/-- The number of different ways to schedule the chess match -/
def number_of_schedules (schedule : ChessMatchSchedule) : Nat :=
  Nat.factorial schedule.total_rounds

/-- Theorem stating that there are 24 different ways to schedule the chess match -/
theorem chess_match_schedules_count :
  ∃ (schedule : ChessMatchSchedule),
    schedule.players_per_school = 4 ∧
    schedule.games_per_round = 4 ∧
    number_of_schedules schedule = 24 := by
  sorry

end chess_match_schedules_count_l1109_110922


namespace solution_set_f_less_than_2_range_of_a_for_solutions_l1109_110937

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) := by sorry

-- Theorem for the range of a
theorem range_of_a_for_solutions (a : ℝ) :
  (∃ x, f x ≤ a - a^2/2) ↔ a ∈ Set.Icc (-1 : ℝ) 3 := by sorry

end solution_set_f_less_than_2_range_of_a_for_solutions_l1109_110937


namespace quadruple_reappearance_l1109_110990

/-- The transformation function that generates the next quadruple -/
def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

/-- The sequence of quadruples generated by repeatedly applying the transformation -/
def quadruple_sequence (initial : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => initial
  | n + 1 => transform (quadruple_sequence initial n)

theorem quadruple_reappearance (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ (n : ℕ), n > 0 ∧ quadruple_sequence (a, b, c, d) n = (a, b, c, d)) →
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
sorry

end quadruple_reappearance_l1109_110990


namespace solution_of_f_1001_l1109_110980

def f₁ (x : ℚ) : ℚ := 2/3 - 3/(3*x+1)

def f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => f₁ x
  | n+1 => f₁ (f n x)

theorem solution_of_f_1001 :
  ∃ x : ℚ, f 1001 x = x - 3 ∧ x = 5/3 := by sorry

end solution_of_f_1001_l1109_110980


namespace searchlight_dark_period_l1109_110925

/-- Given a searchlight that makes 3 revolutions per minute, 
    prove that if the probability of staying in the dark is 0.75, 
    then the duration of the dark period is 15 seconds. -/
theorem searchlight_dark_period 
  (revolutions_per_minute : ℝ) 
  (probability_dark : ℝ) 
  (h1 : revolutions_per_minute = 3) 
  (h2 : probability_dark = 0.75) : 
  (probability_dark * (60 / revolutions_per_minute)) = 15 := by
  sorry

end searchlight_dark_period_l1109_110925


namespace cost_price_for_given_profit_l1109_110932

/-- Given a profit percentage, calculates the cost price as a percentage of the selling price -/
def cost_price_percentage (profit_percentage : Real) : Real :=
  100 - profit_percentage

/-- Theorem stating that when the profit percentage is 4.166666666666666%,
    the cost price is 95.83333333333334% of the selling price -/
theorem cost_price_for_given_profit :
  cost_price_percentage 4.166666666666666 = 95.83333333333334 := by
  sorry

#eval cost_price_percentage 4.166666666666666

end cost_price_for_given_profit_l1109_110932


namespace sqrt_75_plus_30sqrt3_form_l1109_110979

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m.sqrt ^ 2 ∣ n → m.sqrt ^ 2 = 1

theorem sqrt_75_plus_30sqrt3_form :
  ∃ (a b c : ℤ), (c : ℝ) > 0 ∧ is_square_free c.toNat ∧
  Real.sqrt (75 + 30 * Real.sqrt 3) = a + b * Real.sqrt c ∧
  a + b + c = 12 :=
sorry

end sqrt_75_plus_30sqrt3_form_l1109_110979


namespace santinos_mango_trees_l1109_110913

theorem santinos_mango_trees :
  let papaya_trees : ℕ := 2
  let papayas_per_tree : ℕ := 10
  let mangos_per_tree : ℕ := 20
  let total_fruits : ℕ := 80
  ∃ mango_trees : ℕ,
    papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits ∧
    mango_trees = 3 :=
by sorry

end santinos_mango_trees_l1109_110913


namespace mod_power_difference_l1109_110931

theorem mod_power_difference (n : ℕ) : 35^1723 - 16^1723 ≡ 1 [ZMOD 6] := by sorry

end mod_power_difference_l1109_110931


namespace cos_15_degrees_l1109_110918

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degrees_l1109_110918


namespace exactly_one_true_l1109_110992

-- Define what it means for three numbers to be in geometric progression
def in_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

-- Define the original proposition
def original_proposition : Prop :=
  ∀ a b c : ℝ, in_geometric_progression a b c → b^2 = a * c

-- Define the converse
def converse : Prop :=
  ∀ a b c : ℝ, b^2 = a * c → in_geometric_progression a b c

-- Define the inverse
def inverse : Prop :=
  ∀ a b c : ℝ, ¬(in_geometric_progression a b c) → b^2 ≠ a * c

-- Define the contrapositive
def contrapositive : Prop :=
  ∀ a b c : ℝ, b^2 ≠ a * c → ¬(in_geometric_progression a b c)

-- Theorem to prove
theorem exactly_one_true :
  (original_proposition ∧
   (converse ∨ inverse ∨ contrapositive) ∧
   ¬(converse ∧ inverse) ∧
   ¬(converse ∧ contrapositive) ∧
   ¬(inverse ∧ contrapositive)) :=
sorry

end exactly_one_true_l1109_110992


namespace loaves_sold_l1109_110974

/-- The number of loaves sold in a supermarket given initial, delivered, and final counts. -/
theorem loaves_sold (initial : ℕ) (delivered : ℕ) (final : ℕ) :
  initial = 2355 →
  delivered = 489 →
  final = 2215 →
  initial + delivered - final = 629 := by
  sorry

#check loaves_sold

end loaves_sold_l1109_110974


namespace f_properties_l1109_110959

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * abs (x - a)

theorem f_properties :
  (∀ x : ℝ, (f (-1) x = 1 ↔ x ≤ -1 ∨ x = 1)) ∧
  (∀ a : ℝ, (StrictMono (f a) ↔ a ≥ 1/3)) ∧
  (∀ a : ℝ, a < 1 → (∀ x : ℝ, f a x ≥ 2*x - 3) ↔ a ∈ Set.Icc (-3) 1) :=
by sorry

end f_properties_l1109_110959


namespace lucille_earnings_lucille_earnings_proof_l1109_110960

/-- Calculates the amount of money Lucille has left after weeding and buying a soda -/
theorem lucille_earnings (cents_per_weed : ℕ) (flower_bed_weeds : ℕ) (vegetable_patch_weeds : ℕ) 
  (grass_weeds : ℕ) (soda_cost : ℕ) : ℕ :=
  let total_weeds := flower_bed_weeds + vegetable_patch_weeds + grass_weeds / 2
  let total_earnings := total_weeds * cents_per_weed
  total_earnings - soda_cost

/-- Proves that Lucille has 147 cents left after weeding and buying a soda -/
theorem lucille_earnings_proof :
  lucille_earnings 6 11 14 32 99 = 147 := by
  sorry

end lucille_earnings_lucille_earnings_proof_l1109_110960


namespace average_of_abc_l1109_110972

theorem average_of_abc (a b c : ℝ) : 
  (4 + 6 + 9 + a + b + c) / 6 = 18 → (a + b + c) / 3 = 29 + 2/3 := by
  sorry

end average_of_abc_l1109_110972


namespace sum_of_digits_of_B_l1109_110967

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4444^4444 -/
def A : ℕ := sum_of_digits (4444^4444)

/-- B is the sum of digits of A -/
def B : ℕ := sum_of_digits A

/-- Theorem: The sum of digits of B is 7 -/
theorem sum_of_digits_of_B : sum_of_digits B = 7 := by sorry

end sum_of_digits_of_B_l1109_110967


namespace binary_1101_equals_13_l1109_110971

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_equals_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end binary_1101_equals_13_l1109_110971


namespace tea_mixture_price_l1109_110907

/-- Given three varieties of tea with prices and mixing ratios, calculate the price of the mixture --/
theorem tea_mixture_price (p1 p2 p3 : ℚ) (r1 r2 r3 : ℚ) : 
  p1 = 126 → p2 = 135 → p3 = 173.5 → r1 = 1 → r2 = 1 → r3 = 2 →
  (p1 * r1 + p2 * r2 + p3 * r3) / (r1 + r2 + r3) = 152 := by
  sorry

#check tea_mixture_price

end tea_mixture_price_l1109_110907


namespace future_cup_analysis_l1109_110963

/-- Represents a class's defensive performance in the "Future Cup" football match --/
structure DefensivePerformance where
  average_goals_conceded : ℝ
  standard_deviation : ℝ

/-- The defensive performance of Class A --/
def class_a : DefensivePerformance :=
  { average_goals_conceded := 1.9,
    standard_deviation := 0.3 }

/-- The defensive performance of Class B --/
def class_b : DefensivePerformance :=
  { average_goals_conceded := 1.3,
    standard_deviation := 1.2 }

theorem future_cup_analysis :
  (class_b.average_goals_conceded < class_a.average_goals_conceded) ∧
  (class_b.standard_deviation > class_a.standard_deviation) ∧
  (class_a.average_goals_conceded + class_a.standard_deviation < 
   class_b.average_goals_conceded + class_b.standard_deviation) :=
by sorry

end future_cup_analysis_l1109_110963


namespace max_area_quadrilateral_l1109_110966

/-- Given a rectangle ABCD with AB = c and AD = d, and points E on AB and F on AD
    such that AE = AF = x, the maximum area of quadrilateral CDFE is (c + d)^2 / 8. -/
theorem max_area_quadrilateral (c d : ℝ) (h_c : c > 0) (h_d : d > 0) :
  ∃ x : ℝ, 0 < x ∧ x < min c d ∧
    ∀ y : ℝ, 0 < y ∧ y < min c d →
      x * (c + d - 2*x) / 2 ≥ y * (c + d - 2*y) / 2 ∧
      x * (c + d - 2*x) / 2 = (c + d)^2 / 8 :=
by sorry


end max_area_quadrilateral_l1109_110966


namespace coal_shoveling_ratio_l1109_110900

/-- Represents the coal shoveling scenario -/
structure CoalScenario where
  people : ℕ
  days : ℕ
  coal : ℕ

/-- Calculates the daily rate of coal shoveling -/
def daily_rate (s : CoalScenario) : ℚ :=
  s.coal / (s.people * s.days)

theorem coal_shoveling_ratio :
  let original := CoalScenario.mk 10 10 10000
  let new := CoalScenario.mk (10 / 2) 80 40000
  daily_rate original = daily_rate new ∧
  (new.people : ℚ) / original.people = 1 / 2 := by
  sorry

end coal_shoveling_ratio_l1109_110900


namespace gift_spending_theorem_l1109_110970

def num_siblings : ℕ := 3
def num_parents : ℕ := 2
def cost_per_sibling : ℕ := 30
def cost_per_parent : ℕ := 30

def total_cost : ℕ := num_siblings * cost_per_sibling + num_parents * cost_per_parent

theorem gift_spending_theorem : total_cost = 150 := by
  sorry

end gift_spending_theorem_l1109_110970


namespace pizza_slice_count_l1109_110901

/-- The total number of pizza slices given the conditions -/
def totalPizzaSlices (totalPizzas smallPizzaSlices largePizzaSlices : ℕ) : ℕ :=
  let smallPizzas := totalPizzas / 3
  let largePizzas := 2 * smallPizzas
  smallPizzas * smallPizzaSlices + largePizzas * largePizzaSlices

/-- Theorem stating that the total number of pizza slices is 384 -/
theorem pizza_slice_count :
  totalPizzaSlices 36 8 12 = 384 := by
  sorry

end pizza_slice_count_l1109_110901


namespace vector_norm_difference_l1109_110956

theorem vector_norm_difference (a b : ℝ × ℝ) :
  (‖a‖ = 2) → (‖b‖ = 1) → (‖a + b‖ = Real.sqrt 3) → ‖a - b‖ = Real.sqrt 7 := by
  sorry

end vector_norm_difference_l1109_110956


namespace intersection_complement_when_m_is_one_union_equals_A_iff_m_in_range_l1109_110936

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 5*x - 6 < 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m + 1}

-- Part 1
theorem intersection_complement_when_m_is_one :
  A ∩ (Set.univ \ B 1) = {x | -6 < x ∧ x ≤ -1} := by sorry

-- Part 2
theorem union_equals_A_iff_m_in_range :
  ∀ m : ℝ, A ∪ B m = A ↔ m ≤ 0 := by sorry

end intersection_complement_when_m_is_one_union_equals_A_iff_m_in_range_l1109_110936


namespace max_value_of_f_l1109_110988

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

theorem max_value_of_f (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  (∃ x : ℝ, f a x = 5 * Real.exp (-3) ∧ ∀ y : ℝ, f a y ≤ 5 * Real.exp (-3)) :=
by sorry

end max_value_of_f_l1109_110988


namespace ellipse_circle_tangent_relation_l1109_110976

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def is_on_circle (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the tangent line
def is_on_line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the condition that the line is tangent to the circle
def is_tangent_to_circle (k m r : ℝ) : Prop := m^2 = (1 + k^2) * r^2

-- Main theorem
theorem ellipse_circle_tangent_relation 
  (a b r k m x₁ y₁ x₂ y₂ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : 0 < r) (hrb : r < b) :
  is_on_ellipse x₁ y₁ a b ∧ 
  is_on_ellipse x₂ y₂ a b ∧
  is_on_line x₁ y₁ k m ∧
  is_on_line x₂ y₂ k m ∧
  is_tangent_to_circle k m r ∧
  x₁ * x₂ + y₁ * y₂ = 0 →
  r^2 * (a^2 + b^2) = a^2 * b^2 :=
by sorry

end ellipse_circle_tangent_relation_l1109_110976


namespace vector_operation_l1109_110924

/-- Given two 2D vectors a and b, prove that the result of the vector operation is (-1, 2) -/
theorem vector_operation (a b : Fin 2 → ℝ) (ha : a = ![1, 1]) (hb : b = ![1, -1]) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = ![(-1 : ℝ), 2] := by sorry

end vector_operation_l1109_110924


namespace quadratic_roots_imply_coefficients_l1109_110933

theorem quadratic_roots_imply_coefficients (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a + 1)*x + a*b = 0 ↔ x = -1 ∨ x = 4) →
  a = -4 ∧ b = 1 := by
sorry

end quadratic_roots_imply_coefficients_l1109_110933


namespace laptop_to_phone_charger_ratio_l1109_110952

/-- Given a person with 4 phone chargers and 24 total chargers, 
    prove that the ratio of laptop chargers to phone chargers is 5. -/
theorem laptop_to_phone_charger_ratio : 
  ∀ (phone_chargers laptop_chargers : ℕ),
    phone_chargers = 4 →
    phone_chargers + laptop_chargers = 24 →
    laptop_chargers / phone_chargers = 5 := by
  sorry

end laptop_to_phone_charger_ratio_l1109_110952


namespace erroneous_product_equals_correct_l1109_110982

/-- Given a positive integer, reverse its digits --/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is two-digit --/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem erroneous_product_equals_correct (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ is_two_digit b ∧ a * (reverse_digits b) = 180 → a * b = 180 := by
  sorry

end erroneous_product_equals_correct_l1109_110982


namespace marlon_lollipops_l1109_110958

/-- The number of lollipops Marlon had initially -/
def initial_lollipops : ℕ := 42

/-- The fraction of lollipops Marlon gave to Emily -/
def emily_fraction : ℚ := 2/3

/-- The number of lollipops Marlon kept for himself -/
def marlon_kept : ℕ := 4

/-- The number of lollipops Lou received -/
def lou_received : ℕ := 10

theorem marlon_lollipops :
  (initial_lollipops : ℚ) * (1 - emily_fraction) = (marlon_kept + lou_received : ℚ) := by
  sorry

#check marlon_lollipops

end marlon_lollipops_l1109_110958


namespace x_power_ten_equals_one_l1109_110985

theorem x_power_ten_equals_one (x : ℂ) (h : x + 1/x = Real.sqrt 5) : x^10 = 1 := by
  sorry

end x_power_ten_equals_one_l1109_110985


namespace divided_square_area_l1109_110946

/-- A square divided into five rectangles of equal area -/
structure DividedSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of one rectangle -/
  rect_width : ℝ
  /-- The height of the central rectangle -/
  central_height : ℝ
  /-- All rectangles have equal area -/
  equal_area : ℝ
  /-- The given width of one rectangle is 5 -/
  width_condition : rect_width = 5
  /-- The square is divided into 5 rectangles -/
  division_condition : side = rect_width + 2 * central_height
  /-- Area of each rectangle -/
  area_condition : equal_area = rect_width * central_height
  /-- Total area of the square -/
  total_area : ℝ
  /-- Total area is the square of the side length -/
  area_calculation : total_area = side * side

/-- The theorem stating that the area of the divided square is 400 -/
theorem divided_square_area (s : DividedSquare) : s.total_area = 400 := by
  sorry

end divided_square_area_l1109_110946


namespace range_of_a_l1109_110991

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l1109_110991


namespace ratio_equality_l1109_110902

theorem ratio_equality (a b : ℚ) (h : a / b = 7 / 6) : 6 * a = 7 * b := by
  sorry

end ratio_equality_l1109_110902


namespace complex_number_problem_l1109_110995

def z (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_problem (b : ℝ) 
  (h : ∃ (y : ℝ), (1 + 3 * Complex.I) * z b = y * Complex.I) :
  z b = 3 + Complex.I ∧ Complex.abs (z b / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end complex_number_problem_l1109_110995


namespace unique_plane_through_line_and_point_l1109_110950

-- Define the 3D space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable [Fact (finrank ℝ V = 3)]

-- Define a line in 3D space
def Line (p q : V) : Set V :=
  {x | ∃ t : ℝ, x = p + t • (q - p)}

-- Define a plane in 3D space
def Plane (n : V) (c : ℝ) : Set V :=
  {x | inner n x = c}

-- State the theorem
theorem unique_plane_through_line_and_point 
  (l : Set V) (A : V) (p q : V) (h_line : l = Line p q) (h_not_on : A ∉ l) :
  ∃! P : Set V, ∃ n : V, ∃ c : ℝ, 
    P = Plane n c ∧ l ⊆ P ∧ A ∈ P :=
sorry

end unique_plane_through_line_and_point_l1109_110950


namespace prob_three_students_same_group_l1109_110983

/-- The total number of students -/
def total_students : ℕ := 800

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of a student being assigned to a specific group -/
def prob_one_group : ℚ := 1 / num_groups

/-- The probability that three specific students are assigned to the same lunch group -/
theorem prob_three_students_same_group :
  (prob_one_group * prob_one_group : ℚ) = 1 / 16 :=
sorry

end prob_three_students_same_group_l1109_110983


namespace ap_terms_count_l1109_110965

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Even n → 
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 36 →
  (n / 2 : ℚ) * (2 * a + (n - 1) * d) = 44 →
  a + (n - 1) * d - a = 12 →
  n = 8 := by
sorry

end ap_terms_count_l1109_110965


namespace trig_identity_l1109_110939

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end trig_identity_l1109_110939


namespace valid_arrangements_count_l1109_110944

/-- Represents a student in the line -/
inductive Student
  | boyA
  | boyB
  | girl1
  | girl2
  | girl3

/-- Represents a row of students -/
def Row := List Student

/-- Checks if exactly two of the three girls are adjacent in the row -/
def exactlyTwoGirlsAdjacent (row : Row) : Bool := sorry

/-- Checks if boy A is not at either end of the row -/
def boyANotAtEnds (row : Row) : Bool := sorry

/-- Generates all valid permutations of the students -/
def validPermutations : List Row := sorry

/-- Counts the number of valid arrangements -/
def countValidArrangements : Nat :=
  validPermutations.filter (λ row => exactlyTwoGirlsAdjacent row && boyANotAtEnds row) |>.length

theorem valid_arrangements_count :
  countValidArrangements = 36 := by sorry

end valid_arrangements_count_l1109_110944


namespace initial_price_increase_l1109_110914

theorem initial_price_increase (P : ℝ) (x : ℝ) : 
  P * (1 + x / 100) * (1 - 10 / 100) = P * (1 + 12.5 / 100) → 
  x = 25 := by
sorry

end initial_price_increase_l1109_110914


namespace line_separate_from_circle_l1109_110926

/-- A line with negative slope intersecting a circle is separate from another circle -/
theorem line_separate_from_circle (k : ℝ) (h_k : k < 0) : 
  ∃ (x y : ℝ), y = k * x ∧ (x + 3)^2 + (y + 2)^2 = 9 →
  ∀ (x y : ℝ), y = k * x → x^2 + (y - 2)^2 > 1 := by
sorry

end line_separate_from_circle_l1109_110926


namespace unique_solution_for_a_l1109_110953

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2|

-- Define the function p(x, a)
def p (x a : ℝ) : ℝ := |x| + a

-- Define the domain of f
def D_f : Set ℝ := {x | x ≠ 2 ∧ x ≠ -1}

-- Theorem statement
theorem unique_solution_for_a (a : ℝ) :
  a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 ↔
  ∃! (x : ℝ), x ∈ D_f ∧ f x = p x a :=
sorry

end unique_solution_for_a_l1109_110953


namespace baga_answer_variability_l1109_110949

/-- Represents a BAGA problem -/
structure BAGAProblem where
  conditions : Set String
  approach : String

/-- Represents the answer to a BAGA problem -/
structure BAGAAnswer where
  value : String

/-- Function that solves a BAGA problem -/
noncomputable def solveBagaProblem (problem : BAGAProblem) : BAGAAnswer :=
  sorry

/-- Theorem stating that small variations in BAGA problems can lead to different answers -/
theorem baga_answer_variability 
  (p1 p2 : BAGAProblem) 
  (h_small_diff : p1.conditions ≠ p2.conditions ∨ p1.approach ≠ p2.approach) : 
  ∃ (a1 a2 : BAGAAnswer), solveBagaProblem p1 = a1 ∧ solveBagaProblem p2 = a2 ∧ a1 ≠ a2 :=
sorry

end baga_answer_variability_l1109_110949


namespace complex_quadrant_l1109_110954

theorem complex_quadrant (z : ℂ) (h : (1 + 2*I)/z = 1 - I) : 
  z.re < 0 ∧ z.im > 0 := by
sorry

end complex_quadrant_l1109_110954


namespace weekly_card_pack_size_l1109_110911

theorem weekly_card_pack_size (total_weeks : ℕ) (remaining_cards : ℕ) : 
  total_weeks = 52 →
  remaining_cards = 520 →
  (remaining_cards * 2) / total_weeks = 20 :=
by sorry

end weekly_card_pack_size_l1109_110911


namespace parabola_solutions_l1109_110955

/-- A parabola defined by y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_solutions (p : Parabola) (m : ℝ) :
  p.y (-4) = m →
  p.y 0 = m →
  p.y 2 = 1 →
  p.y 4 = 0 →
  (∀ x : ℝ, p.y x = 0 ↔ x = 4 ∨ x = -8) :=
sorry

end parabola_solutions_l1109_110955


namespace find_a_min_value_of_sum_l1109_110998

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (I)
theorem find_a :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∃ a : ℝ, a = 2) :=
sorry

-- Theorem for part (II)
theorem min_value_of_sum (x : ℝ) :
  ∃ m : ℝ, m = 5/3 ∧ ∀ x : ℝ, f 2 (3*x) + f 2 (x+3) ≥ m :=
sorry

end find_a_min_value_of_sum_l1109_110998


namespace eleven_billion_scientific_notation_l1109_110921

theorem eleven_billion_scientific_notation :
  (11 : ℝ) * (10 ^ 9 : ℝ) = (1.1 : ℝ) * (10 ^ 10 : ℝ) := by
  sorry

end eleven_billion_scientific_notation_l1109_110921


namespace f_max_min_l1109_110905

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + x + 1

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3/2 }

theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max = 5/4 ∧
    min = 1/4 := by sorry

end f_max_min_l1109_110905


namespace problem_solution_l1109_110973

theorem problem_solution (N : ℚ) : 
  (4 / 5 : ℚ) * (3 / 8 : ℚ) * N = 24 → (5 / 2 : ℚ) * N = 200 := by
  sorry

end problem_solution_l1109_110973


namespace P_root_characteristics_l1109_110996

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^7 - 4*x^5 - 8*x^3 - x + 12

-- Theorem statement
theorem P_root_characteristics :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) := by sorry

end P_root_characteristics_l1109_110996


namespace cosine_translation_monotonicity_l1109_110977

/-- Given a function g(x) = 2cos(2x - π/3) that is monotonically increasing
    in the intervals [0, a/3] and [2a, 7π/6], prove that π/3 ≤ a ≤ π/2. -/
theorem cosine_translation_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (a / 3), Monotone (fun x => 2 * Real.cos (2 * x - π / 3))) ∧
  (∀ x ∈ Set.Icc (2 * a) (7 * π / 6), Monotone (fun x => 2 * Real.cos (2 * x - π / 3))) →
  π / 3 ≤ a ∧ a ≤ π / 2 :=
by sorry

end cosine_translation_monotonicity_l1109_110977


namespace impossible_to_equalize_l1109_110961

/-- Represents the circular arrangement of six numbers -/
def CircularArrangement := Fin 6 → ℕ

/-- The initial arrangement of numbers from 1 to 6 -/
def initial_arrangement : CircularArrangement :=
  fun i => i.val + 1

/-- Adds 1 to three consecutive numbers in the arrangement -/
def add_to_consecutive (a : CircularArrangement) (start : Fin 6) : CircularArrangement :=
  fun i => if i = start ∨ i = start.succ ∨ i = start.succ.succ then a i + 1 else a i

/-- Subtracts 1 from three alternating numbers in the arrangement -/
def subtract_from_alternating (a : CircularArrangement) (start : Fin 6) : CircularArrangement :=
  fun i => if i = start ∨ i = start.succ.succ ∨ i = start.succ.succ.succ.succ then a i - 1 else a i

/-- Checks if all numbers in the arrangement are equal -/
def all_equal (a : CircularArrangement) : Prop :=
  ∀ i j : Fin 6, a i = a j

/-- Main theorem: It's impossible to equalize all numbers using the given operations -/
theorem impossible_to_equalize :
  ¬ ∃ (ops : List (CircularArrangement → CircularArrangement)),
    all_equal (ops.foldl (fun acc op => op acc) initial_arrangement) :=
sorry

end impossible_to_equalize_l1109_110961


namespace fraction_of_powers_equals_five_thirds_l1109_110997

theorem fraction_of_powers_equals_five_thirds :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5/3 := by
  sorry

end fraction_of_powers_equals_five_thirds_l1109_110997


namespace torn_pages_theorem_l1109_110923

/-- Represents a set of consecutive pages torn from a book --/
structure TornPages where
  first : ℕ  -- First page number
  count : ℕ  -- Number of pages torn out

/-- The sum of consecutive integers from n to n + k - 1 --/
def sum_consecutive (n : ℕ) (k : ℕ) : ℕ :=
  k * (2 * n + k - 1) / 2

theorem torn_pages_theorem (pages : TornPages) :
  sum_consecutive pages.first pages.count = 344 →
  (344 = 2^3 * 43 ∧
   pages.first + (pages.first + pages.count - 1) = 43 ∧
   pages.count = 16) := by
  sorry


end torn_pages_theorem_l1109_110923


namespace difference_of_squares_multiplication_l1109_110975

theorem difference_of_squares_multiplication (a b : ℕ) :
  58 * 42 = 2352 := by
  sorry

end difference_of_squares_multiplication_l1109_110975


namespace workshop_pairing_probability_l1109_110957

theorem workshop_pairing_probability (n : ℕ) (h : n = 24) :
  let total_participants := n
  let pairing_probability := (1 : ℚ) / (n - 1 : ℚ)
  pairing_probability = (1 : ℚ) / (23 : ℚ) :=
by sorry

end workshop_pairing_probability_l1109_110957


namespace line_inclination_angle_l1109_110928

theorem line_inclination_angle (x y : ℝ) :
  y - 3 = Real.sqrt 3 * (x - 4) →
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ Real.tan α = Real.sqrt 3 ∧ α = π / 3 :=
by sorry

end line_inclination_angle_l1109_110928


namespace octagon_proof_l1109_110999

theorem octagon_proof (n : ℕ) (h : n > 2) : 
  (n * (n - 3)) / 2 = n + 2 * (n - 2) → n = 8 := by
sorry

end octagon_proof_l1109_110999


namespace box_has_four_balls_l1109_110968

/-- A color of a ball -/
inductive Color
| Red
| Blue
| Other

/-- A box containing balls of different colors -/
structure Box where
  balls : List Color

/-- Checks if a list of colors contains at least one red and one blue -/
def hasRedAndBlue (colors : List Color) : Prop :=
  Color.Red ∈ colors ∧ Color.Blue ∈ colors

/-- The main theorem stating that the box must contain exactly 4 balls -/
theorem box_has_four_balls (box : Box) : 
  (∀ (a b c : Color), a ∈ box.balls → b ∈ box.balls → c ∈ box.balls → 
    a ≠ b → b ≠ c → a ≠ c → hasRedAndBlue [a, b, c]) →
  (3 < box.balls.length) →
  box.balls.length = 4 := by
  sorry


end box_has_four_balls_l1109_110968


namespace max_d_value_l1109_110935

def a (n : ℕ+) : ℕ := 101 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ+), d k = 3 ∧ ∀ (n : ℕ+), d n ≤ 3 :=
sorry

end max_d_value_l1109_110935


namespace angle_of_inclination_sqrt_3_l1109_110919

/-- The angle of inclination (in radians) for a line with slope √3 is π/3 (60°) -/
theorem angle_of_inclination_sqrt_3 :
  let slope : ℝ := Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 3 := by
sorry


end angle_of_inclination_sqrt_3_l1109_110919


namespace hyperbola_equation_given_conditions_l1109_110945

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Check if a point is on a hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  hyperbola_equation h p

theorem hyperbola_equation_given_conditions 
  (E : Hyperbola)
  (center : Point)
  (focus : Point)
  (N : Point)
  (h_center : center.x = 0 ∧ center.y = 0)
  (h_focus : focus.x = 3 ∧ focus.y = 0)
  (h_midpoint : N.x = -12 ∧ N.y = -15)
  (h_on_hyperbola : ∃ (A B : Point), 
    point_on_hyperbola E A ∧ 
    point_on_hyperbola E B ∧ 
    N.x = (A.x + B.x) / 2 ∧ 
    N.y = (A.y + B.y) / 2) :
  E.a^2 = 4 ∧ E.b^2 = 5 := by
  sorry

#check hyperbola_equation_given_conditions

end hyperbola_equation_given_conditions_l1109_110945


namespace apple_sale_theorem_l1109_110917

/-- Calculates the total number of apples sold given the number of red apples and the ratio of red:green:yellow apples -/
def total_apples (red_apples : ℕ) (red_ratio green_ratio yellow_ratio : ℕ) : ℕ :=
  let total_ratio := red_ratio + green_ratio + yellow_ratio
  let apples_per_part := red_apples / red_ratio
  red_apples + (green_ratio * apples_per_part) + (yellow_ratio * apples_per_part)

/-- Theorem stating that given 32 red apples and a ratio of 8:3:5 for red:green:yellow apples, the total number of apples sold is 64 -/
theorem apple_sale_theorem : total_apples 32 8 3 5 = 64 := by
  sorry

end apple_sale_theorem_l1109_110917


namespace mass_percentage_H_in_NH4I_l1109_110934

-- Define atomic masses
def atomic_mass_N : ℝ := 14.01
def atomic_mass_H : ℝ := 1.01
def atomic_mass_I : ℝ := 126.90

-- Define the composition of NH4I
def NH4I_composition : Fin 3 → ℕ
  | 0 => 1  -- N
  | 1 => 4  -- H
  | 2 => 1  -- I
  | _ => 0

-- Define the molar mass of NH4I
def molar_mass_NH4I : ℝ :=
  NH4I_composition 0 * atomic_mass_N +
  NH4I_composition 1 * atomic_mass_H +
  NH4I_composition 2 * atomic_mass_I

-- Define the mass of hydrogen in NH4I
def mass_H_in_NH4I : ℝ := NH4I_composition 1 * atomic_mass_H

-- Theorem statement
theorem mass_percentage_H_in_NH4I :
  abs ((mass_H_in_NH4I / molar_mass_NH4I) * 100 - 2.79) < 0.01 := by
  sorry

end mass_percentage_H_in_NH4I_l1109_110934


namespace sum_of_largest_and_smallest_prime_factors_of_1260_l1109_110943

theorem sum_of_largest_and_smallest_prime_factors_of_1260 : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ 1260 ∧ q ∣ 1260 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 1260 → p ≤ r ∧ r ≤ q) ∧
  p + q = 9 :=
by sorry

end sum_of_largest_and_smallest_prime_factors_of_1260_l1109_110943


namespace inequality_solution_set_l1109_110938

theorem inequality_solution_set (k : ℝ) :
  (∃ x : ℝ, |x - 2| - |x - 5| > k) → k < 3 := by
  sorry

end inequality_solution_set_l1109_110938


namespace min_value_sum_squares_l1109_110912

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + 3*c = 1 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 1/14) := by
  sorry

end min_value_sum_squares_l1109_110912


namespace cos_225_degrees_l1109_110941

theorem cos_225_degrees : Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l1109_110941


namespace circle_radius_from_tangents_l1109_110910

/-- Given a circle with diameter AB, tangents AD and BC, and a line through D and C
    intersecting the circle at E, prove that the radius is (c+d)/2 when AD = c and BC = d. -/
theorem circle_radius_from_tangents (c d : ℝ) (h : c ≠ d) :
  let circle : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 0)^2 + (y - 0)^2 = ((c + d)/2)^2}
  let A : ℝ × ℝ := (-(c + d)/2, 0)
  let B : ℝ × ℝ := ((c + d)/2, 0)
  let D : ℝ × ℝ := (-c, c)
  let C : ℝ × ℝ := (d, d)
  let E : ℝ × ℝ := (0, (c + d)/2)
  (∀ p ∈ circle, (p.1 - A.1)^2 + (p.2 - A.2)^2 = ((c + d)/2)^2) ∧
  (∀ p ∈ circle, (p.1 - B.1)^2 + (p.2 - B.2)^2 = ((c + d)/2)^2) ∧
  (D ∉ circle) ∧ (C ∉ circle) ∧
  ((D.1 - A.1) * (D.2 - A.2) + (D.1 - 0) * (D.2 - 0) = 0) ∧
  ((C.1 - B.1) * (C.2 - B.2) + (C.1 - 0) * (C.2 - 0) = 0) ∧
  (E ∈ circle) ∧
  (D.2 - A.2)/(D.1 - A.1) = (E.2 - D.2)/(E.1 - D.1) ∧
  (C.2 - B.2)/(C.1 - B.1) = (E.2 - C.2)/(E.1 - C.1) →
  (c + d)/2 = (c + d)/2 := by
sorry

end circle_radius_from_tangents_l1109_110910


namespace budget_allocation_l1109_110964

theorem budget_allocation (salaries utilities equipment supplies transportation : ℝ) 
  (h1 : salaries = 60)
  (h2 : utilities = 5)
  (h3 : equipment = 4)
  (h4 : supplies = 2)
  (h5 : transportation = 72 / 360 * 100)
  (h6 : salaries + utilities + equipment + supplies + transportation < 100) :
  100 - (salaries + utilities + equipment + supplies + transportation) = 9 := by
sorry

end budget_allocation_l1109_110964


namespace quadratic_factorization_l1109_110920

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l1109_110920


namespace percentage_decrease_l1109_110916

theorem percentage_decrease (t : ℝ) (x : ℝ) : 
  t = 80 → 
  (t + 0.125 * t) - (t - x / 100 * t) = 30 → 
  x = 25 := by
sorry

end percentage_decrease_l1109_110916


namespace second_project_depth_l1109_110989

/-- Represents a digging project with its dimensions and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  days : ℝ

/-- Calculates the volume of a digging project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := 30,
  days := 12
}

/-- The second digging project with unknown depth -/
def project2 (depth : ℝ) : DiggingProject := {
  depth := depth,
  length := 20,
  breadth := 50,
  days := 12
}

/-- Theorem stating that the depth of the second project is 75 meters -/
theorem second_project_depth : 
  ∃ (depth : ℝ), volume project1 = volume (project2 depth) ∧ depth = 75 := by
  sorry


end second_project_depth_l1109_110989


namespace probability_three_face_cards_different_suits_value_l1109_110986

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of face cards in a standard deck. -/
def FaceCards : ℕ := 12

/-- The number of suits in a standard deck. -/
def Suits : ℕ := 4

/-- The number of face cards per suit. -/
def FaceCardsPerSuit : ℕ := FaceCards / Suits

/-- The probability of selecting three face cards of different suits from a standard deck without replacement. -/
def probability_three_face_cards_different_suits : ℚ :=
  (FaceCards : ℚ) / StandardDeck *
  (FaceCards - FaceCardsPerSuit : ℚ) / (StandardDeck - 1) *
  (FaceCards - 2 * FaceCardsPerSuit : ℚ) / (StandardDeck - 2)

theorem probability_three_face_cards_different_suits_value :
  probability_three_face_cards_different_suits = 4 / 915 := by
  sorry

end probability_three_face_cards_different_suits_value_l1109_110986


namespace johns_age_l1109_110993

theorem johns_age (john : ℕ) (matt : ℕ) : 
  matt = 4 * john - 3 → 
  john + matt = 52 → 
  john = 11 := by
sorry

end johns_age_l1109_110993


namespace triangle_max_sum_l1109_110994

theorem triangle_max_sum (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = 3 →
  1 + (Real.tan A) / (Real.tan B) = 2 * c / b →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b > 0 ∧ c > 0 →
  (∀ b' c' : ℝ, b' > 0 ∧ c' > 0 →
    a^2 = b'^2 + c'^2 - 2 * b' * c' * Real.cos A →
    b' + c' ≤ b + c) →
  b + c = 3 * Real.sqrt 2 :=
by sorry

end triangle_max_sum_l1109_110994


namespace sine_translation_l1109_110915

/-- Given a function g(x) obtained by translating y = sin(2x) to the right by π/12 units,
    prove that g(π/12) = 0 -/
theorem sine_translation (g : ℝ → ℝ) : 
  (∀ x, g x = Real.sin (2 * (x - π/12))) → g (π/12) = 0 := by
  sorry

end sine_translation_l1109_110915


namespace company_picnic_attendance_l1109_110903

/-- Percentage of employees who attended the company picnic -/
def picnic_attendance (men_percentage : Real) (women_percentage : Real) 
  (men_attendance : Real) (women_attendance : Real) : Real :=
  men_percentage * men_attendance + (1 - men_percentage) * women_attendance

theorem company_picnic_attendance :
  picnic_attendance 0.45 0.55 0.20 0.40 = 0.31 := by
  sorry

end company_picnic_attendance_l1109_110903
