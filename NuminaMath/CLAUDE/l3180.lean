import Mathlib

namespace valid_arrangements_count_l3180_318021

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_teachers : ℕ := male_teachers + female_teachers
def head_teachers_needed : ℕ := 3

def valid_arrangements : ℕ := 
  Nat.factorial total_teachers / Nat.factorial (total_teachers - head_teachers_needed) -
  (Nat.factorial male_teachers / Nat.factorial (male_teachers - head_teachers_needed) +
   Nat.factorial female_teachers / Nat.factorial (female_teachers - head_teachers_needed))

theorem valid_arrangements_count : valid_arrangements = 420 := by
  sorry

end valid_arrangements_count_l3180_318021


namespace new_person_weight_l3180_318080

/-- Given a group of 8 persons where one person weighing 65 kg is replaced by a new person,
    and the average weight of the group increases by 2.5 kg,
    prove that the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 85 :=
by sorry

end new_person_weight_l3180_318080


namespace consecutive_points_length_l3180_318006

/-- Given 5 consecutive points on a straight line, prove that ab = 5 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 8) →            -- de = 8
  (c - a = 11) →           -- ac = 11
  (e - a = 21) →           -- ae = 21
  (b - a = 5) :=           -- ab = 5
by sorry

end consecutive_points_length_l3180_318006


namespace rotation_result_l3180_318015

-- Define the shapes
inductive Shape
  | SmallCircle
  | Triangle
  | Square
  | Pentagon

-- Define the rotation directions
inductive RotationDirection
  | Clockwise
  | Counterclockwise

-- Define the configuration of shapes
structure Configuration :=
  (smallCircle : ℝ)  -- Angle of rotation for small circle
  (triangle : ℝ)     -- Angle of rotation for triangle
  (pentagon : ℝ)     -- Angle of rotation for pentagon
  (overall : ℝ)      -- Overall rotation of the configuration

-- Define the rotation function
def rotate (shape : Shape) (angle : ℝ) (direction : RotationDirection) : ℝ :=
  match direction with
  | RotationDirection.Clockwise => angle
  | RotationDirection.Counterclockwise => -angle

-- Define the initial configuration
def initialConfig : Configuration :=
  { smallCircle := 0, triangle := 0, pentagon := 0, overall := 0 }

-- Define the final configuration after rotations
def finalConfig (initial : Configuration) : Configuration :=
  { smallCircle := initial.smallCircle + rotate Shape.SmallCircle 45 RotationDirection.Counterclockwise,
    triangle := initial.triangle + rotate Shape.Triangle 180 RotationDirection.Clockwise,
    pentagon := initial.pentagon + rotate Shape.Pentagon 120 RotationDirection.Clockwise,
    overall := initial.overall + rotate Shape.Square 90 RotationDirection.Clockwise }

-- Theorem statement
theorem rotation_result :
  let final := finalConfig initialConfig
  final.smallCircle = -45 ∧
  final.triangle = 180 ∧
  final.pentagon = 120 ∧
  final.overall = 90 :=
by sorry

end rotation_result_l3180_318015


namespace contest_participants_l3180_318028

theorem contest_participants (P : ℕ) 
  (h1 : (P / 2 : ℚ) = P * (1 / 2 : ℚ)) 
  (h2 : (P / 2 + P / 2 / 7 : ℚ) = P * (57.14285714285714 / 100 : ℚ)) : 
  ∃ k : ℕ, P = 7 * k :=
sorry

end contest_participants_l3180_318028


namespace water_bottle_consumption_l3180_318078

theorem water_bottle_consumption (total_bottles : ℕ) (days : ℕ) (bottles_per_day : ℕ) : 
  total_bottles = 153 → 
  days = 17 → 
  total_bottles = bottles_per_day * days → 
  bottles_per_day = 9 := by
  sorry

end water_bottle_consumption_l3180_318078


namespace initial_price_is_four_l3180_318070

/-- Represents the sales data for a day --/
structure DaySales where
  price : ℝ
  quantity : ℝ

/-- Represents the sales data for three days --/
structure ThreeDaySales where
  day1 : DaySales
  day2 : DaySales
  day3 : DaySales

/-- Calculates the revenue for a given day --/
def revenue (day : DaySales) : ℝ :=
  day.price * day.quantity

/-- Checks if the sales data satisfies the problem conditions --/
def satisfiesConditions (sales : ThreeDaySales) : Prop :=
  sales.day2.price = sales.day1.price - 1 ∧
  sales.day2.quantity = sales.day1.quantity + 100 ∧
  sales.day3.price = sales.day2.price + 3 ∧
  sales.day3.quantity = sales.day2.quantity - 200 ∧
  revenue sales.day1 = revenue sales.day2 ∧
  revenue sales.day2 = revenue sales.day3

/-- The main theorem: if the sales data satisfies the conditions, the initial price was 4 yuan --/
theorem initial_price_is_four (sales : ThreeDaySales) :
  satisfiesConditions sales → sales.day1.price = 4 := by
  sorry

end initial_price_is_four_l3180_318070


namespace basketball_team_score_lower_bound_l3180_318084

theorem basketball_team_score_lower_bound (n : ℕ) (player_scores : Fin n → ℕ) 
  (h1 : n = 12) 
  (h2 : ∀ i, player_scores i ≥ 7) 
  (h3 : ∀ i, player_scores i ≤ 23) : 
  (Finset.sum Finset.univ player_scores) ≥ 84 := by
  sorry

#check basketball_team_score_lower_bound

end basketball_team_score_lower_bound_l3180_318084


namespace male_listeners_count_l3180_318097

/-- Represents the survey data for Radio Wave XFM --/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Calculates the number of male listeners given the survey data --/
def male_listeners (data : SurveyData) : ℕ :=
  data.total_listeners - data.female_listeners

/-- Theorem stating that the number of male listeners is 75 --/
theorem male_listeners_count (data : SurveyData)
  (h1 : data.total_listeners = 150)
  (h2 : data.total_non_listeners = 180)
  (h3 : data.female_listeners = 75)
  (h4 : data.male_non_listeners = 84) :
  male_listeners data = 75 := by
  sorry

#eval male_listeners { total_listeners := 150, total_non_listeners := 180, female_listeners := 75, male_non_listeners := 84 }

end male_listeners_count_l3180_318097


namespace sqrt_product_simplification_l3180_318014

theorem sqrt_product_simplification (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := by
  sorry

end sqrt_product_simplification_l3180_318014


namespace no_real_solutions_l3180_318079

theorem no_real_solutions : ¬∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -abs x := by
  sorry

end no_real_solutions_l3180_318079


namespace equation_solution_l3180_318058

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ x = -15 := by
  sorry

end equation_solution_l3180_318058


namespace marble_ratio_l3180_318073

/-- Proves that the ratio of marbles Lori gave to marbles Hilton lost is 2:1 --/
theorem marble_ratio (initial : ℕ) (found : ℕ) (lost : ℕ) (final : ℕ) 
  (h_initial : initial = 26)
  (h_found : found = 6)
  (h_lost : lost = 10)
  (h_final : final = 42) :
  (final - (initial + found - lost)) / lost = 2 := by
  sorry

end marble_ratio_l3180_318073


namespace f_positive_iff_f_plus_abs_lower_bound_f_plus_abs_equality_exists_l3180_318090

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part (1)
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > 1 ∨ x < -5 := by sorry

-- Theorem for part (2)
theorem f_plus_abs_lower_bound (x : ℝ) : f x + 3*|x - 4| ≥ 9 := by sorry

-- Theorem for the existence of equality in part (2)
theorem f_plus_abs_equality_exists : ∃ x : ℝ, f x + 3*|x - 4| = 9 := by sorry

end f_positive_iff_f_plus_abs_lower_bound_f_plus_abs_equality_exists_l3180_318090


namespace watson_class_first_graders_l3180_318050

/-- The number of first graders in Ms. Watson's class -/
def first_graders (total : ℕ) (kindergartners : ℕ) (second_graders : ℕ) : ℕ :=
  total - (kindergartners + second_graders)

/-- Theorem stating the number of first graders in Ms. Watson's class -/
theorem watson_class_first_graders :
  first_graders 42 14 4 = 24 := by
  sorry

end watson_class_first_graders_l3180_318050


namespace investment_distribution_l3180_318045

/-- Investment problem with given conditions and amounts -/
theorem investment_distribution (total : ℝ) (bonds stocks mutual_funds : ℝ) : 
  total = 210000 ∧ 
  stocks = 2 * bonds ∧ 
  mutual_funds = 4 * stocks ∧ 
  total = bonds + stocks + mutual_funds →
  bonds = 19090.91 ∧ 
  stocks = 38181.82 ∧ 
  mutual_funds = 152727.27 := by
  sorry

end investment_distribution_l3180_318045


namespace divisible_by_eight_expression_l3180_318067

theorem divisible_by_eight_expression :
  ∃ (A B C : ℕ), (A % 8 ≠ 0) ∧ (B % 8 ≠ 0) ∧ (C % 8 ≠ 0) ∧
    (∀ n : ℕ, (A * 5^n + B * 3^(n-1) + C) % 8 = 0) :=
by sorry

end divisible_by_eight_expression_l3180_318067


namespace polly_tweets_l3180_318048

/-- Represents the tweet rate (tweets per minute) for different states of Polly --/
structure TweetRate where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration (in minutes) Polly spends in each state --/
structure Duration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given tweet rates and durations --/
def totalTweets (rate : TweetRate) (duration : Duration) : ℕ :=
  rate.happy * duration.happy + rate.hungry * duration.hungry + rate.mirror * duration.mirror

/-- Theorem stating that given the specific conditions, Polly tweets 1340 times --/
theorem polly_tweets : 
  ∀ (rate : TweetRate) (duration : Duration),
  rate.happy = 18 ∧ rate.hungry = 4 ∧ rate.mirror = 45 ∧
  duration.happy = 20 ∧ duration.hungry = 20 ∧ duration.mirror = 20 →
  totalTweets rate duration = 1340 := by
  sorry


end polly_tweets_l3180_318048


namespace house_renovation_time_l3180_318032

theorem house_renovation_time :
  let num_bedrooms : ℕ := 3
  let bedroom_time : ℕ := 4
  let kitchen_time : ℕ := bedroom_time + bedroom_time / 2
  let bedrooms_and_kitchen_time : ℕ := num_bedrooms * bedroom_time + kitchen_time
  let living_room_time : ℕ := 2 * bedrooms_and_kitchen_time
  let total_time : ℕ := bedrooms_and_kitchen_time + living_room_time
  total_time = 54 := by sorry

end house_renovation_time_l3180_318032


namespace quadratic_sum_l3180_318024

/-- Given a quadratic function f(x) = 8x^2 - 48x - 288, when expressed in the form a(x+b)^2 + c,
    the sum of a, b, and c is -355. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 8 * x^2 - 48 * x - 288) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = -355 := by
  sorry

end quadratic_sum_l3180_318024


namespace circle_passes_through_origin_l3180_318069

/-- Definition of the circle C with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m-1)*x + 2*(m-1)*y + 2*m^2 - 6*m + 4 = 0

/-- Theorem stating that the circle passes through the origin when m = 2 -/
theorem circle_passes_through_origin :
  ∃ m : ℝ, circle_equation 0 0 m ∧ m = 2 :=
sorry

end circle_passes_through_origin_l3180_318069


namespace sqrt_a_div_sqrt_b_l3180_318041

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (3/5)^2 + (2/7)^2 / ((2/9)^2 + (1/6)^2) = 28*a/(45*b)) :
  Real.sqrt a / Real.sqrt b = 2 * Real.sqrt 105 / 7 := by
  sorry

end sqrt_a_div_sqrt_b_l3180_318041


namespace books_remaining_after_loans_and_returns_l3180_318003

/-- Calculates the number of books remaining in a special collection after loans and returns. -/
theorem books_remaining_after_loans_and_returns 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 45 → 
  return_rate = 4/5 → 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 66 := by
  sorry

#check books_remaining_after_loans_and_returns

end books_remaining_after_loans_and_returns_l3180_318003


namespace room_length_is_ten_l3180_318054

/-- Proves that the length of a rectangular room is 10 meters given specific conditions. -/
theorem room_length_is_ten (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 42750 →
  paving_rate = 900 →
  total_cost / paving_rate / width = 10 := by
  sorry


end room_length_is_ten_l3180_318054


namespace integral_x_minus_one_l3180_318057

theorem integral_x_minus_one : ∫ x in (0 : ℝ)..2, (x - 1) = 0 := by sorry

end integral_x_minus_one_l3180_318057


namespace smallest_n_for_roots_of_unity_l3180_318061

-- Define the complex polynomial z^5 - z^3 + z
def f (z : ℂ) : ℂ := z^5 - z^3 + z

-- Define the property of being an nth root of unity
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

-- State the theorem
theorem smallest_n_for_roots_of_unity : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), f z = 0 → is_nth_root_of_unity z n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (w : ℂ), f w = 0 ∧ ¬(is_nth_root_of_unity w m)) ∧
  n = 12 :=
sorry

end smallest_n_for_roots_of_unity_l3180_318061


namespace id_number_permutations_l3180_318064

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The problem statement -/
theorem id_number_permutations :
  permutations 4 = 24 := by
  sorry

end id_number_permutations_l3180_318064


namespace inequality_proof_l3180_318020

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b*c)) + (b^3 / (c*a)) + (c^3 / (a*b)) ≥ a + b + c := by
  sorry

end inequality_proof_l3180_318020


namespace photographer_choices_l3180_318082

theorem photographer_choices (n : ℕ) (k₁ k₂ : ℕ) (h₁ : n = 7) (h₂ : k₁ = 4) (h₃ : k₂ = 5) :
  Nat.choose n k₁ + Nat.choose n k₂ = 56 := by
  sorry

end photographer_choices_l3180_318082


namespace oxygen_atoms_in_compound_l3180_318005

-- Define atomic weights
def atomic_weight_H : ℝ := 1
def atomic_weight_Br : ℝ := 79.9
def atomic_weight_O : ℝ := 16

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 129

-- Define the number of atoms for H and Br
def num_H : ℕ := 1
def num_Br : ℕ := 1

-- Theorem to prove
theorem oxygen_atoms_in_compound :
  ∃ (n : ℕ), n * atomic_weight_O = molecular_weight - (num_H * atomic_weight_H + num_Br * atomic_weight_Br) ∧ n = 3 := by
  sorry

end oxygen_atoms_in_compound_l3180_318005


namespace intersection_of_A_and_B_l3180_318065

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l3180_318065


namespace additional_carrots_is_38_l3180_318029

/-- The number of additional carrots picked by Carol and her mother -/
def additional_carrots (carol_carrots mother_carrots total_bad_carrots : ℝ) : ℝ :=
  total_bad_carrots - (carol_carrots + mother_carrots)

/-- Theorem stating that the number of additional carrots picked is 38 -/
theorem additional_carrots_is_38 :
  additional_carrots 29 16 83 = 38 := by
  sorry

end additional_carrots_is_38_l3180_318029


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3180_318000

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * r₁^2 + b * r₁ + c = 0) ∧ (a * r₂^2 + b * r₂ + c = 0) →
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (4 + Real.sqrt (16 - 12)) / 2
  let r₂ := (4 - Real.sqrt (16 - 12)) / 2
  (r₁^2 - 4*r₁ + 3 = 0) ∧ (r₂^2 - 4*r₂ + 3 = 0) →
  r₁ + r₂ = 4 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3180_318000


namespace inequality_proof_l3180_318027

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l3180_318027


namespace inequality_solution_set_l3180_318008

theorem inequality_solution_set :
  let S := {x : ℝ | (3*x + 1)*(1 - 2*x) > 0}
  S = {x : ℝ | -1/3 < x ∧ x < 1/2} := by
  sorry

end inequality_solution_set_l3180_318008


namespace percentage_problem_l3180_318068

theorem percentage_problem (total : ℝ) (part : ℝ) (h1 : total = 300) (h2 : part = 75) :
  (part / total) * 100 = 25 := by
  sorry

end percentage_problem_l3180_318068


namespace successive_discounts_equivalent_to_single_discount_l3180_318055

-- Define the original price and discount rates
def original_price : ℝ := 50
def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the equivalent single discount
def equivalent_discount : ℝ := 0.4645

-- Theorem statement
theorem successive_discounts_equivalent_to_single_discount :
  (1 - discount1) * (1 - discount2) * (1 - discount3) = 1 - equivalent_discount :=
by sorry

end successive_discounts_equivalent_to_single_discount_l3180_318055


namespace fraction_subtraction_l3180_318071

theorem fraction_subtraction : (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by
  sorry

end fraction_subtraction_l3180_318071


namespace sector_area_l3180_318092

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 8) (h2 : central_angle = 2) :
  let radius := (perimeter - central_angle * (perimeter / (2 + central_angle))) / 2
  let arc_length := central_angle * radius
  (1 / 2) * radius * arc_length = 4 := by sorry

end sector_area_l3180_318092


namespace ratio_a_to_c_l3180_318033

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end ratio_a_to_c_l3180_318033


namespace consecutive_odd_squares_difference_divisible_by_8_l3180_318077

theorem consecutive_odd_squares_difference_divisible_by_8 (n : ℤ) : 
  ∃ k : ℤ, (2*n + 3)^2 - (2*n + 1)^2 = 8 * k := by
  sorry

end consecutive_odd_squares_difference_divisible_by_8_l3180_318077


namespace min_value_z_l3180_318053

theorem min_value_z (x y : ℝ) (h1 : 2*x + 3*y - 3 ≤ 0) (h2 : 2*x - 3*y + 3 ≥ 0) (h3 : y + 3 ≥ 0) :
  ∀ z : ℝ, z = 2*x + y → z ≥ -3 ∧ ∃ x₀ y₀ : ℝ, 2*x₀ + 3*y₀ - 3 ≤ 0 ∧ 2*x₀ - 3*y₀ + 3 ≥ 0 ∧ y₀ + 3 ≥ 0 ∧ 2*x₀ + y₀ = -3 :=
by
  sorry

end min_value_z_l3180_318053


namespace parking_garage_open_spots_l3180_318031

/-- Represents the number of open parking spots on each level of a parking garage -/
structure ParkingGarage where
  first_level : ℕ
  second_level : ℕ
  third_level : ℕ
  fourth_level : ℕ

/-- Theorem stating the number of open parking spots on the first level of the parking garage -/
theorem parking_garage_open_spots (g : ParkingGarage) : g.first_level = 58 :=
  by
  have h1 : g.second_level = g.first_level + 2 := sorry
  have h2 : g.third_level = g.second_level + 5 := sorry
  have h3 : g.fourth_level = 31 := sorry
  have h4 : g.first_level + g.second_level + g.third_level + g.fourth_level = 400 - 186 := sorry
  sorry

#check parking_garage_open_spots

end parking_garage_open_spots_l3180_318031


namespace min_dot_product_OA_OP_l3180_318063

/-- The minimum dot product of OA and OP -/
theorem min_dot_product_OA_OP : ∃ (min : ℝ),
  (∀ x y : ℝ, x > 0 → y = 9 / x → (1 * x + 1 * y) ≥ min) ∧
  (∃ x y : ℝ, x > 0 ∧ y = 9 / x ∧ 1 * x + 1 * y = min) ∧
  min = 6 := by sorry

end min_dot_product_OA_OP_l3180_318063


namespace texts_sent_per_month_l3180_318098

/-- Represents the number of texts sent per month -/
def T : ℕ := sorry

/-- Represents the cost of the current plan in dollars -/
def current_plan_cost : ℕ := 12

/-- Represents the cost per 30 texts in dollars -/
def text_cost_per_30 : ℕ := 1

/-- Represents the cost per 20 minutes of calls in dollars -/
def call_cost_per_20_min : ℕ := 3

/-- Represents the number of minutes spent on calls per month -/
def call_minutes : ℕ := 60

/-- Represents the cost difference between current and alternative plans in dollars -/
def cost_difference : ℕ := 1

theorem texts_sent_per_month :
  T = 60 ∧
  (T / 30 : ℚ) * text_cost_per_30 + 
  (call_minutes / 20 : ℚ) * call_cost_per_20_min = 
  current_plan_cost - cost_difference :=
by sorry

end texts_sent_per_month_l3180_318098


namespace matching_pair_probability_for_sue_l3180_318001

/-- Represents the number of pairs of shoes for each color --/
structure ShoePairs :=
  (black : Nat)
  (brown : Nat)
  (gray : Nat)
  (red : Nat)

/-- Calculates the probability of picking a matching pair of shoes --/
def matchingPairProbability (shoes : ShoePairs) : Rat :=
  let totalShoes := 2 * (shoes.black + shoes.brown + shoes.gray + shoes.red)
  let matchingPairs := 
    shoes.black * (shoes.black - 1) + 
    shoes.brown * (shoes.brown - 1) + 
    shoes.gray * (shoes.gray - 1) + 
    shoes.red * (shoes.red - 1)
  matchingPairs / (totalShoes * (totalShoes - 1))

theorem matching_pair_probability_for_sue : 
  let sueShoes : ShoePairs := { black := 7, brown := 4, gray := 3, red := 2 }
  matchingPairProbability sueShoes = 39 / 248 := by
  sorry

end matching_pair_probability_for_sue_l3180_318001


namespace find_a_l3180_318040

-- Define the complex numbers a, b, c
variable (a b c : ℂ)

-- Define the conditions
def condition1 : Prop := a + b + c = 5
def condition2 : Prop := a * b + b * c + c * a = 7
def condition3 : Prop := a * b * c = 6
def condition4 : Prop := a.im = 0  -- a is real

-- Theorem statement
theorem find_a (h1 : condition1 a b c) (h2 : condition2 a b c) 
                (h3 : condition3 a b c) (h4 : condition4 a) : 
  a = 1 := by sorry

end find_a_l3180_318040


namespace steel_rusting_not_LeChatelier_l3180_318039

/-- Le Chatelier's principle states that if a change in conditions is imposed on a system at equilibrium, 
    the equilibrium will shift in a direction that tends to reduce that change. -/
def LeChatelier_principle : Prop := sorry

/-- Rusting of steel in humid air -/
def steel_rusting : Prop := sorry

/-- A chemical process that can be explained by Le Chatelier's principle -/
def explainable_by_LeChatelier (process : Prop) : Prop := sorry

theorem steel_rusting_not_LeChatelier : 
  ¬(explainable_by_LeChatelier steel_rusting) := by sorry

end steel_rusting_not_LeChatelier_l3180_318039


namespace ad_cost_per_square_inch_l3180_318007

/-- Proves that the cost per square inch for advertising is $8 --/
theorem ad_cost_per_square_inch :
  let page_length : ℝ := 9
  let page_width : ℝ := 12
  let full_page_area : ℝ := page_length * page_width
  let ad_area : ℝ := full_page_area / 2
  let total_cost : ℝ := 432
  let cost_per_square_inch : ℝ := total_cost / ad_area
  cost_per_square_inch = 8 := by
  sorry

end ad_cost_per_square_inch_l3180_318007


namespace probability_calculation_l3180_318081

/-- Represents a bag of bills -/
structure Bag where
  tens : ℕ
  fives : ℕ
  ones : ℕ

/-- Calculates the total value of bills in a bag -/
def bagValue (b : Bag) : ℕ := 10 * b.tens + 5 * b.fives + b.ones

/-- Calculates the number of ways to choose 2 bills from a bag -/
def chooseTwo (b : Bag) : ℕ := (b.tens + b.fives + b.ones) * (b.tens + b.fives + b.ones - 1) / 2

/-- Calculates the probability of the sum of remaining bills in bag A being greater than the sum of remaining bills in bag B -/
def probabilityAGreaterThanB (bagA bagB : Bag) : ℚ :=
  let totalOutcomes := chooseTwo bagA * chooseTwo bagB
  let favorableOutcomes := 3 * 18  -- This is a simplification based on the problem's specific conditions
  ↑favorableOutcomes / ↑totalOutcomes

theorem probability_calculation (bagA bagB : Bag) 
  (hA : bagA = ⟨2, 0, 3⟩) 
  (hB : bagB = ⟨0, 4, 3⟩) : 
  probabilityAGreaterThanB bagA bagB = 9/35 := by
  sorry

#eval probabilityAGreaterThanB ⟨2, 0, 3⟩ ⟨0, 4, 3⟩

end probability_calculation_l3180_318081


namespace product_trailing_zeros_l3180_318052

/-- The number of trailing zeros in base 12 for the product 33 * 59 -/
def trailing_zeros_base_12 : ℕ := 2

/-- The product we're working with -/
def product : ℕ := 33 * 59

/-- Conversion to base 12 -/
def to_base_12 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 12) ((m % 12) :: acc)
    aux n []

/-- Count trailing zeros in a list of digits -/
def count_trailing_zeros (digits : List ℕ) : ℕ :=
  digits.reverse.takeWhile (· = 0) |>.length

theorem product_trailing_zeros :
  count_trailing_zeros (to_base_12 product) = trailing_zeros_base_12 := by
  sorry

#eval to_base_12 product
#eval count_trailing_zeros (to_base_12 product)

end product_trailing_zeros_l3180_318052


namespace product_minus_quotient_l3180_318096

theorem product_minus_quotient : 11 * 13 * 17 - 33 / 3 = 2420 := by
  sorry

end product_minus_quotient_l3180_318096


namespace hyperbola_focal_distance_l3180_318042

/-- The focal distance of a hyperbola with equation x²/20 - y²/5 = 1 is 10 -/
theorem hyperbola_focal_distance : 
  ∃ (c : ℝ), c > 0 ∧ c^2 = 25 ∧ 2*c = 10 :=
by sorry

end hyperbola_focal_distance_l3180_318042


namespace average_age_problem_l3180_318094

theorem average_age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 29 →
  b = 23 →
  (a + c) / 2 = 32 := by
sorry

end average_age_problem_l3180_318094


namespace cricket_target_runs_l3180_318009

/-- Calculates the target number of runs in a cricket game -/
def targetRuns (totalOvers runRateFirst8 runRateRemaining : ℕ) : ℕ :=
  let runsFirst8 := (runRateFirst8 * 8) / 10
  let runsRemaining := (runRateRemaining * 20) / 10
  runsFirst8 + runsRemaining

/-- Theorem stating the target number of runs for the given conditions -/
theorem cricket_target_runs :
  targetRuns 28 23 120 = 259 := by
  sorry

#eval targetRuns 28 23 120

end cricket_target_runs_l3180_318009


namespace prime_sequence_finite_l3180_318060

/-- A sequence of primes satisfying the given conditions -/
def PrimeSequence (p : ℕ → ℕ) : Prop :=
  (∀ n, Nat.Prime (p n)) ∧ 
  (∀ i ≥ 2, p i = 2 * p (i-1) - 1 ∨ p i = 2 * p (i-1) + 1)

/-- The theorem stating that any such sequence is finite -/
theorem prime_sequence_finite (p : ℕ → ℕ) (h : PrimeSequence p) : 
  ∃ N, ∀ n > N, ¬ Nat.Prime (p n) :=
sorry

end prime_sequence_finite_l3180_318060


namespace negative_one_to_zero_equals_one_l3180_318035

theorem negative_one_to_zero_equals_one :
  (-1 : ℝ) ^ (0 : ℝ) = 1 := by sorry

end negative_one_to_zero_equals_one_l3180_318035


namespace cube_volume_from_surface_area_l3180_318016

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 → volume = (surface_area / 6) ^ (3/2) → volume = 125 := by
  sorry

end cube_volume_from_surface_area_l3180_318016


namespace gloria_ticket_boxes_l3180_318085

/-- Given that Gloria has 45 tickets and each box holds 5 tickets,
    prove that the number of boxes Gloria has is 9. -/
theorem gloria_ticket_boxes : ∀ (total_tickets boxes_count tickets_per_box : ℕ),
  total_tickets = 45 →
  tickets_per_box = 5 →
  total_tickets = boxes_count * tickets_per_box →
  boxes_count = 9 := by
  sorry

end gloria_ticket_boxes_l3180_318085


namespace pen_distribution_l3180_318038

theorem pen_distribution (num_students : ℕ) (red_pens : ℕ) (black_pens : ℕ) 
  (month1 : ℕ) (month2 : ℕ) (month3 : ℕ) (month4 : ℕ) : 
  num_students = 6 →
  red_pens = 85 →
  black_pens = 92 →
  month1 = 77 →
  month2 = 89 →
  month3 = 102 →
  month4 = 68 →
  (num_students * (red_pens + black_pens) - (month1 + month2 + month3 + month4)) / num_students = 121 := by
  sorry

#check pen_distribution

end pen_distribution_l3180_318038


namespace max_intersection_points_l3180_318072

-- Define a circle on a plane
def Circle : Type := Unit

-- Define a line on a plane
def Line : Type := Unit

-- Function to count intersection points between a circle and a line
def circleLineIntersections (c : Circle) (l : Line) : ℕ := 2

-- Function to count intersection points between two lines
def lineLineIntersections (l1 l2 : Line) : ℕ := 1

-- Theorem stating the maximum number of intersection points
theorem max_intersection_points (c : Circle) (l1 l2 l3 : Line) :
  ∃ (n : ℕ), n ≤ 9 ∧ 
  (∀ (m : ℕ), m ≤ circleLineIntersections c l1 + 
               circleLineIntersections c l2 + 
               circleLineIntersections c l3 + 
               lineLineIntersections l1 l2 + 
               lineLineIntersections l1 l3 + 
               lineLineIntersections l2 l3 → m ≤ n) :=
by sorry

end max_intersection_points_l3180_318072


namespace kyle_track_laps_l3180_318076

/-- The number of laps Kyle jogged in P.E. class -/
def pe_laps : ℝ := 1.12

/-- The total number of laps Kyle jogged -/
def total_laps : ℝ := 3.25

/-- The number of laps Kyle jogged during track practice -/
def track_laps : ℝ := total_laps - pe_laps

theorem kyle_track_laps : track_laps = 2.13 := by
  sorry

end kyle_track_laps_l3180_318076


namespace line_relation_in_plane_l3180_318087

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and subset relations
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the positional relationships between lines
inductive LineRelation : Type
  | parallel : LineRelation
  | skew : LineRelation
  | intersecting : LineRelation

-- Define the theorem
theorem line_relation_in_plane (a b : Line) (α : Plane) 
  (h1 : parallel a α) (h2 : subset b α) :
  (∃ r : LineRelation, r = LineRelation.parallel ∨ r = LineRelation.skew) ∧
  ¬(∃ r : LineRelation, r = LineRelation.intersecting) :=
sorry

end line_relation_in_plane_l3180_318087


namespace video_game_time_increase_l3180_318086

/-- Calculates the percentage increase in video game time given the original rate,
    total reading time, and additional time after raise. -/
theorem video_game_time_increase
  (original_rate : ℕ)  -- Original minutes of video game time per hour of reading
  (reading_time : ℕ)   -- Total hours of reading
  (additional_time : ℕ) -- Additional minutes of video game time after raise
  (h1 : original_rate = 30)
  (h2 : reading_time = 12)
  (h3 : additional_time = 72) :
  (additional_time : ℚ) / (original_rate * reading_time) * 100 = 20 := by
  sorry

end video_game_time_increase_l3180_318086


namespace range_of_a_l3180_318062

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def B : Set ℝ := {x | (x - 2) * (3 - x) > 0}

-- Define the proposition p and q
def p (a : ℝ) (x : ℝ) : Prop := x ∈ A a
def q (x : ℝ) : Prop := x ∈ B

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(p a x) → ¬(q x)) →
  a ∈ Set.Icc (-1 : ℝ) 6 :=
sorry

end range_of_a_l3180_318062


namespace footprint_calculation_l3180_318056

/-- Calculates the total number of footprints left by Pogo and Grimzi -/
def total_footprints (pogo_rate : ℚ) (grimzi_rate : ℚ) (distance : ℚ) : ℚ :=
  pogo_rate * distance + grimzi_rate * distance

/-- Theorem stating the total number of footprints left by Pogo and Grimzi -/
theorem footprint_calculation :
  let pogo_rate : ℚ := 4
  let grimzi_rate : ℚ := 1/2
  let distance : ℚ := 6000
  total_footprints pogo_rate grimzi_rate distance = 27000 := by
sorry

#eval total_footprints 4 (1/2) 6000

end footprint_calculation_l3180_318056


namespace snake_count_l3180_318074

/-- The number of snakes counted at the zoo --/
def snakes : ℕ := sorry

/-- The number of arctic foxes counted at the zoo --/
def arctic_foxes : ℕ := 80

/-- The number of leopards counted at the zoo --/
def leopards : ℕ := 20

/-- The number of bee-eaters counted at the zoo --/
def bee_eaters : ℕ := 10 * leopards

/-- The number of cheetahs counted at the zoo --/
def cheetahs : ℕ := snakes / 2

/-- The number of alligators counted at the zoo --/
def alligators : ℕ := 2 * (arctic_foxes + leopards)

/-- The total number of animals counted at the zoo --/
def total_animals : ℕ := 670

/-- Theorem stating that the number of snakes counted is 113 --/
theorem snake_count : snakes = 113 := by sorry

end snake_count_l3180_318074


namespace anthony_transaction_percentage_l3180_318066

/-- Proves that Anthony handled 10% more transactions than Mabel given the conditions in the problem. -/
theorem anthony_transaction_percentage (mabel_transactions cal_transactions jade_transactions : ℕ) 
  (anthony_transactions : ℕ) (anthony_percentage : ℚ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 19 →
  jade_transactions = 85 →
  anthony_transactions = mabel_transactions * (1 + anthony_percentage / 100) →
  anthony_percentage = 10 := by
  sorry

end anthony_transaction_percentage_l3180_318066


namespace factorization_equality_l3180_318083

theorem factorization_equality (x : ℝ) : 12 * x^2 + 18 * x - 24 = 6 * (2 * x - 1) * (x + 4) := by
  sorry

end factorization_equality_l3180_318083


namespace perimeter_approx_40_l3180_318047

/-- Represents a figure composed of three squares and one rectangle -/
structure CompositeFigure where
  square_side : ℝ
  total_area : ℝ

/-- Checks if the CompositeFigure satisfies the given conditions -/
def is_valid_figure (f : CompositeFigure) : Prop :=
  f.total_area = 150 ∧ 
  3 * f.square_side^2 + 2 * f.square_side^2 = f.total_area

/-- Calculates the perimeter of the CompositeFigure -/
def perimeter (f : CompositeFigure) : ℝ :=
  8 * f.square_side

/-- Theorem stating that the perimeter of a valid CompositeFigure is approximately 40 -/
theorem perimeter_approx_40 (f : CompositeFigure) (h : is_valid_figure f) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (perimeter f - 40) < ε :=
sorry

end perimeter_approx_40_l3180_318047


namespace table_area_proof_l3180_318043

theorem table_area_proof (total_runner_area : ℝ) (coverage_percentage : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_runner_area = 224) 
  (h2 : coverage_percentage = 0.8)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 30) : 
  ∃ (table_area : ℝ), table_area = 175 ∧ 
    coverage_percentage * table_area = 
      (total_runner_area - 2 * two_layer_area - 3 * three_layer_area) + 
      two_layer_area + three_layer_area := by
  sorry

end table_area_proof_l3180_318043


namespace monthly_rent_is_400_l3180_318046

/-- Calculates the monthly rent per resident in a rental building -/
def monthly_rent_per_resident (total_units : ℕ) (occupancy_rate : ℚ) (total_annual_rent : ℕ) : ℚ :=
  let occupied_units : ℚ := total_units * occupancy_rate
  let annual_rent_per_resident : ℚ := total_annual_rent / occupied_units
  annual_rent_per_resident / 12

/-- Proves that the monthly rent per resident is $400 -/
theorem monthly_rent_is_400 :
  monthly_rent_per_resident 100 (3/4) 360000 = 400 := by
  sorry

end monthly_rent_is_400_l3180_318046


namespace sentence_B_is_correct_l3180_318091

/-- Represents a sentence in English --/
structure Sentence where
  text : String

/-- Checks if a sentence is grammatically correct --/
def is_grammatically_correct (s : Sentence) : Prop := sorry

/-- The four sentences given in the problem --/
def sentence_A : Sentence := { text := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams." }

def sentence_B : Sentence := { text := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." }

def sentence_C : Sentence := { text := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high." }

def sentence_D : Sentence := { text := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves." }

/-- Theorem stating that sentence B is grammatically correct --/
theorem sentence_B_is_correct :
  is_grammatically_correct sentence_B ∧
  ¬is_grammatically_correct sentence_A ∧
  ¬is_grammatically_correct sentence_C ∧
  ¬is_grammatically_correct sentence_D :=
by
  sorry

end sentence_B_is_correct_l3180_318091


namespace smallest_solution_congruence_l3180_318013

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 22 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 22 % 31 → x ≤ y :=
by sorry

end smallest_solution_congruence_l3180_318013


namespace select_subjects_with_distinct_grades_l3180_318095

/-- Represents a grade for a single subject -/
def Grade : Type := ℕ

/-- Represents the grades of a student for all subjects -/
def StudentGrades : Type := Fin 12 → Grade

/-- The number of students -/
def numStudents : ℕ := 7

/-- The number of subjects -/
def numSubjects : ℕ := 12

/-- The number of subjects to be selected -/
def numSelectedSubjects : ℕ := 6

theorem select_subjects_with_distinct_grades 
  (grades : Fin numStudents → StudentGrades)
  (h : ∀ i j, i ≠ j → ∃ k, grades i k ≠ grades j k) :
  ∃ (selected : Fin numSelectedSubjects → Fin numSubjects),
    (∀ i j, i ≠ j → ∃ k, grades i (selected k) ≠ grades j (selected k)) :=
sorry

end select_subjects_with_distinct_grades_l3180_318095


namespace number_of_divisors_180_l3180_318026

theorem number_of_divisors_180 : ∃ (n : ℕ), n = 18 ∧ 
  (∀ d : ℕ, d > 0 ∧ (180 % d = 0) ↔ d ∈ Finset.range n) :=
by
  sorry

end number_of_divisors_180_l3180_318026


namespace ceiling_product_equation_l3180_318051

theorem ceiling_product_equation (x : ℝ) : 
  ⌈x⌉ * x = 198 ↔ x = 13.2 := by sorry

end ceiling_product_equation_l3180_318051


namespace geometric_progression_solution_l3180_318036

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  b₁ : ℚ  -- First term
  q : ℚ   -- Common ratio

/-- The nth term of a geometric progression -/
def GeometricProgression.nthTerm (gp : GeometricProgression) (n : ℕ) : ℚ :=
  gp.b₁ * gp.q ^ (n - 1)

theorem geometric_progression_solution :
  ∃ (gp : GeometricProgression),
    gp.nthTerm 3 = -1 ∧
    gp.nthTerm 6 = 27/8 ∧
    gp.b₁ = -4/9 ∧
    gp.q = -3/2 := by
  sorry

end geometric_progression_solution_l3180_318036


namespace square_area_difference_l3180_318099

def original_side_length : ℝ := 6
def increase_in_length : ℝ := 1

theorem square_area_difference :
  let new_side_length := original_side_length + increase_in_length
  let original_area := original_side_length ^ 2
  let new_area := new_side_length ^ 2
  new_area - original_area = 13 := by sorry

end square_area_difference_l3180_318099


namespace sqrt_360000_equals_600_l3180_318049

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_equals_600_l3180_318049


namespace max_toys_buyable_l3180_318023

def initial_amount : ℕ := 57
def game_cost : ℕ := 27
def toy_cost : ℕ := 6

theorem max_toys_buyable : 
  (initial_amount - game_cost) / toy_cost = 5 :=
by sorry

end max_toys_buyable_l3180_318023


namespace complement_of_A_in_U_l3180_318088

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 5} := by sorry

end complement_of_A_in_U_l3180_318088


namespace geometric_sequence_properties_l3180_318037

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum : a 2 + a 3 + a 4 = 28)
  (h_mean : a 3 + 2 = (a 2 + a 4) / 2) :
  (∃ q : ℝ, ∀ n : ℕ, a n = (1/2)^(n - 6)) ∧
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = (1/2) * a n) :=
sorry

end geometric_sequence_properties_l3180_318037


namespace expression_evaluation_l3180_318011

theorem expression_evaluation (m : ℝ) (h : m = -Real.sqrt 5) :
  (2 * m - 1)^2 - (m - 5) * (m + 1) = 21 := by
  sorry

end expression_evaluation_l3180_318011


namespace toy_robot_shipment_l3180_318025

theorem toy_robot_shipment (displayed_percentage : ℚ) (stored : ℕ) : 
  displayed_percentage = 30 / 100 →
  stored = 140 →
  (1 - displayed_percentage) * 200 = stored :=
by sorry

end toy_robot_shipment_l3180_318025


namespace clock_angle_120_elapsed_time_l3180_318010

/-- Represents the angle between clock hands at a given time --/
def clockAngle (hours minutes : ℝ) : ℝ :=
  (30 * hours + 0.5 * minutes) - (6 * minutes)

/-- Finds the time when the clock hands form a 120° angle after 6:00 PM --/
def findNextAngle120 : ℝ :=
  let f := fun t : ℝ => abs (clockAngle (6 + t / 60) (t % 60) - 120)
  sorry -- Minimize f(t) for 0 ≤ t < 60

theorem clock_angle_120_elapsed_time :
  ∃ t : ℝ, 0 < t ∧ t < 60 ∧ 
  abs (clockAngle 6 0 - 120) < 0.01 ∧
  abs (clockAngle (6 + t / 60) (t % 60) - 120) < 0.01 ∧
  abs (t - 43.64) < 0.01 :=
sorry

end clock_angle_120_elapsed_time_l3180_318010


namespace min_value_2a_plus_b_l3180_318019

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 4) :
  2 * a + b ≥ 2 * Real.sqrt 3 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a * b * (a + b) = 4 ∧ 2 * a + b = 2 * Real.sqrt 3 :=
by sorry

end min_value_2a_plus_b_l3180_318019


namespace existence_of_m_satisfying_inequality_l3180_318017

theorem existence_of_m_satisfying_inequality (a t : ℝ) 
  (ha : a ∈ Set.Icc (-1 : ℝ) 1)
  (ht : t ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ m : ℝ, (∀ x₁ x₂ : ℝ, 
    x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    (4 * x₁ + a * x₁^2 - (2/3) * x₁^3 = 2 * x₁ + (1/3) * x₁^3) ∧
    (4 * x₂ + a * x₂^2 - (2/3) * x₂^3 = 2 * x₂ + (1/3) * x₂^3) →
    m^2 + t * m + 1 ≥ |x₁ - x₂|) ∧
  (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end existence_of_m_satisfying_inequality_l3180_318017


namespace brown_shirts_count_l3180_318075

def initial_blue_shirts : ℕ := 26

def remaining_blue_shirts : ℕ := initial_blue_shirts / 2

theorem brown_shirts_count (initial_brown_shirts : ℕ) : 
  remaining_blue_shirts + (initial_brown_shirts - initial_brown_shirts / 3) = 37 →
  initial_brown_shirts = 36 := by
  sorry

end brown_shirts_count_l3180_318075


namespace min_S_independent_of_P_l3180_318089

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = x² + c -/
structure Parabola where
  c : ℝ

/-- Represents the area bounded by a line and a parabola -/
def boundedArea (p₁ p₂ : Point) (C : Parabola) : ℝ := sorry

/-- The sum of areas S as described in the problem -/
def S (P : Point) (C₁ C₂ : Parabola) (m : ℕ) : ℝ := sorry

/-- The minimum value of S -/
def minS (m : ℕ) : ℝ := sorry

theorem min_S_independent_of_P (m : ℕ) :
  ∀ P : Point, P.y = P.x^2 + m^2 → minS m = m^3 / 3 := by sorry

end min_S_independent_of_P_l3180_318089


namespace correct_num_pants_purchased_l3180_318059

/-- Represents the purchase and refund scenario at a clothing retailer -/
structure ClothingPurchase where
  shirtPrice : ℝ
  pantsPrice : ℝ
  totalCost : ℝ
  refundRate : ℝ
  numShirts : ℕ

/-- The number of pairs of pants purchased given the conditions -/
def numPantsPurchased (purchase : ClothingPurchase) : ℕ :=
  1

theorem correct_num_pants_purchased (purchase : ClothingPurchase) 
  (h1 : purchase.shirtPrice ≠ purchase.pantsPrice)
  (h2 : purchase.shirtPrice = 45)
  (h3 : purchase.numShirts = 2)
  (h4 : purchase.totalCost = 120)
  (h5 : purchase.refundRate = 0.25)
  : numPantsPurchased purchase = 1 := by
  sorry

#check correct_num_pants_purchased

end correct_num_pants_purchased_l3180_318059


namespace dune_buggy_speed_l3180_318012

theorem dune_buggy_speed (S : ℝ) : 
  (1/3 : ℝ) * S + (1/3 : ℝ) * (S + 12) + (1/3 : ℝ) * (S - 18) = 58 → S = 60 := by
  sorry

end dune_buggy_speed_l3180_318012


namespace triangle_side_length_l3180_318030

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  distance t.A t.B = 7 →
  distance t.A t.C = 5 →
  angle t.A t.C t.B = 2 * Real.pi / 3 →
  distance t.B t.C = 3 := by
  sorry

end triangle_side_length_l3180_318030


namespace greatest_x_value_l3180_318004

theorem greatest_x_value (x : ℝ) : 
  ((5*x - 20)^2 / (4*x - 5)^2 + (5*x - 20) / (4*x - 5) = 12) → 
  x ≤ 40/21 :=
by sorry

end greatest_x_value_l3180_318004


namespace system_solution_proof_l3180_318022

theorem system_solution_proof (x y z : ℝ) : 
  x = 0.38 ∧ y = 0.992 ∧ z = -0.7176 →
  4 * x - 6 * y + 2 * z = -3 ∧
  8 * x + 3 * y - z = 5.3 ∧
  -x + 4 * y + 5 * z = 0 := by
sorry

end system_solution_proof_l3180_318022


namespace first_expression_second_expression_l3180_318002

-- First expression
theorem first_expression (x y : ℝ) : (-3 * x^2 * y)^3 = -27 * x^6 * y^3 := by sorry

-- Second expression
theorem second_expression (a : ℝ) : (-2*a - 1) * (2*a - 1) = 1 - 4*a^2 := by sorry

end first_expression_second_expression_l3180_318002


namespace equation_satisfaction_l3180_318044

theorem equation_satisfaction (a b c : ℤ) :
  a = c ∧ b - 1 = a →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end equation_satisfaction_l3180_318044


namespace cos_negative_sixty_degrees_l3180_318034

theorem cos_negative_sixty_degrees : Real.cos (-(60 * π / 180)) = 1 / 2 := by
  sorry

end cos_negative_sixty_degrees_l3180_318034


namespace paintable_area_is_1572_l3180_318018

/-- Calculate the total paintable wall area for multiple identical bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let total_wall_area := 2 * (length * height + width * height)
  let paintable_area_per_room := total_wall_area - unpaintable_area
  num_rooms * paintable_area_per_room

/-- Theorem stating that the total paintable wall area for the given conditions is 1572 square feet -/
theorem paintable_area_is_1572 :
  total_paintable_area 4 15 11 9 75 = 1572 := by
  sorry

#eval total_paintable_area 4 15 11 9 75

end paintable_area_is_1572_l3180_318018


namespace fraction_simplification_l3180_318093

theorem fraction_simplification (c d : ℝ) : 
  (5 + 4 * c - 3 * d) / 9 + 5 = (50 + 4 * c - 3 * d) / 9 := by
  sorry

end fraction_simplification_l3180_318093
