import Mathlib

namespace son_l167_16771

theorem son's_age (S F : ℕ) (h1 : F = S + 26) (h2 : F + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_l167_16771


namespace at_least_six_on_circle_l167_16707

-- Defining the types for point and circle
variable (Point : Type)
variable (Circle : Type)

-- Assuming the existence of a well-defined predicate that checks whether points lie on the same circle
variable (lies_on_circle : Circle → Point → Prop)
variable (exists_circle : Point → Point → Point → Point → Circle)
variable (five_points_condition : ∀ (p1 p2 p3 p4 p5 : Point), 
  ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                   lies_on_circle c p3 ∧ lies_on_circle c p4)

-- Given 13 points on a plane
variables (P : List Point)
variable (length_P : P.length = 13)

-- The main theorem statement
theorem at_least_six_on_circle : 
  (∀ (P : List Point) (h : P.length = 13),
    (∀ p1 p2 p3 p4 p5 : Point, ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                               lies_on_circle c p3 ∧ lies_on_circle c p4)) →
    (∃ (c : Circle), ∃ (l : List Point), l.length ≥ 6 ∧ ∀ p ∈ l, lies_on_circle c p) :=
sorry

end at_least_six_on_circle_l167_16707


namespace string_cuts_l167_16769

theorem string_cuts (L S : ℕ) (h_diff : L - S = 48) (h_sum : L + S = 64) : 
  (L / S) = 7 :=
by
  sorry

end string_cuts_l167_16769


namespace polygon_area_is_1008_l167_16711

variables (vertices : List (ℕ × ℕ)) (units : ℕ)

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
sorry -- The function would compute the area based on vertices.

theorem polygon_area_is_1008 :
  vertices = [(0, 0), (12, 0), (24, 12), (24, 0), (36, 0), (36, 24), (24, 36), (12, 36), (0, 36), (0, 24), (0, 0)] →
  units = 1 →
  polygon_area vertices = 1008 :=
sorry

end polygon_area_is_1008_l167_16711


namespace starting_player_can_ensure_integer_roots_l167_16718

theorem starting_player_can_ensure_integer_roots :
  ∃ (a b c : ℤ), ∀ (x : ℤ), (x^3 + a * x^2 + b * x + c = 0) →
  (∃ r1 r2 r3 : ℤ, x = r1 ∨ x = r2 ∨ x = r3) :=
sorry

end starting_player_can_ensure_integer_roots_l167_16718


namespace Sandy_tokens_difference_l167_16710

theorem Sandy_tokens_difference :
  let total_tokens : ℕ := 1000000
  let siblings : ℕ := 4
  let Sandy_tokens : ℕ := total_tokens / 2
  let sibling_tokens : ℕ := Sandy_tokens / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  sorry

end Sandy_tokens_difference_l167_16710


namespace tank_capacity_l167_16714

theorem tank_capacity (T : ℝ) (h : 0.4 * T = 0.9 * T - 36) : T = 72 := by
  sorry

end tank_capacity_l167_16714


namespace final_price_correct_l167_16776

-- Define the initial price of the iPhone
def initial_price : ℝ := 1000

-- Define the discount rates for the first and second month
def first_month_discount : ℝ := 0.10
def second_month_discount : ℝ := 0.20

-- Calculate the price after the first month's discount
def price_after_first_month (price : ℝ) : ℝ := price * (1 - first_month_discount)

-- Calculate the price after the second month's discount
def price_after_second_month (price : ℝ) : ℝ := price * (1 - second_month_discount)

-- Final price calculation after both discounts
def final_price : ℝ := price_after_second_month (price_after_first_month initial_price)

-- Proof statement
theorem final_price_correct : final_price = 720 := by
  sorry

end final_price_correct_l167_16776


namespace part1_part2_l167_16701

-- Defining set A
def A : Set ℝ := {x | x^2 + 4 * x = 0}

-- Defining set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

-- Problem 1: Prove that if A ∩ B = A ∪ B, then a = 1
theorem part1 (a : ℝ) : (A ∩ (B a) = A ∪ (B a)) → a = 1 := by
  sorry

-- Problem 2: Prove the range of values for a if A ∩ B = B
theorem part2 (a : ℝ) : (A ∩ (B a) = B a) → a ∈ Set.Iic (-1) ∪ {1} := by
  sorry

end part1_part2_l167_16701


namespace Jorge_goals_total_l167_16708

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end Jorge_goals_total_l167_16708


namespace find_principal_amount_l167_16729

noncomputable def principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (SI * 100) / (R * T)

theorem find_principal_amount :
  principal_amount 130 4.166666666666667 4 = 780 :=
by
  -- Sorry is used to denote that the proof is yet to be provided
  sorry

end find_principal_amount_l167_16729


namespace find_a2_l167_16796

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry -- Define the geometric sequence

variable (a1 : ℝ) (a3a5_eq : ℝ) -- Variables for given conditions

-- Main theorem statement
theorem find_a2 (h_geo : ∀ n, geometric_sequence n = a1 * (2 : ℝ) ^ (n - 1))
  (h_a1 : a1 = 1 / 4)
  (h_a3a5 : (geometric_sequence 3) * (geometric_sequence 5) = 4 * (geometric_sequence 4 - 1)) :
  geometric_sequence 2 = 1 / 2 :=
sorry  -- Proof is omitted

end find_a2_l167_16796


namespace park_area_l167_16741

theorem park_area (length breadth : ℝ) (x : ℝ) 
  (h1 : length = 3 * x) 
  (h2 : breadth = x) 
  (h3 : 2 * length + 2 * breadth = 800) 
  (h4 : 12 * (4 / 60) * 1000 = 800) : 
  length * breadth = 30000 := by
sorry

end park_area_l167_16741


namespace find_value_l167_16731

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom explicit_form : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem statement
theorem find_value : f (-5/2) = -1/2 :=
by
  -- Here would be the place to start the proof based on the above axioms
  sorry

end find_value_l167_16731


namespace min_value_x_plus_2_div_x_l167_16751

theorem min_value_x_plus_2_div_x (x : ℝ) (hx : x > 0) : x + 2 / x ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2_div_x_l167_16751


namespace area_of_given_triangle_l167_16770

noncomputable def area_of_triangle (a A B : ℝ) : ℝ :=
  let C := Real.pi - A - B
  let b := a * (Real.sin B / Real.sin A)
  let S := (1 / 2) * a * b * Real.sin C
  S

theorem area_of_given_triangle : area_of_triangle 4 (Real.pi / 4) (Real.pi / 3) = 6 + 2 * Real.sqrt 3 := 
by 
  sorry

end area_of_given_triangle_l167_16770


namespace prove_weight_of_a_l167_16722

noncomputable def weight_proof (A B C D : ℝ) : Prop :=
  (A + B + C) / 3 = 60 ∧
  50 ≤ A ∧ A ≤ 80 ∧
  50 ≤ B ∧ B ≤ 80 ∧
  50 ≤ C ∧ C ≤ 80 ∧
  60 ≤ D ∧ D ≤ 90 ∧
  (A + B + C + D) / 4 = 65 ∧
  70 ≤ D + 3 ∧ D + 3 ≤ 100 ∧
  (B + C + D + (D + 3)) / 4 = 64 → 
  A = 87

-- Adding a theorem statement to make it clear we need to prove this.
theorem prove_weight_of_a (A B C D : ℝ) : weight_proof A B C D :=
sorry

end prove_weight_of_a_l167_16722


namespace man_son_age_ratio_l167_16777

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end man_son_age_ratio_l167_16777


namespace intersection_eq_expected_result_l167_16752

def M := { x : ℝ | x - 2 > 0 }
def N := { x : ℝ | (x - 3) * (x - 1) < 0 }
def expected_result := { x : ℝ | 2 < x ∧ x < 3 }

theorem intersection_eq_expected_result : M ∩ N = expected_result := 
by
  sorry

end intersection_eq_expected_result_l167_16752


namespace sum_of_angles_is_540_l167_16786

variables (angle1 angle2 angle3 angle4 angle5 angle6 angle7 : ℝ)

theorem sum_of_angles_is_540
  (h : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540 :=
sorry

end sum_of_angles_is_540_l167_16786


namespace investment_ratio_l167_16791

theorem investment_ratio 
  (P Q : ℝ) 
  (profitP profitQ : ℝ)
  (h1 : profitP = 7 * (profitP + profitQ) / 17) 
  (h2 : profitQ = 10 * (profitP + profitQ) / 17)
  (tP : ℝ := 10)
  (tQ : ℝ := 20) 
  (h3 : profitP / profitQ = (P * tP) / (Q * tQ)) :
  P / Q = 7 / 5 := 
sorry

end investment_ratio_l167_16791


namespace theatre_lost_revenue_l167_16792

def ticket_price (category : String) : Float :=
  match category with
  | "general" => 10.0
  | "children" => 6.0
  | "senior" => 8.0
  | "veteran" => 8.0  -- $10.00 - $2.00 discount
  | _ => 0.0

def vip_price (base_price : Float) : Float :=
  base_price + 5.0

def calculate_revenue_sold : Float :=
  let general_revenue := 12 * ticket_price "general" + 3 * (vip_price $ ticket_price "general") / 2
  let children_revenue := 3 * ticket_price "children" + vip_price (ticket_price "children")
  let senior_revenue := 4 * ticket_price "senior" + (vip_price (ticket_price "senior")) / 2
  let veteran_revenue := 2 * ticket_price "veteran" + vip_price (ticket_price "veteran")
  general_revenue + children_revenue + senior_revenue + veteran_revenue

def potential_total_revenue : Float :=
  40 * ticket_price "general" + 10 * vip_price (ticket_price "general")

def potential_revenue_lost : Float :=
  potential_total_revenue - calculate_revenue_sold

theorem theatre_lost_revenue : potential_revenue_lost = 224.0 :=
  sorry

end theatre_lost_revenue_l167_16792


namespace bolts_per_box_l167_16713

def total_bolts_and_nuts_used : Nat := 113
def bolts_left_over : Nat := 3
def nuts_left_over : Nat := 6
def boxes_of_bolts : Nat := 7
def boxes_of_nuts : Nat := 3
def nuts_per_box : Nat := 15

theorem bolts_per_box :
  let total_bolts_and_nuts := total_bolts_and_nuts_used + bolts_left_over + nuts_left_over
  let total_nuts := boxes_of_nuts * nuts_per_box
  let total_bolts := total_bolts_and_nuts - total_nuts
  let bolts_per_box := total_bolts / boxes_of_bolts
  bolts_per_box = 11 := by
  sorry

end bolts_per_box_l167_16713


namespace capital_of_a_l167_16782

variable (P P' TotalCapital Ca : ℝ)

theorem capital_of_a 
  (h1 : a_income_5_percent = (2/3) * P)
  (h2 : a_income_7_percent = (2/3) * P')
  (h3 : a_income_7_percent - a_income_5_percent = 200)
  (h4 : P = 0.05 * TotalCapital)
  (h5 : P' = 0.07 * TotalCapital)
  : Ca = (2/3) * TotalCapital :=
by
  sorry

end capital_of_a_l167_16782


namespace mixture_correct_l167_16720

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end mixture_correct_l167_16720


namespace sum_of_coefficients_of_factorized_polynomial_l167_16785

theorem sum_of_coefficients_of_factorized_polynomial : 
  ∃ (a b c d e : ℕ), 
    (216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 36) :=
sorry

end sum_of_coefficients_of_factorized_polynomial_l167_16785


namespace quadratic_completion_l167_16795

theorem quadratic_completion (x : ℝ) : 
  (2 * x^2 + 3 * x - 1) = 2 * (x + 3 / 4)^2 - 17 / 8 := 
by 
  -- Proof isn't required, we just state the theorem.
  sorry

end quadratic_completion_l167_16795


namespace valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l167_16794

-- Definition: City is divided by roads, and there are initial and additional currency exchange points

structure City := 
(exchange_points : ℕ)   -- Number of exchange points in the city
(parts : ℕ)             -- Number of parts the city is divided into

-- Given: Initial conditions with one existing exchange point and divided parts
def initialCity : City :=
{ exchange_points := 1, parts := 2 }

-- Function to add exchange points in the city
def addExchangePoints (c : City) (new_points : ℕ) : City :=
{ exchange_points := c.exchange_points + new_points, parts := c.parts }

-- Function to verify that each part has exactly two exchange points
def isValidConfiguration (c : City) : Prop :=
c.exchange_points = 2 * c.parts

-- Theorem: Prove that each configuration of new points is valid
theorem valid_first_configuration : 
  isValidConfiguration (addExchangePoints initialCity 3) := 
sorry

theorem valid_second_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_third_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_fourth_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

end valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l167_16794


namespace correct_transformation_l167_16728

-- Definitions of the equations and their transformations
def optionA := (forall (x : ℝ), ((x / 5) + 1 = x / 2) -> (2 * x + 10 = 5 * x))
def optionB := (forall (x : ℝ), (5 - 2 * (x - 1) = x + 3) -> (5 - 2 * x + 2 = x + 3))
def optionC := (forall (x : ℝ), (5 * x + 3 = 8) -> (5 * x = 8 - 3))
def optionD := (forall (x : ℝ), (3 * x = -7) -> (x = -7 / 3))

-- Theorem stating that option D is the correct transformation
theorem correct_transformation : optionD := 
by 
  sorry

end correct_transformation_l167_16728


namespace first_discount_percentage_l167_16761

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (additional_discount : ℝ) (first_discount : ℝ) : 
  original_price = 600 → final_price = 513 → additional_discount = 0.05 →
  600 * (1 - first_discount / 100) * (1 - 0.05) = 513 →
  first_discount = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end first_discount_percentage_l167_16761


namespace johnPaysPerYear_l167_16760

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l167_16760


namespace quadratic_has_root_in_interval_l167_16727

theorem quadratic_has_root_in_interval (a b c : ℝ) (h : 2 * a + 3 * b + 6 * c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_has_root_in_interval_l167_16727


namespace clock_in_probability_l167_16798

-- Definitions
def start_time := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_start := 495 -- 8:15 in minutes from 00:00 (495 minutes)
def arrival_start := 470 -- 7:50 in minutes from 00:00 (470 minutes)
def arrival_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)

-- Conditions
def arrival_window := arrival_end - arrival_start -- 40 minutes window
def valid_clock_in_window := valid_clock_in_end - valid_clock_in_start -- 15 minutes window

-- Required proof statement
theorem clock_in_probability :
  (valid_clock_in_window : ℚ) / (arrival_window : ℚ) = 3 / 8 :=
by
  sorry

end clock_in_probability_l167_16798


namespace find_m_l167_16788

noncomputable def f (x : ℝ) := 4 * x^2 - 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -14 :=
by
  sorry

end find_m_l167_16788


namespace joan_friends_kittens_l167_16705

theorem joan_friends_kittens (initial_kittens final_kittens friends_kittens : ℕ) 
  (h1 : initial_kittens = 8) 
  (h2 : final_kittens = 10) 
  (h3 : friends_kittens = 2) : 
  final_kittens - initial_kittens = friends_kittens := 
by 
  -- Sorry is used here as a placeholder to indicate where the proof would go.
  sorry

end joan_friends_kittens_l167_16705


namespace no_other_integer_solutions_l167_16726

theorem no_other_integer_solutions :
  (∀ (x : ℤ), (x + 1) ^ 3 + (x + 2) ^ 3 + (x + 3) ^ 3 = (x + 4) ^ 3 → x = 2) := 
by sorry

end no_other_integer_solutions_l167_16726


namespace correct_f_l167_16734

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom functional_equation (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem correct_f (x : ℝ) : f x = x + 1 := sorry

end correct_f_l167_16734


namespace simplify_fraction_l167_16750

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end simplify_fraction_l167_16750


namespace photo_gallery_total_l167_16735

theorem photo_gallery_total (initial_photos: ℕ) (first_day_photos: ℕ) (second_day_photos: ℕ)
  (h_initial: initial_photos = 400) 
  (h_first_day: first_day_photos = initial_photos / 2)
  (h_second_day: second_day_photos = first_day_photos + 120) : 
  initial_photos + first_day_photos + second_day_photos = 920 :=
by
  sorry

end photo_gallery_total_l167_16735


namespace interval_solution_l167_16783

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end interval_solution_l167_16783


namespace taxi_fare_proof_l167_16719

/-- Given equations representing the taxi fare conditions:
1. x + 7y = 16.5 (Person A's fare)
2. x + 11y = 22.5 (Person B's fare)

And using the value of the initial fare and additional charge per kilometer conditions,
prove the initial fare and additional charge and calculate the fare for a 7-kilometer ride. -/
theorem taxi_fare_proof (x y : ℝ) 
  (h1 : x + 7 * y = 16.5)
  (h2 : x + 11 * y = 22.5)
  (h3 : x = 6)
  (h4 : y = 1.5) :
  x = 6 ∧ y = 1.5 ∧ (x + y * (7 - 3)) = 12 :=
by
  sorry

end taxi_fare_proof_l167_16719


namespace factorial_trailing_zeros_l167_16781

theorem factorial_trailing_zeros :
  ∃ (S : Finset ℕ), (∀ m ∈ S, 1 ≤ m ∧ m ≤ 30) ∧ (S.card = 24) ∧ (∀ m ∈ S, 
    ∃ n : ℕ, ∃ k : ℕ,  n ≥ k * 5 ∧ n ≤ (k + 1) * 5 - 1 ∧ 
      m = (n / 5) + (n / 25) + (n / 125) ∧ ((n / 5) % 5 = 0)) :=
sorry

end factorial_trailing_zeros_l167_16781


namespace find_r_l167_16758

theorem find_r (a b m p r : ℝ) (h_roots1 : a * b = 6) 
  (h_eq1 : ∀ x, x^2 - m*x + 6 = 0) 
  (h_eq2 : ∀ x, x^2 - p*x + r = 0) :
  r = 32 / 3 :=
by
  sorry

end find_r_l167_16758


namespace prism_visibility_percentage_l167_16799

theorem prism_visibility_percentage
  (base_edge : ℝ)
  (height : ℝ)
  (cell_side : ℝ)
  (wraps : ℕ)
  (lateral_surface_area : ℝ)
  (transparent_area : ℝ) :
  base_edge = 3.2 →
  height = 5 →
  cell_side = 1 →
  wraps = 2 →
  lateral_surface_area = base_edge * height * 3 →
  transparent_area = 13.8 →
  (transparent_area / lateral_surface_area) * 100 = 28.75 :=
by
  intros h_base_edge h_height h_cell_side h_wraps h_lateral_surface_area h_transparent_area
  sorry

end prism_visibility_percentage_l167_16799


namespace rhombus_area_l167_16765

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) : 
  (d1 * d2) / 2 = 157.5 :=
by
  sorry

end rhombus_area_l167_16765


namespace probability_blue_face_l167_16779

-- Define the total number of faces and the number of blue faces
def total_faces : ℕ := 4 + 2 + 6
def blue_faces : ℕ := 6

-- Calculate the probability of a blue face being up when rolled
theorem probability_blue_face :
  (blue_faces : ℚ) / total_faces = 1 / 2 := by
  sorry

end probability_blue_face_l167_16779


namespace correct_options_l167_16736

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l167_16736


namespace find_k_l167_16743

theorem find_k : 
  ∃ (k : ℚ), 
    (∃ (x y : ℚ), y = 3 * x + 7 ∧ y = -4 * x + 1) ∧ 
    ∃ (x y : ℚ), y = 3 * x + 7 ∧ y = 2 * x + k ∧ k = 43 / 7 := 
sorry

end find_k_l167_16743


namespace find_x_l167_16732

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 :=
by sorry

end find_x_l167_16732


namespace product_of_three_numbers_l167_16716

theorem product_of_three_numbers :
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ a = 2 * (b + c) ∧ b = 6 * c ∧ a * b * c = 12000 / 49 :=
by
  sorry

end product_of_three_numbers_l167_16716


namespace imaginary_part_of_z_l167_16742

open Complex

-- Condition
def equation_z (z : ℂ) : Prop := (z * (1 + I) * I^3) / (1 - I) = 1 - I

-- Problem statement
theorem imaginary_part_of_z (z : ℂ) (h : equation_z z) : z.im = -1 := 
by 
  sorry

end imaginary_part_of_z_l167_16742


namespace total_mangoes_l167_16745

-- Definitions of the entities involved
variables (Alexis Dilan Ashley Ben : ℚ)

-- Conditions given in the problem
def condition1 : Prop := Alexis = 4 * (Dilan + Ashley) ∧ Alexis = 60
def condition2 : Prop := Ashley = 2 * Dilan
def condition3 : Prop := Ben = (1/2) * (Dilan + Ashley)

-- The theorem we want to prove: total mangoes is 82.5
theorem total_mangoes (Alexis Dilan Ashley Ben : ℚ)
  (h1 : condition1 Alexis Dilan Ashley)
  (h2 : condition2 Dilan Ashley)
  (h3 : condition3 Dilan Ashley Ben) :
  Alexis + Dilan + Ashley + Ben = 82.5 :=
sorry

end total_mangoes_l167_16745


namespace work_rate_calculate_l167_16767

theorem work_rate_calculate (A_time B_time C_time total_time: ℕ) 
  (hA : A_time = 4) 
  (hB : B_time = 8)
  (hTotal : total_time = 2) : 
  C_time = 8 :=
by
  sorry

end work_rate_calculate_l167_16767


namespace Edward_money_left_l167_16759

theorem Edward_money_left {initial_amount item_cost sales_tax_rate sales_tax total_cost money_left : ℝ} 
    (h_initial : initial_amount = 18) 
    (h_item : item_cost = 16.35) 
    (h_rate : sales_tax_rate = 0.075) 
    (h_sales_tax : sales_tax = item_cost * sales_tax_rate) 
    (h_sales_tax_rounded : sales_tax = 1.23) 
    (h_total : total_cost = item_cost + sales_tax) 
    (h_money_left : money_left = initial_amount - total_cost) :
    money_left = 0.42 :=
by sorry

end Edward_money_left_l167_16759


namespace speech_competition_score_l167_16739

theorem speech_competition_score :
  let speech_content := 90
  let speech_skills := 80
  let speech_effects := 85
  let content_ratio := 4
  let skills_ratio := 2
  let effects_ratio := 4
  (speech_content * content_ratio + speech_skills * skills_ratio + speech_effects * effects_ratio) / (content_ratio + skills_ratio + effects_ratio) = 86 := by
  sorry

end speech_competition_score_l167_16739


namespace age_weight_not_proportional_l167_16733

theorem age_weight_not_proportional (age weight : ℕ) : ¬(∃ k, ∀ (a w : ℕ), w = k * a → age / weight = k) :=
by
  sorry

end age_weight_not_proportional_l167_16733


namespace min_pieces_pie_l167_16700

theorem min_pieces_pie (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n : ℕ, n = p + q - 1 ∧ 
    (∀ m, m < n → ¬ (∀ k : ℕ, (k < p → n % p = 0) ∧ (k < q → n % q = 0))) :=
sorry

end min_pieces_pie_l167_16700


namespace track_time_is_80_l167_16737

noncomputable def time_to_complete_track
  (a b : ℕ) 
  (meetings : a = 15 ∧ b = 25) : ℕ :=
a + b

theorem track_time_is_80 (a b : ℕ) (meetings : a = 15 ∧ b = 25) : time_to_complete_track a b meetings = 80 := by
  sorry

end track_time_is_80_l167_16737


namespace c_share_correct_l167_16703

def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def total_profit : ℕ := 5000

def total_investment : ℕ := investment_a + investment_b + investment_c
def c_ratio : ℚ := investment_c / total_investment
def c_share : ℚ := total_profit * c_ratio

theorem c_share_correct : c_share = 3000 := by
  sorry

end c_share_correct_l167_16703


namespace percentage_land_mr_william_l167_16766

noncomputable def tax_rate_arable := 0.01
noncomputable def tax_rate_orchard := 0.02
noncomputable def tax_rate_pasture := 0.005

noncomputable def subsidy_arable := 100
noncomputable def subsidy_orchard := 50
noncomputable def subsidy_pasture := 20

noncomputable def total_tax_village := 3840
noncomputable def tax_mr_william := 480

theorem percentage_land_mr_william : 
  (tax_mr_william / total_tax_village : ℝ) * 100 = 12.5 :=
by
  sorry

end percentage_land_mr_william_l167_16766


namespace surface_area_of_sphere_l167_16789

theorem surface_area_of_sphere (l w h : ℝ) (s t : ℝ) :
  l = 3 ∧ w = 2 ∧ h = 1 ∧ (s = (l^2 + w^2 + h^2).sqrt / 2) → t = 4 * Real.pi * s^2 → t = 14 * Real.pi :=
by
  intros
  sorry

end surface_area_of_sphere_l167_16789


namespace find_a_minus_b_l167_16793

def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 5
def h (a b x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 7

theorem find_a_minus_b (a b : ℝ) :
  (∀ x : ℝ, h a b x = -3 * a * x + 5 * a + b) ∧
  (∀ x : ℝ, h_inv (h a b x) = x) ∧
  (∀ x : ℝ, h a b x = x - 7) →
  a - b = 5 :=
by
  sorry

end find_a_minus_b_l167_16793


namespace maple_tree_taller_than_pine_tree_l167_16774

def improper_fraction (a b : ℕ) : ℚ := a + (b : ℚ) / 4
def mixed_number_to_improper_fraction (n m : ℕ) : ℚ := improper_fraction n m

def pine_tree_height : ℚ := mixed_number_to_improper_fraction 12 1
def maple_tree_height : ℚ := mixed_number_to_improper_fraction 18 3

theorem maple_tree_taller_than_pine_tree :
  maple_tree_height - pine_tree_height = 6 + 1 / 2 :=
by sorry

end maple_tree_taller_than_pine_tree_l167_16774


namespace find_h_at_2_l167_16768

noncomputable def h (x : ℝ) : ℝ := x^4 + 2 * x^3 - 12 * x^2 - 14 * x + 24

lemma poly_value_at_minus_2 : h (-2) = -4 := by
  sorry

lemma poly_value_at_1 : h 1 = -1 := by
  sorry

lemma poly_value_at_minus_4 : h (-4) = -16 := by
  sorry

lemma poly_value_at_3 : h 3 = -9 := by
  sorry

theorem find_h_at_2 : h 2 = -20 := by
  sorry

end find_h_at_2_l167_16768


namespace eq1_eq2_eq3_l167_16778

theorem eq1 (x : ℝ) : (x - 2)^2 - 5 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by 
  intro h
  sorry

theorem eq2 (x : ℝ) : x^2 + 4 * x = -3 → x = -1 ∨ x = -3 := 
by 
  intro h
  sorry
  
theorem eq3 (x : ℝ) : 4 * x * (x - 2) = x - 2 → x = 2 ∨ x = 1/4 := 
by 
  intro h
  sorry

end eq1_eq2_eq3_l167_16778


namespace John_meeting_percentage_l167_16755

def hours_to_minutes (h : ℕ) : ℕ := 60 * h

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 60
def third_meeting_duration : ℕ := 2 * first_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

def total_workday_duration : ℕ := hours_to_minutes 12

def percentage_of_meetings (total_meeting_time total_workday_time : ℕ) : ℕ := 
  (total_meeting_time * 100) / total_workday_time

theorem John_meeting_percentage : 
  percentage_of_meetings total_meeting_duration total_workday_duration = 21 :=
by
  sorry

end John_meeting_percentage_l167_16755


namespace probability_at_least_one_female_l167_16764

open Nat

theorem probability_at_least_one_female :
  let males := 2
  let females := 3
  let total_students := males + females
  let select := 2
  let total_ways := choose total_students select
  let ways_at_least_one_female : ℕ := (choose females 1) * (choose males 1) + choose females 2
  (ways_at_least_one_female / total_ways : ℚ) = 9 / 10 := by
  sorry

end probability_at_least_one_female_l167_16764


namespace intersection_complement_eq_l167_16757

def A : Set ℝ := { x | 1 ≤ x ∧ x < 3 }

def B : Set ℝ := { x | x^2 ≥ 4 }

def complementB : Set ℝ := { x | -2 < x ∧ x < 2 }

def intersection (A : Set ℝ) (B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_complement_eq : 
  intersection A complementB = { x | 1 ≤ x ∧ x < 2 } := 
sorry

end intersection_complement_eq_l167_16757


namespace A_wins_one_prob_A_wins_at_least_2_of_3_prob_l167_16702

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Definition of the independent events for A and B
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- The probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * prob_B_incorrect

-- Proof (statement) that A's probability of winning one activity is 1/3
theorem A_wins_one_prob :
  prob_A_wins_one = 1/3 :=
sorry

-- Binomial coefficient for choosing 2 wins out of 3 activities
def binom_coeff_n_2 (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Probability of A winning exactly 2 out of 3 activities
def prob_A_wins_exactly_2_of_3 : ℚ :=
  binom_coeff_n_2 3 2 * prob_A_wins_one^2 * (1 - prob_A_wins_one)

-- Probability of A winning all 3 activities
def prob_A_wins_all_3 : ℚ :=
  prob_A_wins_one^3

-- The probability of A winning at least 2 out of 3 activities
def prob_A_wins_at_least_2_of_3 : ℚ :=
  prob_A_wins_exactly_2_of_3 + prob_A_wins_all_3

-- Proof (statement) that A's probability of winning at least 2 out of 3 activities is 7/27
theorem A_wins_at_least_2_of_3_prob :
  prob_A_wins_at_least_2_of_3 = 7/27 :=
sorry

end A_wins_one_prob_A_wins_at_least_2_of_3_prob_l167_16702


namespace carnival_friends_l167_16704

theorem carnival_friends (F : ℕ) (h1 : 865 % F ≠ 0) (h2 : 873 % F = 0) : F = 3 :=
by
  -- proof is not required
  sorry

end carnival_friends_l167_16704


namespace factory_hours_per_day_l167_16738

def hour_worked_forth_machine := 12
def production_rate_per_hour := 2
def selling_price_per_kg := 50
def total_earnings := 8100

def h := 23

theorem factory_hours_per_day
  (num_machines : ℕ)
  (num_machines := 3)
  (production_first_three : ℕ := num_machines * production_rate_per_hour * h)
  (production_fourth : ℕ := hour_worked_forth_machine * production_rate_per_hour)
  (total_production : ℕ := production_first_three + production_fourth)
  (total_earnings_eq : total_production * selling_price_per_kg = total_earnings) :
  h = 23 := by
  sorry

end factory_hours_per_day_l167_16738


namespace algebra_statements_correct_l167_16773

theorem algebra_statements_correct (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (ac < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (ab > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
sorry

end algebra_statements_correct_l167_16773


namespace evaluate_exponent_l167_16740

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l167_16740


namespace geometric_sum_thm_l167_16715

variable (S : ℕ → ℝ)

theorem geometric_sum_thm (h1 : S n = 48) (h2 : S (2 * n) = 60) : S (3 * n) = 63 :=
sorry

end geometric_sum_thm_l167_16715


namespace sandy_took_200_l167_16748

variable (X : ℝ)

/-- Given that Sandy had $140 left after spending 30% of the money she took for shopping,
we want to prove that Sandy took $200 for shopping. -/
theorem sandy_took_200 (h : 0.70 * X = 140) : X = 200 :=
by
  sorry

end sandy_took_200_l167_16748


namespace evaluate_g_at_2_l167_16721

def g (x : ℝ) : ℝ := x^3 + x^2 - 1

theorem evaluate_g_at_2 : g 2 = 11 := by
  sorry

end evaluate_g_at_2_l167_16721


namespace smallest_n_for_sum_condition_l167_16784

theorem smallest_n_for_sum_condition :
  ∃ n, n ≥ 4 ∧ (∀ S : Finset ℤ, S.card = n → ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a + b - c - d) % 20 = 0) ∧ n = 9 :=
by
  sorry

end smallest_n_for_sum_condition_l167_16784


namespace sum_of_ages_l167_16746

variable (S F : ℕ)

-- Conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F + 6 = 2 * (S + 6)

-- Theorem Statement
theorem sum_of_ages (h1 : condition1 S F) (h2 : condition2 S F) : S + 6 + (F + 6) = 36 := by
  sorry

end sum_of_ages_l167_16746


namespace symmetric_point_l167_16730

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def plane_eq (M : Point3D) : Prop :=
  2 * M.x - 4 * M.y - 4 * M.z - 13 = 0

-- Given Point M
def M : Point3D := { x := 3, y := -3, z := -1 }

-- Symmetric Point M'
def M' : Point3D := { x := 2, y := -1, z := 1 }

theorem symmetric_point (H : plane_eq M) : plane_eq M' ∧ 
  (M'.x = 2 * (3 + 2 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.x) ∧ 
  (M'.y = 2 * (-3 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.y) ∧ 
  (M'.z = 2 * (-1 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.z) :=
sorry

end symmetric_point_l167_16730


namespace green_apples_ordered_l167_16712

-- Definitions based on the conditions
variable (red_apples : Nat := 25)
variable (students : Nat := 10)
variable (extra_apples : Nat := 32)
variable (G : Nat)

-- The mathematical problem to prove
theorem green_apples_ordered :
  red_apples + G - students = extra_apples → G = 17 := by
  sorry

end green_apples_ordered_l167_16712


namespace ratio_of_radii_l167_16706

-- Given conditions
variables {b a c : ℝ}
variables (h1 : π * b^2 - π * c^2 = 2 * π * a^2)
variables (h2 : c = 1.5 * a)

-- Define and prove the ratio
theorem ratio_of_radii (h1: π * b^2 - π * c^2 = 2 * π * a^2) (h2: c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 :=
sorry

end ratio_of_radii_l167_16706


namespace salary_increase_after_five_years_l167_16787

theorem salary_increase_after_five_years :
  ∀ (S : ℝ), (S * (1.15)^5 - S) / S * 100 = 101.14 := by
sorry

end salary_increase_after_five_years_l167_16787


namespace fourth_derivative_l167_16724

noncomputable def f (x : ℝ) : ℝ := (5 * x - 8) * 2^(-x)

theorem fourth_derivative (x : ℝ) : 
  deriv (deriv (deriv (deriv f))) x = 2^(-x) * (Real.log 2)^4 * (5 * x - 9) :=
sorry

end fourth_derivative_l167_16724


namespace min_value_x4_y3_z2_l167_16797

theorem min_value_x4_y3_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1/x + 1/y + 1/z = 9) : 
  x^4 * y^3 * z^2 ≥ 1 / 9^9 :=
by 
  -- Proof goes here
  sorry

end min_value_x4_y3_z2_l167_16797


namespace units_digit_of_product_is_eight_l167_16744

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l167_16744


namespace problem1_extr_vals_l167_16754

-- Definitions from conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

theorem problem1_extr_vals :
  ∃ a b : ℝ, a = g (1/3) ∧ b = g 1 ∧ a = 31/27 ∧ b = 1 :=
by
  sorry

end problem1_extr_vals_l167_16754


namespace scrap_metal_collected_l167_16717

theorem scrap_metal_collected (a b : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9)
  (h₂ : 900 + 10 * a + b - (100 * a + 10 * b + 9) = 216) :
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by
  sorry

end scrap_metal_collected_l167_16717


namespace dan_helmet_craters_l167_16747

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end dan_helmet_craters_l167_16747


namespace polygon_sides_l167_16756

theorem polygon_sides (n : ℕ) (D : ℕ) (hD : D = 77) (hFormula : D = n * (n - 3) / 2) (hVertex : n = n) : n + 1 = 15 :=
by
  sorry

end polygon_sides_l167_16756


namespace arrangements_no_adjacent_dances_arrangements_alternating_order_l167_16709

-- Part (1)
theorem arrangements_no_adjacent_dances (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 43200 := 
by sorry

-- Part (2)
theorem arrangements_alternating_order (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 2880 := 
by sorry

end arrangements_no_adjacent_dances_arrangements_alternating_order_l167_16709


namespace square_area_l167_16775

theorem square_area (x : ℝ) (h1 : x = 60) : x^2 = 1200 :=
by
  sorry

end square_area_l167_16775


namespace planned_pencils_is_49_l167_16780

def pencils_planned (x : ℕ) : ℕ := x
def pencils_bought (x : ℕ) : ℕ := x + 12
def total_pencils_bought (x : ℕ) : ℕ := 61

theorem planned_pencils_is_49 (x : ℕ) :
  pencils_bought (pencils_planned x) = total_pencils_bought x → x = 49 :=
sorry

end planned_pencils_is_49_l167_16780


namespace books_sold_at_overall_loss_l167_16763

-- Defining the conditions and values
def total_cost : ℝ := 540
def C1 : ℝ := 315
def loss_percentage_C1 : ℝ := 0.15
def gain_percentage_C2 : ℝ := 0.19
def C2 : ℝ := total_cost - C1
def loss_C1 := (loss_percentage_C1 * C1)
def SP1 := C1 - loss_C1
def gain_C2 := (gain_percentage_C2 * C2)
def SP2 := C2 + gain_C2
def total_selling_price := SP1 + SP2
def overall_loss := total_cost - total_selling_price

-- Formulating the theorem based on the conditions and required proof
theorem books_sold_at_overall_loss : overall_loss = 4.50 := 
by 
  sorry

end books_sold_at_overall_loss_l167_16763


namespace count_valid_N_l167_16772

theorem count_valid_N : ∃ (N : ℕ), N = 1174 ∧ ∀ (n : ℕ), (1 ≤ n ∧ n < 2000) → ∃ (x : ℝ), x ^ (⌊x⌋ + 1) = n :=
by
  sorry

end count_valid_N_l167_16772


namespace bucket_full_weight_l167_16725

theorem bucket_full_weight (c d : ℝ) (x y : ℝ) (h1 : x + (1 / 4) * y = c) (h2 : x + (3 / 4) * y = d) : 
  x + y = (3 * d - 3 * c) / 2 :=
by
  sorry

end bucket_full_weight_l167_16725


namespace problem_1_problem_2_problem_3_l167_16749

-- The sequence S_n and its given condition
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2 * n

-- Definitions for a_1, a_2, and a_3 based on S_n conditions
theorem problem_1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 14 :=
sorry

-- Definition of sequence b_n and its property of being geometric
def b (n : ℕ) (a : ℕ → ℕ) : ℕ := a n + 2

theorem problem_2 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n ≥ 1, b n a = 2 * b (n - 1) a :=
sorry

-- The sum of the first n terms of the sequence {na_n}, denoted by T_n
def T (n : ℕ) (a : ℕ → ℕ) : ℕ := (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1)

theorem problem_3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n, T n a = (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1) :=
sorry

end problem_1_problem_2_problem_3_l167_16749


namespace victor_decks_l167_16762

theorem victor_decks (V : ℕ) (cost_per_deck total_spent friend_decks : ℕ) 
  (h1 : cost_per_deck = 8)
  (h2 : total_spent = 64)
  (h3 : friend_decks = 2) 
  (h4 : 8 * V + 8 * friend_decks = total_spent) : 
  V = 6 :=
by sorry

end victor_decks_l167_16762


namespace find_smallest_number_l167_16723

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end find_smallest_number_l167_16723


namespace cubic_function_not_monotonically_increasing_l167_16753

theorem cubic_function_not_monotonically_increasing (b : ℝ) :
  ¬(∀ x y : ℝ, x ≤ y → (1/3)*x^3 + b*x^2 + (b+2)*x + 3 ≤ (1/3)*y^3 + b*y^2 + (b+2)*y + 3) ↔ b ∈ (Set.Iio (-1) ∪ Set.Ioi 2) :=
by sorry

end cubic_function_not_monotonically_increasing_l167_16753


namespace candy_bar_cost_l167_16790

theorem candy_bar_cost :
  ∀ (members : ℕ) (avg_candy_bars : ℕ) (total_earnings : ℝ), 
  members = 20 →
  avg_candy_bars = 8 →
  total_earnings = 80 →
  total_earnings / (members * avg_candy_bars) = 0.50 :=
by
  intros members avg_candy_bars total_earnings h_mem h_avg h_earn
  sorry

end candy_bar_cost_l167_16790
