import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_l1133_113370

/-- The area of a parallelogram with base 21 and height 11 is 231 -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 21 → 
    height = 11 → 
    area = base * height → 
    area = 231 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1133_113370


namespace NUMINAMATH_CALUDE_range_of_a_l1133_113350

-- Define the sets A, B, and C
def A : Set ℝ := {x | 0 < 2*x + 4 ∧ 2*x + 4 < 10}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0}

-- Define the union of A and B
def AUB : Set ℝ := {x | x ∈ A ∨ x ∈ B}

-- Define the complement of A ∪ B
def comp_AUB : Set ℝ := {x | x ∉ AUB}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ comp_AUB → x ∈ C a) → -2 < a ∧ a < -4/3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1133_113350


namespace NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l1133_113352

/-- The curve defined by the polar equation θ = π/4 is a straight line -/
theorem polar_equation_pi_over_four_is_line : 
  ∀ (r : ℝ), ∃ (x y : ℝ), x = r * Real.cos (π/4) ∧ y = r * Real.sin (π/4) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l1133_113352


namespace NUMINAMATH_CALUDE_meal_price_before_coupon_l1133_113311

theorem meal_price_before_coupon
  (num_people : ℕ)
  (individual_contribution : ℝ)
  (coupon_value : ℝ)
  (h1 : num_people = 3)
  (h2 : individual_contribution = 21)
  (h3 : coupon_value = 4)
  : ↑num_people * individual_contribution + coupon_value = 67 :=
by sorry

end NUMINAMATH_CALUDE_meal_price_before_coupon_l1133_113311


namespace NUMINAMATH_CALUDE_parking_lot_cars_l1133_113333

theorem parking_lot_cars : ∃ (total : ℕ), 
  (total / 3 : ℚ) + (total / 2 : ℚ) + 86 = total ∧ total = 516 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l1133_113333


namespace NUMINAMATH_CALUDE_distance_between_trees_l1133_113368

/-- Given a yard of length 375 meters with 26 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
    (h1 : yard_length = 375)
    (h2 : num_trees = 26) :
    yard_length / (num_trees - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1133_113368


namespace NUMINAMATH_CALUDE_stream_speed_l1133_113307

/-- Given a canoe that rows upstream at 4 km/hr and downstream at 12 km/hr, 
    the speed of the stream is 4 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 12) : 
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1133_113307


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_17_l1133_113355

theorem modular_inverse_of_3_mod_17 :
  ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 16 ∧ (3 * x) % 17 = 1 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_17_l1133_113355


namespace NUMINAMATH_CALUDE_linear_relation_holds_l1133_113328

def points : List (ℤ × ℤ) := [(0, 200), (1, 160), (2, 120), (3, 80), (4, 40)]

theorem linear_relation_holds (p : ℤ × ℤ) (h : p ∈ points) : 
  (p.2 : ℤ) = 200 - 40 * p.1 := by
  sorry

end NUMINAMATH_CALUDE_linear_relation_holds_l1133_113328


namespace NUMINAMATH_CALUDE_sequence_a_500th_term_l1133_113319

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1007 ∧ 
  a 2 = 1008 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem sequence_a_500th_term (a : ℕ → ℕ) (h : sequence_a a) : a 500 = 1173 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_500th_term_l1133_113319


namespace NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_4_l1133_113312

theorem max_gcd_14n_plus_5_9n_plus_4 :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n > 0 → Nat.gcd (14*n + 5) (9*n + 4) ≤ k ∧
  ∃ (m : ℕ), m > 0 ∧ Nat.gcd (14*m + 5) (9*m + 4) = k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_4_l1133_113312


namespace NUMINAMATH_CALUDE_snake_length_difference_l1133_113377

/-- Given two snakes with a combined length of 70 inches, where one snake is 41 inches long,
    prove that the difference in length between the longer and shorter snake is 12 inches. -/
theorem snake_length_difference (combined_length jake_length : ℕ)
  (h1 : combined_length = 70)
  (h2 : jake_length = 41)
  (h3 : jake_length > combined_length - jake_length) :
  jake_length - (combined_length - jake_length) = 12 := by
  sorry

end NUMINAMATH_CALUDE_snake_length_difference_l1133_113377


namespace NUMINAMATH_CALUDE_set_c_is_well_defined_l1133_113391

-- Define the universe of discourse
def Student : Type := sorry

-- Define the properties
def SeniorHighSchool (s : Student) : Prop := sorry
def EnrolledAtDudeSchool (s : Student) : Prop := sorry
def EnrolledInJanuary2013 (s : Student) : Prop := sorry

-- Define the set C
def SetC : Set Student :=
  {s : Student | SeniorHighSchool s ∧ EnrolledAtDudeSchool s ∧ EnrolledInJanuary2013 s}

-- Define the property of being well-defined
def WellDefined (S : Set Student) : Prop :=
  ∀ s : Student, s ∈ S → (∃ (criterion : Student → Prop), criterion s)

-- Theorem statement
theorem set_c_is_well_defined :
  WellDefined SetC ∧ 
  (∀ S : Set Student, S ≠ SetC → ¬WellDefined S) :=
sorry

end NUMINAMATH_CALUDE_set_c_is_well_defined_l1133_113391


namespace NUMINAMATH_CALUDE_dumbbell_weight_l1133_113345

/-- Given information about exercise bands and total weight, calculate the weight of the dumbbell. -/
theorem dumbbell_weight 
  (num_bands : ℕ) 
  (resistance_per_band : ℕ) 
  (total_weight : ℕ) 
  (h1 : num_bands = 2)
  (h2 : resistance_per_band = 5)
  (h3 : total_weight = 30) :
  total_weight - (num_bands * resistance_per_band) = 20 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_weight_l1133_113345


namespace NUMINAMATH_CALUDE_undefined_rock_ratio_l1133_113341

-- Define the number of rocks Ted and Bill toss
def ted_rocks : ℕ := 10
def bill_rocks : ℕ := 0

-- Define a function to calculate the ratio
def rock_ratio (a b : ℕ) : Option ℚ :=
  if b = 0 then none else some (a / b)

-- Theorem statement
theorem undefined_rock_ratio :
  rock_ratio ted_rocks bill_rocks = none := by
sorry

end NUMINAMATH_CALUDE_undefined_rock_ratio_l1133_113341


namespace NUMINAMATH_CALUDE_prob_same_color_specific_l1133_113331

/-- The probability of drawing 4 marbles of the same color from an urn -/
def prob_same_color (red white blue : ℕ) : ℚ :=
  let total := red + white + blue
  let prob_red := (red.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  let prob_white := (white.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  let prob_blue := (blue.descFactorial 4 : ℚ) / (total.descFactorial 4 : ℚ)
  prob_red + prob_white + prob_blue

/-- Theorem: The probability of drawing 4 marbles of the same color from an urn
    containing 5 red, 6 white, and 7 blue marbles is 55/3060 -/
theorem prob_same_color_specific : prob_same_color 5 6 7 = 55 / 3060 := by
  sorry


end NUMINAMATH_CALUDE_prob_same_color_specific_l1133_113331


namespace NUMINAMATH_CALUDE_classroom_ratio_problem_l1133_113322

theorem classroom_ratio_problem (num_girls : ℕ) (ratio : ℚ) (num_boys : ℕ) : 
  num_girls = 10 → ratio = 1/2 → num_girls = ratio * num_boys → num_boys = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_problem_l1133_113322


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l1133_113372

theorem opposite_of_negative_three :
  -(- 3) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l1133_113372


namespace NUMINAMATH_CALUDE_sweater_markup_percentage_l1133_113367

/-- Given a sweater with wholesale cost and retail price, proves the markup percentage. -/
theorem sweater_markup_percentage 
  (W : ℝ) -- Wholesale cost
  (R : ℝ) -- Normal retail price
  (h1 : W > 0) -- Wholesale cost is positive
  (h2 : R > 0) -- Retail price is positive
  (h3 : 0.4 * R = 1.2 * W) -- Condition for 60% discount and 20% profit
  : (R - W) / W * 100 = 200 :=
by sorry

end NUMINAMATH_CALUDE_sweater_markup_percentage_l1133_113367


namespace NUMINAMATH_CALUDE_nh4i_equilibrium_constant_l1133_113340

/-- Equilibrium constant for a chemical reaction --/
def equilibrium_constant (c_nh3 c_hi : ℝ) : ℝ := c_nh3 * c_hi

/-- Concentration of HI produced from NH₄I decomposition --/
def c_hi_from_nh4i (c_hi c_h2 : ℝ) : ℝ := c_hi + 2 * c_h2

theorem nh4i_equilibrium_constant (c_h2 c_hi : ℝ) 
  (h1 : c_h2 = 1) 
  (h2 : c_hi = 4) :
  equilibrium_constant (c_hi_from_nh4i c_hi c_h2) c_hi = 24 := by
  sorry

end NUMINAMATH_CALUDE_nh4i_equilibrium_constant_l1133_113340


namespace NUMINAMATH_CALUDE_zero_exponent_l1133_113373

theorem zero_exponent (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l1133_113373


namespace NUMINAMATH_CALUDE_system_solution_l1133_113304

theorem system_solution : 
  let solutions := [
    (Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
    (Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2),
    (-Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
    (-Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2),
    (Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
    (Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2),
    (-Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
    (-Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2)
  ]
  ∀ (x y : ℝ), (x^2 + y^2 = 1 ∧ 4*x*y*(2*y^2 - 1) = 1) ↔ (x, y) ∈ solutions := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1133_113304


namespace NUMINAMATH_CALUDE_tan_alpha_equals_two_l1133_113325

theorem tan_alpha_equals_two (α : ℝ) 
  (h : Real.sin (2 * α + Real.pi / 4) - 7 * Real.sin (2 * α + 3 * Real.pi / 4) = 5 * Real.sqrt 2) : 
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_two_l1133_113325


namespace NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l1133_113382

theorem linear_diophantine_equation_solutions
  (a b c : ℤ) (x₀ y₀ : ℤ) (h : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c ↔ ∃ k : ℤ, x = x₀ + k * b ∧ y = y₀ - k * a :=
by sorry

end NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l1133_113382


namespace NUMINAMATH_CALUDE_circle_area_sum_l1133_113329

theorem circle_area_sum : 
  let radius : ℕ → ℝ := λ n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (radius n)^2
  let series_sum : ℝ := ∑' n, area n
  series_sum = 9*π/2 := by
sorry

end NUMINAMATH_CALUDE_circle_area_sum_l1133_113329


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l1133_113371

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ := n.factorial / k.factorial

theorem book_arrangement_proof :
  let total_books : ℕ := 7
  let identical_books : ℕ := 3
  let distinct_books : ℕ := 4
  let books_to_arrange : ℕ := total_books - 1
  number_of_arrangements books_to_arrange identical_books = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l1133_113371


namespace NUMINAMATH_CALUDE_autumn_pencils_l1133_113342

def pencil_count (initial : ℕ) (lost : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - lost - broken + found + bought

theorem autumn_pencils :
  pencil_count 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencils_l1133_113342


namespace NUMINAMATH_CALUDE_paper_stack_height_l1133_113344

theorem paper_stack_height (sheets : ℕ) (height : ℝ) : 
  (400 : ℝ) / 4 = sheets / height → sheets = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_stack_height_l1133_113344


namespace NUMINAMATH_CALUDE_ribbon_length_l1133_113300

theorem ribbon_length (A : ℝ) (π : ℝ) (h1 : A = 616) (h2 : π = 22 / 7) : 
  let r := Real.sqrt (A / π)
  let C := 2 * π * r
  C + 5 = 93 := by sorry

end NUMINAMATH_CALUDE_ribbon_length_l1133_113300


namespace NUMINAMATH_CALUDE_max_almonds_in_mixture_l1133_113308

/-- Represents the composition of the nut mixture -/
structure NutMixture where
  almonds : ℕ
  walnuts : ℕ
  cashews : ℕ
  pistachios : ℕ

/-- Represents the cost per pound of each nut type -/
structure NutCosts where
  almonds : ℚ
  walnuts : ℚ
  cashews : ℚ
  pistachios : ℚ

/-- Calculates the total cost of a given nut mixture -/
def totalCost (mixture : NutMixture) (costs : NutCosts) : ℚ :=
  mixture.almonds * costs.almonds +
  mixture.walnuts * costs.walnuts +
  mixture.cashews * costs.cashews +
  mixture.pistachios * costs.pistachios

/-- Theorem stating the maximum possible pounds of almonds in the mixture -/
theorem max_almonds_in_mixture
  (mixture : NutMixture)
  (costs : NutCosts)
  (budget : ℚ)
  (total_weight : ℕ)
  (h_composition : mixture.almonds = 5 ∧ mixture.walnuts = 2 ∧ mixture.cashews = 3 ∧ mixture.pistachios = 4)
  (h_costs : costs.almonds = 6 ∧ costs.walnuts = 5 ∧ costs.cashews = 8 ∧ costs.pistachios = 10)
  (h_budget : budget = 1500)
  (h_total_weight : total_weight = 800)
  (h_almond_percentage : (mixture.almonds : ℚ) / (mixture.almonds + mixture.walnuts + mixture.cashews + mixture.pistachios) ≥ 30 / 100) :
  (mixture.almonds : ℚ) / (mixture.almonds + mixture.walnuts + mixture.cashews + mixture.pistachios) * total_weight ≤ 240 ∧
  totalCost mixture costs ≤ budget :=
sorry

end NUMINAMATH_CALUDE_max_almonds_in_mixture_l1133_113308


namespace NUMINAMATH_CALUDE_max_cards_proof_l1133_113326

/-- The maximum number of trading cards Jasmine can buy --/
def max_cards : ℕ := 8

/-- Jasmine's initial budget --/
def initial_budget : ℚ := 15

/-- Cost per card --/
def card_cost : ℚ := 1.25

/-- Fixed transaction fee --/
def transaction_fee : ℚ := 2

/-- Minimum amount Jasmine wants to keep --/
def min_remaining : ℚ := 3

theorem max_cards_proof :
  (card_cost * max_cards + transaction_fee ≤ initial_budget - min_remaining) ∧
  (∀ n : ℕ, n > max_cards → card_cost * n + transaction_fee > initial_budget - min_remaining) :=
by sorry

end NUMINAMATH_CALUDE_max_cards_proof_l1133_113326


namespace NUMINAMATH_CALUDE_rain_duration_l1133_113398

/-- Calculates the number of minutes it rained to fill a tank given initial conditions -/
theorem rain_duration (initial_water : ℕ) (evaporated : ℕ) (drained : ℕ) (rain_rate : ℕ) (final_water : ℕ) : 
  initial_water = 6000 →
  evaporated = 2000 →
  drained = 3500 →
  rain_rate = 350 →
  final_water = 1550 →
  (final_water - (initial_water - evaporated - drained)) / (rain_rate / 10) * 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_l1133_113398


namespace NUMINAMATH_CALUDE_total_is_99_l1133_113379

/-- The total number of ducks and ducklings in Mary's observation --/
def total_ducks_and_ducklings : ℕ := by
  -- Define the number of ducks in each group
  let ducks_group1 : ℕ := 2
  let ducks_group2 : ℕ := 6
  let ducks_group3 : ℕ := 9
  
  -- Define the number of ducklings per duck in each group
  let ducklings_per_duck_group1 : ℕ := 5
  let ducklings_per_duck_group2 : ℕ := 3
  let ducklings_per_duck_group3 : ℕ := 6
  
  -- Calculate the total number of ducks and ducklings
  exact ducks_group1 * ducklings_per_duck_group1 +
        ducks_group2 * ducklings_per_duck_group2 +
        ducks_group3 * ducklings_per_duck_group3 +
        ducks_group1 + ducks_group2 + ducks_group3

/-- Theorem stating that the total number of ducks and ducklings is 99 --/
theorem total_is_99 : total_ducks_and_ducklings = 99 := by
  sorry

end NUMINAMATH_CALUDE_total_is_99_l1133_113379


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1133_113321

theorem binomial_coefficient_n_minus_two (n : ℕ) (h : n ≥ 2) :
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1133_113321


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1133_113338

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / (2 * a) + 1 / b = 1) :
  ∀ x y, x > 0 → y > 0 → 1 / (2 * x) + 1 / y = 1 → a + 2 * b ≤ x + 2 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1133_113338


namespace NUMINAMATH_CALUDE_probability_win_first_two_given_earn_3_l1133_113387

def win_probability : ℝ := 0.6

def points_for_win (sets_won : ℕ) : ℕ :=
  if sets_won = 3 then 3 else if sets_won = 2 then 2 else 0

def points_for_loss (sets_lost : ℕ) : ℕ :=
  if sets_lost = 2 then 1 else 0

def prob_win_3_0 : ℝ := win_probability ^ 3

def prob_win_3_1 : ℝ := 3 * (win_probability ^ 2) * (1 - win_probability) * win_probability

def prob_earn_3_points : ℝ := prob_win_3_0 + prob_win_3_1

def prob_win_first_two_and_earn_3 : ℝ := 
  prob_win_3_0 + (win_probability ^ 2) * (1 - win_probability) * win_probability

theorem probability_win_first_two_given_earn_3 :
  prob_win_first_two_and_earn_3 / prob_earn_3_points = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_win_first_two_given_earn_3_l1133_113387


namespace NUMINAMATH_CALUDE_age_ratio_after_two_years_l1133_113376

/-- Given two people a and b, where their initial age ratio is 5:3 and b's age is 6,
    prove that their age ratio after 2 years is 3:2 -/
theorem age_ratio_after_two_years 
  (a b : ℕ) 
  (h1 : a = 5 * b / 3)  -- Initial ratio condition
  (h2 : b = 6)          -- b's initial age
  : (a + 2) / (b + 2) = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_after_two_years_l1133_113376


namespace NUMINAMATH_CALUDE_catriona_fish_count_l1133_113303

/-- The number of fish in Catriona's aquarium -/
def total_fish (goldfish angelfish guppies tetras bettas : ℕ) : ℕ :=
  goldfish + angelfish + guppies + tetras + bettas

/-- Theorem stating the total number of fish in Catriona's aquarium -/
theorem catriona_fish_count :
  ∀ (goldfish angelfish guppies tetras bettas : ℕ),
    goldfish = 8 →
    angelfish = goldfish + 4 →
    guppies = 2 * angelfish →
    tetras = goldfish - 3 →
    bettas = tetras + 5 →
    total_fish goldfish angelfish guppies tetras bettas = 59 := by
  sorry

end NUMINAMATH_CALUDE_catriona_fish_count_l1133_113303


namespace NUMINAMATH_CALUDE_max_successful_teams_16_l1133_113399

/-- Represents a football championship --/
structure Championship :=
  (teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Definition of a successful team --/
def is_successful (c : Championship) (points : ℕ) : Prop :=
  points ≥ (c.teams - 1) * c.points_for_win / 2

/-- The maximum number of successful teams in the championship --/
def max_successful_teams (c : Championship) : ℕ := sorry

/-- The main theorem --/
theorem max_successful_teams_16 :
  ∀ c : Championship,
    c.teams = 16 ∧
    c.points_for_win = 3 ∧
    c.points_for_draw = 1 ∧
    c.points_for_loss = 0 →
    max_successful_teams c = 15 := by sorry

end NUMINAMATH_CALUDE_max_successful_teams_16_l1133_113399


namespace NUMINAMATH_CALUDE_R_24_divisible_by_R_4_Q_only_ones_and_zeros_zeros_in_Q_l1133_113316

/-- R_k represents an integer whose decimal representation consists of k consecutive 1s -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The quotient Q is defined as R_24 divided by R_4 -/
def Q : ℕ := R 24 / R 4

/-- count_zeros counts the number of zeros in the decimal representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem R_24_divisible_by_R_4 : R 24 % R 4 = 0 := sorry

theorem Q_only_ones_and_zeros : ∀ d : ℕ, d ∈ Q.digits 10 → d = 0 ∨ d = 1 := sorry

theorem zeros_in_Q : count_zeros Q = 15 := by sorry

end NUMINAMATH_CALUDE_R_24_divisible_by_R_4_Q_only_ones_and_zeros_zeros_in_Q_l1133_113316


namespace NUMINAMATH_CALUDE_range_of_k_l1133_113359

/-- Given a function f(x) = (x^2 + x + 1) / (kx^2 + kx + 1) with domain R,
    the range of k is [0, 4) -/
theorem range_of_k (f : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = (x^2 + x + 1) / (k*x^2 + k*x + 1)) → 
  (∀ x, k*x^2 + k*x + 1 ≠ 0) →
  (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l1133_113359


namespace NUMINAMATH_CALUDE_probability_sum_nine_l1133_113374

/-- A standard die with six faces -/
def Die : Type := Fin 6

/-- The sample space of rolling a die twice -/
def SampleSpace : Type := Die × Die

/-- The event of getting a sum of 9 -/
def SumNine (outcome : SampleSpace) : Prop :=
  (outcome.1.val + 1) + (outcome.2.val + 1) = 9

/-- The number of favorable outcomes (sum of 9) -/
def FavorableOutcomes : ℕ := 4

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 36

/-- The probability of getting a sum of 9 -/
def ProbabilitySumNine : ℚ := FavorableOutcomes / TotalOutcomes

theorem probability_sum_nine :
  ProbabilitySumNine = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_nine_l1133_113374


namespace NUMINAMATH_CALUDE_no_integer_solution_for_rectangle_l1133_113305

theorem no_integer_solution_for_rectangle : 
  ¬ ∃ (w l : ℕ), w > 0 ∧ l > 0 ∧ w * l = 24 ∧ (w = l ∨ w = 2 * l) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_rectangle_l1133_113305


namespace NUMINAMATH_CALUDE_arithmetic_sum_example_l1133_113369

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℤ) (d : ℤ) (l : ℤ) : ℤ :=
  let n : ℤ := (l - a) / d + 1
  n * (a + l) / 2

/-- Theorem: The sum of the arithmetic sequence with first term -41,
    common difference 3, and last term 7 is -289 -/
theorem arithmetic_sum_example : arithmetic_sum (-41) 3 7 = -289 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_example_l1133_113369


namespace NUMINAMATH_CALUDE_ratio_expression_value_l1133_113356

theorem ratio_expression_value (P Q R : ℚ) (h : P / Q = 3 / 2 ∧ Q / R = 2 / 6) :
  (4 * P + 3 * Q) / (5 * R - 2 * P) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l1133_113356


namespace NUMINAMATH_CALUDE_factorization_proof_l1133_113383

theorem factorization_proof (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1133_113383


namespace NUMINAMATH_CALUDE_plane_seats_count_l1133_113315

theorem plane_seats_count : 
  ∀ (total_seats : ℕ),
  (30 : ℕ) + (total_seats / 5 : ℕ) + (total_seats - (30 + total_seats / 5) : ℕ) = total_seats →
  total_seats = 50 := by
sorry

end NUMINAMATH_CALUDE_plane_seats_count_l1133_113315


namespace NUMINAMATH_CALUDE_sweater_price_calculation_l1133_113324

/-- Given the price of shirts and the price difference between shirts and sweaters,
    calculate the total price of sweaters. -/
theorem sweater_price_calculation (shirt_total : ℕ) (shirt_count : ℕ) (sweater_count : ℕ)
    (price_difference : ℕ) (h1 : shirt_total = 360) (h2 : shirt_count = 20)
    (h3 : sweater_count = 45) (h4 : price_difference = 2) :
    let shirt_avg : ℚ := shirt_total / shirt_count
    let sweater_avg : ℚ := shirt_avg + price_difference
    sweater_avg * sweater_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_calculation_l1133_113324


namespace NUMINAMATH_CALUDE_kitty_cleaning_weeks_l1133_113343

/-- The time Kitty spends on each cleaning task and the total time spent -/
structure CleaningTime where
  pickup : ℕ       -- Time spent picking up toys and straightening
  vacuum : ℕ       -- Time spent vacuuming
  windows : ℕ      -- Time spent cleaning windows
  dusting : ℕ      -- Time spent dusting furniture
  total : ℕ        -- Total time spent cleaning

/-- Calculate the number of weeks Kitty has been cleaning -/
def weeks_cleaning (ct : CleaningTime) : ℕ :=
  ct.total / (ct.pickup + ct.vacuum + ct.windows + ct.dusting)

/-- Theorem stating that Kitty has been cleaning for 4 weeks -/
theorem kitty_cleaning_weeks :
  let ct : CleaningTime := {
    pickup := 5,
    vacuum := 20,
    windows := 15,
    dusting := 10,
    total := 200
  }
  weeks_cleaning ct = 4 := by sorry

end NUMINAMATH_CALUDE_kitty_cleaning_weeks_l1133_113343


namespace NUMINAMATH_CALUDE_investment_profit_distribution_l1133_113390

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  total_contribution : ℝ
  a_duration : ℝ
  b_duration : ℝ
  a_final : ℝ
  b_final : ℝ
  a_contribution : ℝ
  b_contribution : ℝ

/-- Theorem stating that the given contributions satisfy the profit distribution -/
theorem investment_profit_distribution (investment : BusinessInvestment)
  (h1 : investment.total_contribution = 1500)
  (h2 : investment.a_duration = 3)
  (h3 : investment.b_duration = 4)
  (h4 : investment.a_final = 690)
  (h5 : investment.b_final = 1080)
  (h6 : investment.a_contribution = 600)
  (h7 : investment.b_contribution = 900)
  (h8 : investment.a_contribution + investment.b_contribution = investment.total_contribution) :
  (investment.a_final - investment.a_contribution) / (investment.b_final - investment.b_contribution) =
  (investment.a_duration * investment.a_contribution) / (investment.b_duration * investment.b_contribution) :=
by sorry

end NUMINAMATH_CALUDE_investment_profit_distribution_l1133_113390


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l1133_113375

theorem douglas_vote_percentage (total_percentage : ℝ) (county_y_percentage : ℝ) :
  total_percentage = 64 →
  county_y_percentage = 40.00000000000002 →
  let county_x_votes : ℝ := 2
  let county_y_votes : ℝ := 1
  let total_votes : ℝ := county_x_votes + county_y_votes
  let county_x_percentage : ℝ := 
    (total_percentage * total_votes - county_y_percentage * county_y_votes) / county_x_votes
  county_x_percentage = 76 := by
sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l1133_113375


namespace NUMINAMATH_CALUDE_tabs_remaining_l1133_113393

theorem tabs_remaining (initial_tabs : ℕ) : 
  initial_tabs = 400 → 
  (initial_tabs - (initial_tabs / 4) - 
   ((initial_tabs - (initial_tabs / 4)) * 2 / 5) -
   ((initial_tabs - (initial_tabs / 4) - 
     ((initial_tabs - (initial_tabs / 4)) * 2 / 5)) / 2)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_tabs_remaining_l1133_113393


namespace NUMINAMATH_CALUDE_circle_trajectory_l1133_113320

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 81
def F₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the center and radius of F₁
def F₁_center : ℝ × ℝ := (-3, 0)
def F₁_radius : ℝ := 9

-- Define the center and radius of F₂
def F₂_center : ℝ × ℝ := (3, 0)
def F₂_radius : ℝ := 1

-- Define the trajectory of the center of circle P
def trajectory (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1

-- Theorem statement
theorem circle_trajectory :
  ∀ (x y r : ℝ),
  (∃ (x₁ y₁ : ℝ), F₁ x₁ y₁ ∧ (x - x₁)^2 + (y - y₁)^2 = r^2) →
  (∃ (x₂ y₂ : ℝ), F₂ x₂ y₂ ∧ (x - x₂)^2 + (y - y₂)^2 = r^2) →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_l1133_113320


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1133_113389

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 5474827
  let d := 12
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1133_113389


namespace NUMINAMATH_CALUDE_DE_DB_ratio_l1133_113327

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom right_angle_ABC : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
axiom AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4
axiom BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 3
axiom right_angle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
axiom AD_length : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 15
axiom C_D_opposite : ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) *
                     ((B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1)) < 0
axiom D_parallel_AC : (D.2 - A.2) * (E.1 - C.1) = (D.1 - A.1) * (E.2 - C.2)
axiom E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (B.1 - C.1), B.2 + t * (B.2 - C.2))

-- Define the theorem
theorem DE_DB_ratio :
  Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) / Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 57 / 80 :=
sorry

end NUMINAMATH_CALUDE_DE_DB_ratio_l1133_113327


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l1133_113397

theorem complex_magnitude_theorem (z : ℂ) (h : z^2 - 2 * Complex.abs z + 3 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l1133_113397


namespace NUMINAMATH_CALUDE_charging_station_profit_l1133_113366

/-- Represents the total profit function for electric car charging stations -/
def profit_function (a b c : ℝ) (x : ℕ+) : ℝ := a * (x : ℝ)^2 + b * (x : ℝ) + c

theorem charging_station_profit 
  (a b c : ℝ) 
  (h1 : profit_function a b c 3 = 2) 
  (h2 : profit_function a b c 6 = 11) 
  (h3 : ∀ x : ℕ+, profit_function a b c x ≤ 11) :
  (∀ x : ℕ+, profit_function a b c x = -10 * (x : ℝ)^2 + 120 * (x : ℝ) - 250) ∧ 
  (∀ x : ℕ+, (profit_function a b c x) / (x : ℝ) ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_charging_station_profit_l1133_113366


namespace NUMINAMATH_CALUDE_min_value_expression_l1133_113378

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  4 * a^2 + 4 * b^2 + 1 / (a + b)^2 ≥ 2 ∧
  (4 * a^2 + 4 * b^2 + 1 / (a + b)^2 = 2 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1133_113378


namespace NUMINAMATH_CALUDE_pinwheel_shaded_area_l1133_113385

/-- Represents the pinwheel toy configuration -/
structure PinwheelToy where
  square_side : Real
  triangle_leg : Real
  π : Real

/-- Calculates the area of the shaded region in the pinwheel toy -/
def shaded_area (toy : PinwheelToy) : Real :=
  -- The actual calculation would go here
  286

/-- Theorem stating that the shaded area of the specific pinwheel toy is 286 square cm -/
theorem pinwheel_shaded_area :
  ∃ (toy : PinwheelToy),
    toy.square_side = 20 ∧
    toy.triangle_leg = 10 ∧
    toy.π = 3.14 ∧
    shaded_area toy = 286 := by
  sorry


end NUMINAMATH_CALUDE_pinwheel_shaded_area_l1133_113385


namespace NUMINAMATH_CALUDE_angle_x_is_180_l1133_113349

-- Define the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC
  angle_ABC : Real
  angle_ACB : Real
  -- Straight angles
  angle_ADC_straight : Bool
  angle_AEB_straight : Bool

-- Theorem statement
theorem angle_x_is_180 (config : GeometricConfiguration) 
  (h1 : config.angle_ABC = 50)
  (h2 : config.angle_ACB = 70)
  (h3 : config.angle_ADC_straight = true)
  (h4 : config.angle_AEB_straight = true) :
  ∃ x : Real, x = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_x_is_180_l1133_113349


namespace NUMINAMATH_CALUDE_three_valid_k_values_l1133_113306

/-- The sum of k consecutive natural numbers starting from n -/
def consecutiveSum (n k : ℕ) : ℕ := k * (2 * n + k - 1) / 2

/-- Predicate to check if k is a valid solution -/
def isValidK (k : ℕ) : Prop :=
  k > 1 ∧ ∃ n : ℕ, consecutiveSum n k = 2000

theorem three_valid_k_values :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ k, k ∈ s ↔ isValidK k :=
sorry

end NUMINAMATH_CALUDE_three_valid_k_values_l1133_113306


namespace NUMINAMATH_CALUDE_linear_inequality_condition_l1133_113347

theorem linear_inequality_condition (m : ℝ) : 
  (|m - 3| = 1 ∧ m - 4 ≠ 0) ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_linear_inequality_condition_l1133_113347


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1133_113337

/-- The length of the hypotenuse of a right triangle with legs 140 and 336 units is 364 units. -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 140 →
  b = 336 →
  c^2 = a^2 + b^2 →
  c = 364 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1133_113337


namespace NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l1133_113380

theorem sqrt_1_minus_x_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l1133_113380


namespace NUMINAMATH_CALUDE_equal_numbers_l1133_113358

theorem equal_numbers (x : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, (x i)^3 + x (i + 1) = (x (i + 1))^3 + x (i + 2)) : 
  ∀ i j : Fin 100, x i = x j := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_l1133_113358


namespace NUMINAMATH_CALUDE_maple_trees_after_cutting_l1133_113339

/-- The number of maple trees remaining after cutting -/
def remaining_maple_trees (initial : ℝ) (cut : ℝ) : ℝ :=
  initial - cut

/-- Proof that the number of maple trees remaining is 7.0 -/
theorem maple_trees_after_cutting :
  remaining_maple_trees 9.0 2.0 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_after_cutting_l1133_113339


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1133_113392

theorem real_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.re ((1 - i) / ((1 + i)^2)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1133_113392


namespace NUMINAMATH_CALUDE_range_of_a_l1133_113317

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - a + 3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ f a x₀ = 0) →
  (a < -3 ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1133_113317


namespace NUMINAMATH_CALUDE_midpoint_locus_of_constant_area_segment_l1133_113386

/-- Given a line segment PQ with endpoints on the parabola y = x^2 such that the area bounded by PQ
    and the parabola is always 4/3, the locus of the midpoint M of PQ is described by the equation
    y = x^2 + 1 -/
theorem midpoint_locus_of_constant_area_segment (P Q : ℝ × ℝ) :
  (∃ α β : ℝ, α < β ∧ 
    P = (α, α^2) ∧ Q = (β, β^2) ∧ 
    (∫ (x : ℝ) in α..β, ((β - α)⁻¹ * ((β * α^2 - α * β^2) + (β - α) * x) - x^2)) = 4/3) →
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  M.2 = M.1^2 + 1 := by
sorry


end NUMINAMATH_CALUDE_midpoint_locus_of_constant_area_segment_l1133_113386


namespace NUMINAMATH_CALUDE_green_flash_count_l1133_113357

def total_time : ℕ := 671
def green_interval : ℕ := 3
def red_interval : ℕ := 5
def blue_interval : ℕ := 7

def green_flashes : ℕ := total_time / green_interval
def green_red_flashes : ℕ := total_time / (Nat.lcm green_interval red_interval)
def green_blue_flashes : ℕ := total_time / (Nat.lcm green_interval blue_interval)
def all_color_flashes : ℕ := total_time / (Nat.lcm green_interval (Nat.lcm red_interval blue_interval))

theorem green_flash_count : 
  green_flashes - green_red_flashes - green_blue_flashes + all_color_flashes = 154 := by
  sorry

end NUMINAMATH_CALUDE_green_flash_count_l1133_113357


namespace NUMINAMATH_CALUDE_june_design_white_tiles_l1133_113313

/-- The number of white tiles in June's design -/
def white_tiles (total : ℕ) (yellow : ℕ) (purple : ℕ) : ℕ :=
  total - (yellow + (yellow + 1) + purple)

/-- Theorem stating the number of white tiles in June's design -/
theorem june_design_white_tiles :
  white_tiles 20 3 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_june_design_white_tiles_l1133_113313


namespace NUMINAMATH_CALUDE_division_of_powers_l1133_113314

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^6 / (-a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l1133_113314


namespace NUMINAMATH_CALUDE_f_has_max_iff_solution_set_when_a_is_one_l1133_113362

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs (x - 2) + x

-- Theorem 1: f has a maximum value iff a ≤ -1
theorem f_has_max_iff (a : ℝ) : 
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) ↔ a ≤ -1 :=
sorry

-- Theorem 2: Solution set of f(x) < |2x - 3| when a = 1
theorem solution_set_when_a_is_one : 
  {x : ℝ | f 1 x < abs (2 * x - 3)} = {x : ℝ | x > 1/2} :=
sorry

end NUMINAMATH_CALUDE_f_has_max_iff_solution_set_when_a_is_one_l1133_113362


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1133_113309

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1/2 > 0) ↔ 
  (a ≤ -1 ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1133_113309


namespace NUMINAMATH_CALUDE_line_symmetry_l1133_113361

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line2 (x y : ℝ) : Prop := x - y + 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l3 : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), 
    l1 x1 y1 → l2 x2 y2 → 
    ∃ (xm ym : ℝ), l3 xm ym ∧ 
      xm = (x1 + x2) / 2 ∧ 
      ym = (y1 + y2) / 2

-- Theorem statement
theorem line_symmetry : symmetric_wrt line1 line3 line2 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l1133_113361


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1133_113360

/-- Given a square carpet with the described shaded squares, prove the total shaded area is 45 square feet. -/
theorem shaded_area_calculation (S T : ℝ) : 
  S > 0 → T > 0 → (12 : ℝ) / S = 4 → S / T = 2 → 
  S^2 + 16 * T^2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1133_113360


namespace NUMINAMATH_CALUDE_discount_and_total_amount_l1133_113332

/-- Given two positive real numbers P and Q where P > Q, 
    this theorem proves the correct calculation of the percentage discount
    and the total amount paid for 10 items. -/
theorem discount_and_total_amount (P Q : ℝ) (h1 : P > Q) (h2 : Q > 0) :
  let d := 100 * (P - Q) / P
  let total := 10 * Q
  (d = 100 * (P - Q) / P) ∧ (total = 10 * Q) := by
  sorry

#check discount_and_total_amount

end NUMINAMATH_CALUDE_discount_and_total_amount_l1133_113332


namespace NUMINAMATH_CALUDE_final_amount_is_16_l1133_113334

def purchase1 : ℚ := 215/100
def purchase2 : ℚ := 475/100
def purchase3 : ℚ := 1060/100
def discount_rate : ℚ := 1/10

def total_before_discount : ℚ := purchase1 + purchase2 + purchase3
def discounted_total : ℚ := total_before_discount * (1 - discount_rate)

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

theorem final_amount_is_16 :
  round_to_nearest_dollar discounted_total = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_is_16_l1133_113334


namespace NUMINAMATH_CALUDE_line_intersection_l1133_113363

/-- Given a line with slope 3/4 passing through (400, 0), 
    prove that it intersects y = -39 at x = 348 -/
theorem line_intersection (m : ℚ) (x₀ y₀ x₁ y₁ : ℚ) : 
  m = 3/4 → x₀ = 400 → y₀ = 0 → y₁ = -39 →
  (y₁ - y₀) = m * (x₁ - x₀) →
  x₁ = 348 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_l1133_113363


namespace NUMINAMATH_CALUDE_emily_toys_sale_l1133_113310

def sell_toys (initial : ℕ) (percent1 : ℕ) (percent2 : ℕ) : ℕ :=
  let remaining_after_day1 := initial - (initial * percent1 / 100)
  let remaining_after_day2 := remaining_after_day1 - (remaining_after_day1 * percent2 / 100)
  remaining_after_day2

theorem emily_toys_sale :
  sell_toys 35 50 60 = 8 := by
  sorry

end NUMINAMATH_CALUDE_emily_toys_sale_l1133_113310


namespace NUMINAMATH_CALUDE_function_domain_range_implies_m_range_l1133_113365

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- Define the theorem
theorem function_domain_range_implies_m_range 
  (m : ℝ) 
  (domain : Set ℝ)
  (range : Set ℝ)
  (h_domain : domain = Set.Icc 0 m)
  (h_range : range = Set.Icc (-6) (-2))
  (h_func_range : ∀ x ∈ domain, f x ∈ range) :
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_m_range_l1133_113365


namespace NUMINAMATH_CALUDE_sequence_formula_l1133_113323

/-- Given a sequence {aₙ} with sum Sₙ = (3/2)aₙ - 3, prove aₙ = 3 * 2^n -/
theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = (3/2) * a n - 3) :
  ∀ n : ℕ, n ≥ 1 → a n = 3 * 2^n := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1133_113323


namespace NUMINAMATH_CALUDE_original_worker_count_l1133_113351

/-- Given a work that can be completed by some workers in 65 days,
    and adding 10 workers reduces the time to 55 days,
    prove that the original number of workers is 55. -/
theorem original_worker_count (work : ℕ) : ∃ (workers : ℕ), 
  (workers * 65 = (workers + 10) * 55) ∧ 
  (workers = 55) := by
  sorry

end NUMINAMATH_CALUDE_original_worker_count_l1133_113351


namespace NUMINAMATH_CALUDE_knitting_productivity_difference_l1133_113302

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ := k.workTime + k.breakTime

/-- Calculates the number of cycles in a given time period -/
def cyclesInPeriod (k : Knitter) (period : ℕ) : ℕ :=
  period / cycleTime k

/-- Calculates the total working time in a given period -/
def workingTimeInPeriod (k : Knitter) (period : ℕ) : ℕ :=
  k.workTime * cyclesInPeriod k period

/-- Theorem stating the productivity difference between two knitters -/
theorem knitting_productivity_difference
  (girl1 : Knitter)
  (girl2 : Knitter)
  (h1 : girl1.workTime = 5)
  (h2 : girl2.workTime = 7)
  (h3 : girl1.breakTime = 1)
  (h4 : girl2.breakTime = 1)
  (h5 : ∃ (t : ℕ), workingTimeInPeriod girl1 t = workingTimeInPeriod girl2 t) :
  (workingTimeInPeriod girl2 24 : ℚ) / (workingTimeInPeriod girl1 24 : ℚ) = 21 / 20 := by
  sorry

end NUMINAMATH_CALUDE_knitting_productivity_difference_l1133_113302


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1133_113348

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2) = π * (r₁^2 - r₂^2) := by sorry

/-- The area of the ring between two concentric circles with radii 15 and 9 -/
theorem area_of_specific_ring : 
  (π * 15^2 - π * 9^2) = 144 * π := by sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1133_113348


namespace NUMINAMATH_CALUDE_unique_integer_triangle_with_unit_incircle_l1133_113353

/-- A triangle with integer side lengths and an inscribed circle of radius 1 -/
structure IntegerTriangleWithUnitIncircle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  h_incircle : (a + b + c) * 2 = (a + b - c) * (b + c - a) * (c + a - b)

/-- The only triangle with integer side lengths and an inscribed circle of radius 1 has sides 5, 4, and 3 -/
theorem unique_integer_triangle_with_unit_incircle :
  ∀ t : IntegerTriangleWithUnitIncircle, t.a = 5 ∧ t.b = 4 ∧ t.c = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_triangle_with_unit_incircle_l1133_113353


namespace NUMINAMATH_CALUDE_triangular_pyramid_projections_not_equal_l1133_113346

/-- Represents a three-dimensional solid object -/
structure Solid :=
  (shape : Type)

/-- Represents an orthogonal projection (view) of a solid -/
structure Projection :=
  (shape : Type)
  (size : ℝ)

/-- Returns the front projection of a solid -/
def front_view (s : Solid) : Projection :=
  sorry

/-- Returns the top projection of a solid -/
def top_view (s : Solid) : Projection :=
  sorry

/-- Returns the side projection of a solid -/
def side_view (s : Solid) : Projection :=
  sorry

/-- Defines a triangular pyramid -/
def triangular_pyramid : Solid :=
  sorry

/-- Theorem stating that a triangular pyramid cannot have all three
    orthogonal projections of the same shape and size -/
theorem triangular_pyramid_projections_not_equal :
  ∃ (p1 p2 : Projection), 
    (p1 = front_view triangular_pyramid ∧
     p2 = top_view triangular_pyramid ∧
     p1 ≠ p2) ∨
    (p1 = front_view triangular_pyramid ∧
     p2 = side_view triangular_pyramid ∧
     p1 ≠ p2) ∨
    (p1 = top_view triangular_pyramid ∧
     p2 = side_view triangular_pyramid ∧
     p1 ≠ p2) :=
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_projections_not_equal_l1133_113346


namespace NUMINAMATH_CALUDE_missing_number_proof_l1133_113381

theorem missing_number_proof : ∃ x : ℚ, (306 / 34) * x + 270 = 405 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1133_113381


namespace NUMINAMATH_CALUDE_river_depth_ratio_l1133_113364

/-- The ratio of river depths in July to June -/
theorem river_depth_ratio :
  let may_depth : ℝ := 5
  let june_depth : ℝ := may_depth + 10
  let july_depth : ℝ := 45
  july_depth / june_depth = 3 := by sorry

end NUMINAMATH_CALUDE_river_depth_ratio_l1133_113364


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1133_113395

/-- Compound interest calculation --/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (interest : ℝ) (h1 : P > 0) (h2 : t > 0) (h3 : interest > 0) :
  let A := P + interest
  let n := 1
  let r := (((A / P) ^ (1 / (n * t))) - 1) * n
  r = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1133_113395


namespace NUMINAMATH_CALUDE_parallel_transitive_l1133_113335

-- Define a type for vectors
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define a predicate for parallel vectors
def parallel (u v : V) : Prop := ∃ (k : ℝ), v = k • u

-- State the theorem
theorem parallel_transitive {a b c : V} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : parallel a b) (hac : parallel a c) : parallel b c := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l1133_113335


namespace NUMINAMATH_CALUDE_lemonade_sales_increase_l1133_113318

/-- Calculates the percentage increase in lemonade sales --/
def percentage_increase (last_week : ℕ) (total : ℕ) : ℚ :=
  let this_week := total - last_week
  ((this_week - last_week : ℚ) / last_week) * 100

/-- Theorem stating the percentage increase in lemonade sales --/
theorem lemonade_sales_increase :
  let last_week := 20
  let total := 46
  percentage_increase last_week total = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_sales_increase_l1133_113318


namespace NUMINAMATH_CALUDE_function_composition_theorem_l1133_113336

theorem function_composition_theorem (a b : ℝ) :
  (∀ x : ℝ, (3 * ((a * x + b) : ℝ) - 6 : ℝ) = 4 * x + 3) →
  a + b = 13 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_theorem_l1133_113336


namespace NUMINAMATH_CALUDE_reduced_oil_price_l1133_113354

/-- Represents the price and quantity of oil before and after a price reduction --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating that given the conditions, the reduced price of oil is 60 --/
theorem reduced_oil_price
  (oil : OilPriceReduction)
  (price_reduction : oil.reduced_price = 0.75 * oil.original_price)
  (quantity_increase : oil.additional_quantity = 5)
  (cost_equality : oil.original_quantity * oil.original_price = 
                   (oil.original_quantity + oil.additional_quantity) * oil.reduced_price)
  (total_cost : oil.total_cost = 1200)
  : oil.reduced_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_reduced_oil_price_l1133_113354


namespace NUMINAMATH_CALUDE_committee_formations_count_l1133_113330

/-- Represents a department in the division of mathematical sciences -/
inductive Department
| Mathematics
| Statistics
| ComputerScience

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents a professor with their department and gender -/
structure Professor :=
  (department : Department)
  (gender : Gender)

/-- The total number of departments -/
def num_departments : Nat := 3

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors required in the committee -/
def male_professors_in_committee : Nat := 4

/-- The number of female professors required in the committee -/
def female_professors_in_committee : Nat := 4

/-- Calculates the number of ways to form a committee satisfying all conditions -/
def count_committee_formations : Nat :=
  sorry

/-- Theorem stating that the number of possible committee formations is 59049 -/
theorem committee_formations_count :
  count_committee_formations = 59049 := by sorry

end NUMINAMATH_CALUDE_committee_formations_count_l1133_113330


namespace NUMINAMATH_CALUDE_total_value_calculation_l1133_113384

-- Define coin quantities
def us_quarters : ℕ := 25
def us_dimes : ℕ := 15
def us_nickels : ℕ := 12
def us_half_dollars : ℕ := 7
def us_dollar_coins : ℕ := 3
def us_pennies : ℕ := 375
def canadian_quarters : ℕ := 10
def canadian_dimes : ℕ := 5
def canadian_nickels : ℕ := 4

-- Define coin values in their respective currencies
def us_quarter_value : ℚ := 0.25
def us_dime_value : ℚ := 0.10
def us_nickel_value : ℚ := 0.05
def us_half_dollar_value : ℚ := 0.50
def us_dollar_coin_value : ℚ := 1.00
def us_penny_value : ℚ := 0.01
def canadian_quarter_value : ℚ := 0.25
def canadian_dime_value : ℚ := 0.10
def canadian_nickel_value : ℚ := 0.05

-- Define exchange rate
def cad_to_usd_rate : ℚ := 0.80

-- Theorem to prove
theorem total_value_calculation :
  (us_quarters * us_quarter_value +
   us_dimes * us_dime_value +
   us_nickels * us_nickel_value +
   us_half_dollars * us_half_dollar_value +
   us_dollar_coins * us_dollar_coin_value +
   us_pennies * us_penny_value) +
  ((canadian_quarters * canadian_quarter_value +
    canadian_dimes * canadian_dime_value +
    canadian_nickels * canadian_nickel_value) * cad_to_usd_rate) = 21.16 := by
  sorry

end NUMINAMATH_CALUDE_total_value_calculation_l1133_113384


namespace NUMINAMATH_CALUDE_ellipse_foci_range_l1133_113301

/-- Given an ellipse with equation x²/9 + y²/m² = 1 where the foci are on the x-axis,
    prove that the range of m is (-3, 0) ∪ (0, 3) -/
theorem ellipse_foci_range (m : ℝ) : 
  (∃ x y : ℝ, x^2/9 + y^2/m^2 = 1 ∧ (∃ c : ℝ, c > 0 ∧ c < 3 ∧ 
    (∀ x y : ℝ, x^2/9 + y^2/m^2 = 1 → x^2 - y^2 = c^2))) ↔ 
  m ∈ Set.union (Set.Ioo (-3) 0) (Set.Ioo 0 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_range_l1133_113301


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1133_113388

theorem quadratic_roots_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  (x₁ + 1) * (x₂ + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l1133_113388


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1133_113396

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes,
    with each box containing at least one ball, is equal to 4. -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1133_113396


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l1133_113394

-- Define a type for angles
def Angle : Type := ℝ

-- Define a function to create vertical angles
def verticalAngles (α β : Angle) : Prop := α = β

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (α β : Angle) (h : verticalAngles α β) : α = β := by
  sorry

-- Note: We don't define or assume anything about other angle relationships

end NUMINAMATH_CALUDE_vertical_angles_equal_l1133_113394
