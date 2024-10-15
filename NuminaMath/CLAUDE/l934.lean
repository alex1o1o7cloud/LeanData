import Mathlib

namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l934_93464

theorem negative_sixty_four_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l934_93464


namespace NUMINAMATH_CALUDE_pet_shelter_adoption_time_l934_93456

/-- Given a pet shelter scenario, calculate the number of days needed to adopt all puppies -/
theorem pet_shelter_adoption_time (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) : 
  initial_puppies = 9 → additional_puppies = 12 → adoption_rate = 3 →
  (initial_puppies + additional_puppies) / adoption_rate = 7 := by
sorry

end NUMINAMATH_CALUDE_pet_shelter_adoption_time_l934_93456


namespace NUMINAMATH_CALUDE_mika_stickers_l934_93453

/-- The number of stickers Mika has left after a series of transactions -/
def stickers_left (initial : Float) (bought : Float) (birthday : Float) (from_friend : Float)
  (to_sister : Float) (used : Float) (sold : Float) : Float :=
  initial + bought + birthday + from_friend - to_sister - used - sold

/-- Theorem stating that Mika has 6 stickers left after the given transactions -/
theorem mika_stickers :
  stickers_left 20.5 26.25 19.75 7.5 6.3 58.5 3.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l934_93453


namespace NUMINAMATH_CALUDE_negative_angle_quadrant_l934_93490

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

def is_in_second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 - 270 < α ∧ α < n * 360 - 180

theorem negative_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_quadrant (-α) := by
  sorry

end NUMINAMATH_CALUDE_negative_angle_quadrant_l934_93490


namespace NUMINAMATH_CALUDE_min_value_and_existence_l934_93438

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + t*x + 1) / Real.log 2

theorem min_value_and_existence (t : ℝ) (h : t > -2) :
  (∀ x ∈ Set.Icc 0 2, f t x ≥ (if -2 < t ∧ t < 0 then Real.log (1 - t^2/4) / Real.log 2 else 0)) ∧
  (∃ a b : ℝ, a ≠ b ∧ a ∈ Set.Ioo 0 2 ∧ b ∈ Set.Ioo 0 2 ∧ 
   f t a = Real.log a / Real.log 2 ∧ f t b = Real.log b / Real.log 2 ↔ 
   t > -3/2 ∧ t < -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_existence_l934_93438


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l934_93401

theorem imaginary_part_of_reciprocal (z : ℂ) : z = 1 - 3*I → (1/z).im = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l934_93401


namespace NUMINAMATH_CALUDE_ladder_rope_difference_l934_93457

/-- Proves that the ladder is 10 feet longer than the rope given the climbing scenario -/
theorem ladder_rope_difference (
  num_flights : ℕ) 
  (flight_height : ℝ) 
  (total_height : ℝ) 
  (h1 : num_flights = 3)
  (h2 : flight_height = 10)
  (h3 : total_height = 70) : 
  let stairs_height := num_flights * flight_height
  let rope_height := stairs_height / 2
  let ladder_height := total_height - (stairs_height + rope_height)
  ladder_height - rope_height = 10 := by
sorry

end NUMINAMATH_CALUDE_ladder_rope_difference_l934_93457


namespace NUMINAMATH_CALUDE_count_proposition_permutations_l934_93415

/-- The number of distinct permutations of letters in "PROPOSITION" -/
def proposition_permutations : ℕ :=
  Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating the number of distinct permutations of "PROPOSITION" -/
theorem count_proposition_permutations :
  proposition_permutations = 453600 := by
  sorry

end NUMINAMATH_CALUDE_count_proposition_permutations_l934_93415


namespace NUMINAMATH_CALUDE_jack_apples_to_father_l934_93446

/-- The number of apples Jack bought -/
def total_apples : ℕ := 55

/-- The number of Jack's friends -/
def num_friends : ℕ := 4

/-- The number of apples each person (Jack and his friends) gets -/
def apples_per_person : ℕ := 9

/-- The number of apples Jack wants to give to his father -/
def apples_to_father : ℕ := total_apples - (num_friends + 1) * apples_per_person

theorem jack_apples_to_father :
  apples_to_father = 10 := by sorry

end NUMINAMATH_CALUDE_jack_apples_to_father_l934_93446


namespace NUMINAMATH_CALUDE_profit_calculation_l934_93447

-- Define the number of items bought and the price paid
def items_bought : ℕ := 60
def price_paid : ℕ := 46

-- Define the discount rate
def discount_rate : ℚ := 1 / 100

-- Define a function to calculate the profit percent
def profit_percent (items : ℕ) (price : ℕ) (discount : ℚ) : ℚ :=
  let cost_per_item : ℚ := price / items
  let selling_price : ℚ := 1 - discount
  let profit_per_item : ℚ := selling_price - cost_per_item
  (profit_per_item / cost_per_item) * 100

-- State the theorem
theorem profit_calculation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (profit_percent items_bought price_paid discount_rate - 2911/100) < ε :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l934_93447


namespace NUMINAMATH_CALUDE_sin_cos_identity_l934_93429

theorem sin_cos_identity (x y : ℝ) :
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l934_93429


namespace NUMINAMATH_CALUDE_x_minus_y_values_l934_93482

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 5) (h3 : x * y < 0) :
  x - y = -7 ∨ x - y = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l934_93482


namespace NUMINAMATH_CALUDE_chi_squared_relationship_confidence_l934_93494

-- Define the chi-squared statistic
def chi_squared : ℝ := 4.073

-- Define the critical values and their corresponding p-values
def critical_value_1 : ℝ := 3.841
def p_value_1 : ℝ := 0.05

def critical_value_2 : ℝ := 5.024
def p_value_2 : ℝ := 0.025

-- Define the confidence level we want to prove
def target_confidence : ℝ := 0.95

-- Theorem statement
theorem chi_squared_relationship_confidence :
  chi_squared > critical_value_1 ∧ chi_squared < critical_value_2 →
  ∃ (confidence : ℝ), confidence ≥ target_confidence ∧
    confidence ≤ 1 - p_value_1 ∧
    confidence > 1 - p_value_2 :=
by sorry

end NUMINAMATH_CALUDE_chi_squared_relationship_confidence_l934_93494


namespace NUMINAMATH_CALUDE_keith_cards_l934_93434

theorem keith_cards (x : ℕ) : 
  (x + 8) / 2 = 46 → x = 84 := by
  sorry

end NUMINAMATH_CALUDE_keith_cards_l934_93434


namespace NUMINAMATH_CALUDE_square_root_of_1024_l934_93402

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l934_93402


namespace NUMINAMATH_CALUDE_trig_identities_l934_93499

theorem trig_identities (α : Real) (h : Real.sin α = 2 * Real.cos α) : 
  ((2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4) ∧
  (Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l934_93499


namespace NUMINAMATH_CALUDE_tree_height_l934_93491

theorem tree_height (tree_shadow : ℝ) (flagpole_shadow : ℝ) (flagpole_height : ℝ)
  (h1 : tree_shadow = 8)
  (h2 : flagpole_shadow = 100)
  (h3 : flagpole_height = 150) :
  (tree_shadow * flagpole_height) / flagpole_shadow = 12 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_l934_93491


namespace NUMINAMATH_CALUDE_red_candies_count_l934_93413

theorem red_candies_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end NUMINAMATH_CALUDE_red_candies_count_l934_93413


namespace NUMINAMATH_CALUDE_kenny_friday_jacks_l934_93451

/-- The number of jumping jacks Kenny did last week -/
def last_week_total : ℕ := 324

/-- The number of jumping jacks Kenny did on Sunday -/
def sunday_jacks : ℕ := 34

/-- The number of jumping jacks Kenny did on Monday -/
def monday_jacks : ℕ := 20

/-- The number of jumping jacks Kenny did on Tuesday -/
def tuesday_jacks : ℕ := 0

/-- The number of jumping jacks Kenny did on Wednesday -/
def wednesday_jacks : ℕ := 123

/-- The number of jumping jacks Kenny did on Thursday -/
def thursday_jacks : ℕ := 64

/-- The number of jumping jacks Kenny did on some unspecified day -/
def some_day_jacks : ℕ := 61

/-- The number of jumping jacks Kenny did on Friday -/
def friday_jacks : ℕ := 23

/-- Theorem stating that Kenny did 23 jumping jacks on Friday -/
theorem kenny_friday_jacks : 
  friday_jacks = 23 ∧ 
  friday_jacks + sunday_jacks + monday_jacks + tuesday_jacks + wednesday_jacks + thursday_jacks + some_day_jacks > last_week_total :=
by sorry

end NUMINAMATH_CALUDE_kenny_friday_jacks_l934_93451


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_l934_93496

def problem (A B C D : ℕ) : Prop :=
  A ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  B ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  C ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  D ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A : ℚ) / B - (C : ℚ) / D = 1

theorem sum_of_A_and_C (A B C D : ℕ) (h : problem A B C D) : A + C = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_l934_93496


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_four_l934_93420

theorem no_solution_iff_m_eq_neg_four :
  ∀ m : ℝ, (∀ x : ℝ, (x ≠ 2 ∧ x ≠ -2) → 
    ((x - 2) / (x + 2) - m * x / (x^2 - 4) ≠ 1)) ↔ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_four_l934_93420


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l934_93443

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the athlete's heart beats 28800 times during the race --/
theorem athlete_heartbeats :
  total_heartbeats 160 30 6 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l934_93443


namespace NUMINAMATH_CALUDE_f_odd_f_max_on_interval_l934_93461

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The function f satisfies the additive property -/
axiom f_additive (x y : ℝ) : f (x + y) = f x + f y

/-- The function f is negative for positive inputs -/
axiom f_neg_for_pos (x : ℝ) (h : x > 0) : f x < 0

/-- The value of f at 1 is -2 -/
axiom f_one : f 1 = -2

/-- f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

/-- The maximum value of f on [-3, 3] is 6 -/
theorem f_max_on_interval : ∃ x ∈ Set.Icc (-3) 3, ∀ y ∈ Set.Icc (-3) 3, f y ≤ f x ∧ f x = 6 := by sorry

end NUMINAMATH_CALUDE_f_odd_f_max_on_interval_l934_93461


namespace NUMINAMATH_CALUDE_purely_imaginary_modulus_l934_93404

theorem purely_imaginary_modulus (a : ℝ) :
  let z : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)
  (∃ b : ℝ, z = b * Complex.I) → Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_modulus_l934_93404


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l934_93472

theorem absolute_value_inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l934_93472


namespace NUMINAMATH_CALUDE_circumcircle_theorem_tangent_circles_theorem_l934_93474

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (0, 0)

-- Define the circumcircle equation
def circumcircle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 3*y = 0

-- Define the circles with center on y-axis and radius 5
def circle_eq_1 (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 25

def circle_eq_2 (x y : ℝ) : Prop :=
  x^2 + (y - 11)^2 = 25

-- Theorem for the circumcircle
theorem circumcircle_theorem :
  circumcircle_eq A.1 A.2 ∧
  circumcircle_eq B.1 B.2 ∧
  circumcircle_eq C.1 C.2 :=
sorry

-- Theorem for the circles tangent to y = 6
theorem tangent_circles_theorem :
  (∃ x y : ℝ, circle_eq_1 x y ∧ y = 6) ∧
  (∃ x y : ℝ, circle_eq_2 x y ∧ y = 6) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_theorem_tangent_circles_theorem_l934_93474


namespace NUMINAMATH_CALUDE_number_sequence_count_l934_93424

/-- The total number of numbers in the sequence -/
def n : ℕ := 8

/-- The average of all numbers -/
def total_average : ℚ := 25

/-- The average of the first two numbers -/
def first_two_average : ℚ := 20

/-- The average of the next three numbers -/
def next_three_average : ℚ := 26

/-- The sixth number in the sequence -/
def sixth_number : ℚ := 14

/-- The last (eighth) number in the sequence -/
def last_number : ℚ := 30

theorem number_sequence_count :
  (2 * first_two_average + 3 * next_three_average + sixth_number + 
   (sixth_number + 4) + (sixth_number + 6) + last_number) / n = total_average := by
  sorry

#check number_sequence_count

end NUMINAMATH_CALUDE_number_sequence_count_l934_93424


namespace NUMINAMATH_CALUDE_second_company_base_rate_l934_93428

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 7

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second telephone company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes for which the bills are equal -/
def equal_minutes : ℝ := 100

/-- The base rate of the second telephone company in dollars -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + united_per_minute * equal_minutes =
  second_base_rate + second_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l934_93428


namespace NUMINAMATH_CALUDE_square_side_length_range_l934_93431

theorem square_side_length_range (a : ℝ) : 
  (a > 0) → (a^2 = 37) → (6 < a ∧ a < 7) := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l934_93431


namespace NUMINAMATH_CALUDE_functional_equation_solution_l934_93442

/-- The functional equation solution for f(x+y) f(x-y) = (f(x))^2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x y : ℝ, f (x + y) * f (x - y) = (f x)^2) :
  ∃ a c : ℝ, ∀ x : ℝ, f x = a * (c^x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l934_93442


namespace NUMINAMATH_CALUDE_intersection_chord_length_l934_93405

-- Define the polar equations
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - 2 * Real.pi / 3) = -Real.sqrt 3

def circle_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the Cartesian equations
def line_cartesian (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

def circle_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ,
  (∃ θ_A ρ_A, line_polar ρ_A θ_A ∧ circle_polar ρ_A θ_A ∧ A = (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A)) →
  (∃ θ_B ρ_B, line_polar ρ_B θ_B ∧ circle_polar ρ_B θ_B ∧ B = (ρ_B * Real.cos θ_B, ρ_B * Real.sin θ_B)) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l934_93405


namespace NUMINAMATH_CALUDE_deduction_from_second_number_l934_93462

theorem deduction_from_second_number 
  (n : ℕ) 
  (avg_initial : ℚ)
  (avg_final : ℚ)
  (deduct_first : ℚ)
  (deduct_third : ℚ)
  (deduct_fourth_to_ninth : List ℚ)
  (h1 : n = 10)
  (h2 : avg_initial = 16)
  (h3 : avg_final = 11.5)
  (h4 : deduct_first = 9)
  (h5 : deduct_third = 7)
  (h6 : deduct_fourth_to_ninth = [6, 5, 4, 3, 2, 1]) :
  ∃ (deduct_second : ℚ), deduct_second = 8 ∧
    (n * avg_final = n * avg_initial - 
      (deduct_first + deduct_second + deduct_third + 
       deduct_fourth_to_ninth.sum)) :=
by sorry

end NUMINAMATH_CALUDE_deduction_from_second_number_l934_93462


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_nut_to_dried_fruit_ratio_dried_fruit_percentage_l934_93444

/-- Represents the trail mix problem with raisins, nuts, and dried fruit. -/
structure TrailMix where
  x : ℝ
  raisin_cost : ℝ
  raisin_weight : ℝ := 3 * x
  nut_weight : ℝ := 4 * x
  dried_fruit_weight : ℝ := 5 * x
  nut_cost : ℝ := 3 * raisin_cost
  dried_fruit_cost : ℝ := 1.5 * raisin_cost

/-- The total cost of raisins is 1/7.5 of the total cost of the mixture. -/
theorem raisin_cost_fraction (mix : TrailMix) :
  (mix.raisin_weight * mix.raisin_cost) / 
  (mix.raisin_weight * mix.raisin_cost + mix.nut_weight * mix.nut_cost + mix.dried_fruit_weight * mix.dried_fruit_cost) = 1 / 7.5 := by
  sorry

/-- The ratio of the cost of nuts to the cost of dried fruit is 2:1. -/
theorem nut_to_dried_fruit_ratio (mix : TrailMix) :
  mix.nut_cost / mix.dried_fruit_cost = 2 := by
  sorry

/-- The total cost of dried fruit is 50% of the total cost of raisins and nuts combined. -/
theorem dried_fruit_percentage (mix : TrailMix) :
  (mix.dried_fruit_weight * mix.dried_fruit_cost) / 
  (mix.raisin_weight * mix.raisin_cost + mix.nut_weight * mix.nut_cost) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_nut_to_dried_fruit_ratio_dried_fruit_percentage_l934_93444


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l934_93485

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 6 = 1 →
  a 7 = 0.25 →
  a 3 + a 4 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l934_93485


namespace NUMINAMATH_CALUDE_same_even_number_probability_l934_93495

-- Define a standard die
def standardDie : ℕ := 6

-- Define the number of even faces on a standard die
def evenFaces : ℕ := 3

-- Define the number of dice rolled
def numDice : ℕ := 4

-- Theorem statement
theorem same_even_number_probability :
  let p : ℚ := (evenFaces / standardDie) * (1 / standardDie)^(numDice - 1)
  p = 1 / 432 := by
  sorry


end NUMINAMATH_CALUDE_same_even_number_probability_l934_93495


namespace NUMINAMATH_CALUDE_complement_of_A_l934_93455

def A : Set ℝ := {x : ℝ | x ≥ 1}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l934_93455


namespace NUMINAMATH_CALUDE_particle_speed_at_2_l934_93409

/-- The position of a particle at time t -/
def particle_position (t : ℝ) : ℝ × ℝ :=
  (t^2 + 2*t + 7, 3*t^2 + 4*t - 13)

/-- The speed of the particle at time t -/
noncomputable def particle_speed (t : ℝ) : ℝ :=
  let pos_t := particle_position t
  let pos_next := particle_position (t + 1)
  let dx := pos_next.1 - pos_t.1
  let dy := pos_next.2 - pos_t.2
  Real.sqrt (dx^2 + dy^2)

/-- Theorem: The speed of the particle at t = 2 is √410 -/
theorem particle_speed_at_2 :
  particle_speed 2 = Real.sqrt 410 := by
  sorry

end NUMINAMATH_CALUDE_particle_speed_at_2_l934_93409


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l934_93488

/-- The perimeter of a semicircle with radius 9 is approximately 46.26 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 9
  let π_approx : ℝ := 3.14
  let semicircle_perimeter := r * π_approx + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 46.26) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l934_93488


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l934_93469

/-- A monic quadratic polynomial with real coefficients -/
def MonicQuadratic (a b : ℝ) : ℂ → ℂ := fun x ↦ x^2 + a*x + b

/-- The given complex number that is a root of the polynomial -/
def givenRoot : ℂ := 2 - 3*Complex.I

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), (MonicQuadratic a b givenRoot = 0) ∧ (a = -4 ∧ b = 13) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l934_93469


namespace NUMINAMATH_CALUDE_initial_number_equation_l934_93481

theorem initial_number_equation : ∃ x : ℝ, 3 * (2 * x + 13) = 93 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_equation_l934_93481


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l934_93432

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (prevFine : ℚ) : ℚ :=
  min (prevFine + 0.3) (prevFine * 2)

/-- Calculates the total fine for a given number of days -/
def totalFine : ℕ → ℚ
  | 0 => 0
  | 1 => 0.05
  | n + 1 => nextDayFine (totalFine n)

theorem fine_on_fifth_day :
  totalFine 5 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l934_93432


namespace NUMINAMATH_CALUDE_tulip_fraction_l934_93478

-- Define the total number of flowers (arbitrary positive real number)
variable (total : ℝ) (total_pos : 0 < total)

-- Define the number of each type of flower
variable (pink_roses : ℝ) (red_roses : ℝ) (pink_tulips : ℝ) (red_tulips : ℝ)

-- All flowers are either roses or tulips, and either pink or red
axiom flower_sum : pink_roses + red_roses + pink_tulips + red_tulips = total

-- 1/4 of pink flowers are roses
axiom pink_rose_ratio : pink_roses = (1/4) * (pink_roses + pink_tulips)

-- 1/3 of red flowers are tulips
axiom red_tulip_ratio : red_tulips = (1/3) * (red_roses + red_tulips)

-- 7/10 of all flowers are red
axiom red_flower_ratio : red_roses + red_tulips = (7/10) * total

-- Theorem: The fraction of flowers that are tulips is 11/24
theorem tulip_fraction :
  (pink_tulips + red_tulips) / total = 11/24 := by sorry

end NUMINAMATH_CALUDE_tulip_fraction_l934_93478


namespace NUMINAMATH_CALUDE_bug_population_zero_l934_93483

/-- Represents the bug population and predator actions in Bill's garden --/
structure GardenState where
  initial_bugs : ℕ
  spiders : ℕ
  ladybugs : ℕ
  mantises : ℕ
  spider_eat_rate : ℕ
  ladybug_eat_rate : ℕ
  mantis_eat_rate : ℕ
  first_spray_rate : ℚ
  second_spray_rate : ℚ

/-- Calculates the final bug population after all actions --/
def final_bug_population (state : GardenState) : ℕ :=
  sorry

/-- Theorem stating that the final bug population is 0 --/
theorem bug_population_zero (state : GardenState) 
  (h1 : state.initial_bugs = 400)
  (h2 : state.spiders = 12)
  (h3 : state.ladybugs = 5)
  (h4 : state.mantises = 8)
  (h5 : state.spider_eat_rate = 7)
  (h6 : state.ladybug_eat_rate = 6)
  (h7 : state.mantis_eat_rate = 4)
  (h8 : state.first_spray_rate = 4/5)
  (h9 : state.second_spray_rate = 7/10) :
  final_bug_population state = 0 :=
sorry

end NUMINAMATH_CALUDE_bug_population_zero_l934_93483


namespace NUMINAMATH_CALUDE_studentG_score_l934_93436

-- Define the answer types
inductive Answer
| Correct
| Incorrect
| Unanswered

-- Define the scoring function
def score (a : Answer) : Nat :=
  match a with
  | Answer.Correct => 2
  | Answer.Incorrect => 0
  | Answer.Unanswered => 1

-- Define Student G's answer pattern
def studentG_answers : List Answer :=
  [Answer.Correct, Answer.Incorrect, Answer.Correct, Answer.Correct, Answer.Incorrect, Answer.Correct]

-- Theorem: Student G's total score is 8 points
theorem studentG_score :
  (studentG_answers.map score).sum = 8 := by
  sorry

end NUMINAMATH_CALUDE_studentG_score_l934_93436


namespace NUMINAMATH_CALUDE_percentage_equality_l934_93403

theorem percentage_equality (x : ℝ) : (60 / 100 * 500 = 50 / 100 * x) → x = 600 :=
by sorry

end NUMINAMATH_CALUDE_percentage_equality_l934_93403


namespace NUMINAMATH_CALUDE_max_value_a_l934_93418

theorem max_value_a (a : ℝ) : 
  (∀ x > 0, x * Real.exp x - a * (x + 1) ≥ Real.log x) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l934_93418


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l934_93477

theorem zero_neither_positive_nor_negative : ¬(0 > 0 ∨ 0 < 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l934_93477


namespace NUMINAMATH_CALUDE_tom_age_l934_93412

theorem tom_age (carla dave emily tom : ℕ) : 
  tom = 2 * carla - 1 →
  dave = carla + 3 →
  emily = carla / 2 →
  carla + dave + emily + tom = 48 →
  tom = 19 := by
  sorry

end NUMINAMATH_CALUDE_tom_age_l934_93412


namespace NUMINAMATH_CALUDE_alex_income_l934_93417

/-- Represents the tax structure and Alex's tax payment --/
structure TaxSystem where
  q : ℝ  -- Base tax rate as a percentage
  income : ℝ  -- Alex's annual income
  total_tax : ℝ  -- Total tax paid by Alex

/-- The tax system satisfies the given conditions --/
def valid_tax_system (ts : TaxSystem) : Prop :=
  ts.total_tax = 
    (if ts.income ≤ 50000 then
      (ts.q / 100) * ts.income
    else
      (ts.q / 100) * 50000 + ((ts.q + 3) / 100) * (ts.income - 50000))
  ∧ ts.total_tax = ((ts.q + 0.5) / 100) * ts.income

/-- Theorem stating that Alex's income is $60000 --/
theorem alex_income (ts : TaxSystem) (h : valid_tax_system ts) : ts.income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_alex_income_l934_93417


namespace NUMINAMATH_CALUDE_homework_group_existence_l934_93425

theorem homework_group_existence :
  ∀ (S : Finset ℕ) (f : Finset ℕ → Finset ℕ → Prop),
    S.card = 21 →
    (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
      (f {a, b, c} {0} ∨ f {a, b, c} {1}) ∧
      ¬(f {a, b, c} {0} ∧ f {a, b, c} {1})) →
    ∃ T : Finset ℕ, T ⊆ S ∧ T.card = 4 ∧
      (∀ a b c : ℕ, a ∈ T → b ∈ T → c ∈ T → a ≠ b → b ≠ c → a ≠ c →
        (f {a, b, c} {0} ∨ f {a, b, c} {1})) :=
by sorry


end NUMINAMATH_CALUDE_homework_group_existence_l934_93425


namespace NUMINAMATH_CALUDE_correct_financial_equation_l934_93445

/-- Represents Howard's financial transactions -/
def howards_finances (W D X Y : ℝ) : Prop :=
  let initial_money : ℝ := 26
  let final_money : ℝ := 52
  let window_washing_income : ℝ := W
  let dog_walking_income : ℝ := D
  let window_supplies_expense : ℝ := X
  let dog_treats_expense : ℝ := Y
  initial_money + window_washing_income + dog_walking_income - window_supplies_expense - dog_treats_expense = final_money

theorem correct_financial_equation (W D X Y : ℝ) :
  howards_finances W D X Y ↔ 26 + W + D - X - Y = 52 := by sorry

end NUMINAMATH_CALUDE_correct_financial_equation_l934_93445


namespace NUMINAMATH_CALUDE_chimney_bricks_l934_93466

/-- Represents the time (in hours) it takes Brenda to build the chimney alone -/
def brenda_time : ℝ := 8

/-- Represents the time (in hours) it takes Bob to build the chimney alone -/
def bob_time : ℝ := 12

/-- Represents the decrease in productivity (in bricks per hour) when working together -/
def productivity_decrease : ℝ := 15

/-- Represents the time (in hours) it takes Brenda and Bob to build the chimney together -/
def joint_time : ℝ := 6

/-- Theorem stating that the number of bricks in the chimney is 360 -/
theorem chimney_bricks : ℝ := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_l934_93466


namespace NUMINAMATH_CALUDE_right_focus_coordinates_l934_93459

/-- The coordinates of the right focus of a hyperbola with equation x^2 - 2y^2 = 1 -/
theorem right_focus_coordinates :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - 2*y^2 = 1}
  ∃ (f : ℝ × ℝ), f ∈ hyperbola ∧ f.1 > 0 ∧ f.2 = 0 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ hyperbola ∧ p.1 > 0 ∧ p.2 = 0 → p = f ∧
    f = (Real.sqrt (3/2), 0) :=
by sorry

end NUMINAMATH_CALUDE_right_focus_coordinates_l934_93459


namespace NUMINAMATH_CALUDE_choose_books_different_languages_l934_93421

theorem choose_books_different_languages (chinese english japanese : ℕ) :
  chinese = 5 → english = 4 → japanese = 3 →
  chinese + english + japanese = 12 :=
by sorry

end NUMINAMATH_CALUDE_choose_books_different_languages_l934_93421


namespace NUMINAMATH_CALUDE_markup_calculation_l934_93471

theorem markup_calculation (purchase_price overhead_percentage net_profit : ℝ) 
  (h1 : purchase_price = 48)
  (h2 : overhead_percentage = 0.35)
  (h3 : net_profit = 18) :
  purchase_price + purchase_price * overhead_percentage + net_profit - purchase_price = 34.80 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l934_93471


namespace NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l934_93468

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ x y : ℤ, y^k = x^2 + x ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l934_93468


namespace NUMINAMATH_CALUDE_m_minus_n_equals_three_l934_93463

-- Define the sets M and N
def M (m : ℕ) : Set ℕ := {1, 2, 3, m}
def N (n : ℕ) : Set ℕ := {4, 7, n^4, n^2 + 3*n}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- State the theorem
theorem m_minus_n_equals_three (m n : ℕ) : 
  (∃ y ∈ M m, ∃ z ∈ N n, f y = z) → m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_three_l934_93463


namespace NUMINAMATH_CALUDE_circular_mat_radius_increase_l934_93450

theorem circular_mat_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 40)
  (h2 : final_circumference = 50) : 
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circular_mat_radius_increase_l934_93450


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l934_93406

theorem smallest_sum_of_reciprocals (x y : ℕ) : 
  x ≠ y → 
  x > 0 → 
  y > 0 → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24 → 
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 → 
  x + y ≤ a + b →
  x + y = 98 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l934_93406


namespace NUMINAMATH_CALUDE_inequality_solution_l934_93435

theorem inequality_solution :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} =
  {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l934_93435


namespace NUMINAMATH_CALUDE_art_piece_value_increase_l934_93416

def original_price : ℝ := 4000
def future_price : ℝ := 3 * original_price

theorem art_piece_value_increase : future_price - original_price = 8000 := by
  sorry

end NUMINAMATH_CALUDE_art_piece_value_increase_l934_93416


namespace NUMINAMATH_CALUDE_inequality_proof_l934_93492

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l934_93492


namespace NUMINAMATH_CALUDE_digit_with_value_difference_l934_93497

def numeral : List Nat := [6, 5, 7, 9, 3]

def local_value (digit : Nat) (place : Nat) : Nat :=
  digit * (10 ^ place)

def face_value (digit : Nat) : Nat := digit

theorem digit_with_value_difference (diff : Nat) :
  ∃ (index : Fin 5), 
    local_value (numeral[index]) (4 - index) - face_value (numeral[index]) = diff →
    numeral[index] = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_digit_with_value_difference_l934_93497


namespace NUMINAMATH_CALUDE_optimal_container_dimensions_l934_93440

/-- Represents the dimensions and volume of a rectangular container --/
structure Container where
  shorter_side : Real
  longer_side : Real
  height : Real
  volume : Real

/-- Calculates the volume of a container given its dimensions --/
def calculate_volume (c : Container) : Real :=
  c.shorter_side * c.longer_side * c.height

/-- Defines the constraints for the container based on the problem --/
def container_constraints (c : Container) : Prop :=
  c.longer_side = c.shorter_side + 0.5 ∧
  c.height = 3.2 - 2 * c.shorter_side ∧
  c.volume = calculate_volume c ∧
  0 < c.shorter_side ∧ c.shorter_side < 1.6

/-- Theorem stating the optimal dimensions and maximum volume of the container --/
theorem optimal_container_dimensions :
  ∃ (c : Container), container_constraints c ∧
    c.shorter_side = 1 ∧
    c.height = 1.2 ∧
    c.volume = 1.8 ∧
    ∀ (c' : Container), container_constraints c' → c'.volume ≤ c.volume :=
  sorry

end NUMINAMATH_CALUDE_optimal_container_dimensions_l934_93440


namespace NUMINAMATH_CALUDE_group_size_calculation_l934_93439

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3 →
  old_weight = 70 →
  new_weight = 94 →
  (new_weight - old_weight) / average_increase = 8 := by
sorry

end NUMINAMATH_CALUDE_group_size_calculation_l934_93439


namespace NUMINAMATH_CALUDE_unique_recurrence_sequence_l934_93489

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 > 1 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

/-- The theorem stating the existence and uniqueness of the sequence -/
theorem unique_recurrence_sequence :
  ∃! a : ℕ → ℤ, RecurrenceSequence a :=
sorry

end NUMINAMATH_CALUDE_unique_recurrence_sequence_l934_93489


namespace NUMINAMATH_CALUDE_candy_sampling_problem_l934_93480

theorem candy_sampling_problem (caught_percentage : ℝ) (total_sampling_percentage : ℝ) 
  (h1 : caught_percentage = 22)
  (h2 : total_sampling_percentage = 25) :
  total_sampling_percentage - caught_percentage = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_problem_l934_93480


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l934_93408

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * total_average / (total_families - childless_families : ℚ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l934_93408


namespace NUMINAMATH_CALUDE_complex_equation_solution_l934_93458

theorem complex_equation_solution (z : ℂ) :
  z * (2 - Complex.I) = 10 + 5 * Complex.I → z = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l934_93458


namespace NUMINAMATH_CALUDE_cubic_difference_l934_93441

theorem cubic_difference (x y : ℝ) (h1 : x + y = 14) (h2 : 3 * x + y = 20) :
  x^3 - y^3 = -1304 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l934_93441


namespace NUMINAMATH_CALUDE_negation_of_proposition_l934_93422

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l934_93422


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l934_93407

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l934_93407


namespace NUMINAMATH_CALUDE_no_solution_exists_l934_93467

theorem no_solution_exists : ¬∃ x : ℝ, x^2 * 1 * 3 - x * 1 * 3^2 = 6 := by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l934_93467


namespace NUMINAMATH_CALUDE_range_of_a_when_proposition_is_false_l934_93487

theorem range_of_a_when_proposition_is_false :
  (¬∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_proposition_is_false_l934_93487


namespace NUMINAMATH_CALUDE_rebecca_current_income_l934_93433

/-- Rebecca's current yearly income --/
def rebecca_income : ℝ := sorry

/-- Jimmy's annual income --/
def jimmy_income : ℝ := 18000

/-- The increase in Rebecca's income --/
def income_increase : ℝ := 7000

/-- The percentage of Rebecca's new income in their combined income --/
def rebecca_percentage : ℝ := 0.55

theorem rebecca_current_income :
  rebecca_income = 15000 ∧
  (rebecca_income + income_increase) = 
    rebecca_percentage * (rebecca_income + income_increase + jimmy_income) :=
by sorry

end NUMINAMATH_CALUDE_rebecca_current_income_l934_93433


namespace NUMINAMATH_CALUDE_city_area_most_reliable_xiao_liang_most_reliable_l934_93452

/-- Represents a survey method for assessing elderly health conditions -/
inductive SurveyMethod
  | Hospital
  | SquareDancing
  | CityArea

/-- Represents the reliability of a survey method -/
def reliability (method : SurveyMethod) : ℕ :=
  match method with
  | .Hospital => 1
  | .SquareDancing => 2
  | .CityArea => 3

/-- Theorem stating that the CityArea survey method is the most reliable -/
theorem city_area_most_reliable :
  ∀ (method : SurveyMethod), method ≠ SurveyMethod.CityArea →
    reliability method < reliability SurveyMethod.CityArea :=
by sorry

/-- Corollary: Xiao Liang's survey (CityArea) is the most reliable -/
theorem xiao_liang_most_reliable :
  reliability SurveyMethod.CityArea = max (reliability SurveyMethod.Hospital)
    (max (reliability SurveyMethod.SquareDancing) (reliability SurveyMethod.CityArea)) :=
by sorry

end NUMINAMATH_CALUDE_city_area_most_reliable_xiao_liang_most_reliable_l934_93452


namespace NUMINAMATH_CALUDE_system_solution_l934_93498

theorem system_solution :
  ∃! (x y : ℤ), 16*x + 24*y = 32 ∧ 24*x + 16*y = 48 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l934_93498


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_14_l934_93410

/-- The weight of a bowling ball in pounds -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of a canoe in pounds -/
def canoe_weight : ℝ := sorry

/-- Theorem stating that one bowling ball weighs 14 pounds -/
theorem bowling_ball_weight_is_14 : bowling_ball_weight = 14 := by
  have h1 : 8 * bowling_ball_weight = 4 * canoe_weight := sorry
  have h2 : 3 * canoe_weight = 84 := sorry
  sorry


end NUMINAMATH_CALUDE_bowling_ball_weight_is_14_l934_93410


namespace NUMINAMATH_CALUDE_flag_distribution_l934_93484

theorem flag_distribution (total_flags : ℕ) (blue_flags red_flags : ℕ) :
  total_flags % 2 = 0 →
  blue_flags + red_flags = total_flags →
  (3 * total_flags / 10 : ℚ) = blue_flags →
  (3 * total_flags / 10 : ℚ) = red_flags →
  (total_flags / 10 : ℚ) = (blue_flags + red_flags - total_flags / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l934_93484


namespace NUMINAMATH_CALUDE_max_log_function_l934_93454

theorem max_log_function (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + 2*y = 1/2) :
  ∃ (max_u : ℝ), max_u = 0 ∧ 
  ∀ (u : ℝ), u = Real.log (8*x*y + 4*y^2 + 1) / Real.log (1/2) → u ≤ max_u :=
sorry

end NUMINAMATH_CALUDE_max_log_function_l934_93454


namespace NUMINAMATH_CALUDE_min_value_abs_2a_minus_b_l934_93475

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y : ℝ), 2 * x^2 - y^2 = 1 → |2 * x - y| ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_abs_2a_minus_b_l934_93475


namespace NUMINAMATH_CALUDE_barium_oxide_required_l934_93465

/-- Represents a chemical substance with its number of moles -/
structure Substance where
  name : String
  moles : ℚ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Substance
  products : List Substance

def barium_oxide_water_reaction : Reaction :=
  { reactants := [
      { name := "BaO", moles := 1 },
      { name := "H2O", moles := 1 }
    ],
    products := [
      { name := "Ba(OH)2", moles := 1 }
    ]
  }

theorem barium_oxide_required (water_moles : ℚ) (barium_hydroxide_moles : ℚ) :
  water_moles = barium_hydroxide_moles →
  (∃ (bao : Substance),
    bao.name = "BaO" ∧
    bao.moles = water_moles ∧
    bao.moles = barium_hydroxide_moles ∧
    (∃ (h2o : Substance) (baoh2 : Substance),
      h2o.name = "H2O" ∧
      h2o.moles = water_moles ∧
      baoh2.name = "Ba(OH)2" ∧
      baoh2.moles = barium_hydroxide_moles ∧
      barium_oxide_water_reaction.reactants = [bao, h2o] ∧
      barium_oxide_water_reaction.products = [baoh2])) :=
by
  sorry

end NUMINAMATH_CALUDE_barium_oxide_required_l934_93465


namespace NUMINAMATH_CALUDE_asha_win_probability_l934_93411

theorem asha_win_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 3/7)
  (h_tie : p_tie = 1/5) :
  1 - p_lose - p_tie = 13/35 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l934_93411


namespace NUMINAMATH_CALUDE_no_solution_exists_l934_93426

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3*y, 4],
    ![2*y, y]]

theorem no_solution_exists (y : ℝ) (h : y + 1 = 0) :
  ¬ ∃ y, (3 * y^2 - 8 * y = 5 ∧ y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l934_93426


namespace NUMINAMATH_CALUDE_net_salary_proof_l934_93449

/-- Represents a person's monthly financial situation -/
structure MonthlySalary where
  net : ℝ
  discretionary : ℝ
  remaining : ℝ

/-- Calculates the net monthly salary given the conditions -/
def calculate_net_salary (m : MonthlySalary) : Prop :=
  m.discretionary = m.net / 5 ∧
  m.remaining = m.discretionary * 0.1 ∧
  m.remaining = 105 ∧
  m.net = 5250

theorem net_salary_proof (m : MonthlySalary) :
  calculate_net_salary m → m.net = 5250 := by
  sorry

end NUMINAMATH_CALUDE_net_salary_proof_l934_93449


namespace NUMINAMATH_CALUDE_company_production_days_l934_93479

/-- Given a company's production data, prove the number of past days. -/
theorem company_production_days (n : ℕ) : 
  (∀ (P : ℕ), P = 80 * n) →  -- Average daily production for past n days
  (∀ (new_total : ℕ), new_total = 80 * n + 220) →  -- Total including today's production
  (∀ (new_avg : ℝ), new_avg = (80 * n + 220) / (n + 1)) →  -- New average
  (new_avg = 95) →  -- New average is 95
  n = 8 := by sorry

end NUMINAMATH_CALUDE_company_production_days_l934_93479


namespace NUMINAMATH_CALUDE_job_completion_time_l934_93437

/-- The time taken for machines to complete a job given specific conditions -/
theorem job_completion_time : 
  -- Machine R completion time
  let r_time : ℝ := 36
  -- Machine S completion time
  let s_time : ℝ := 2
  -- Number of each type of machine used
  let n : ℝ := 0.9473684210526315
  -- Total rate of job completion
  let total_rate : ℝ := n * (1 / r_time) + n * (1 / s_time)
  -- Time taken to complete the job
  let completion_time : ℝ := 1 / total_rate
  -- Proof that the completion time is 2 hours
  completion_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l934_93437


namespace NUMINAMATH_CALUDE_triangle_inradius_l934_93460

/-- Given a triangle with perimeter 39 cm and area 29.25 cm², its inradius is 1.5 cm -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 39) 
  (h2 : A = 29.25) 
  (h3 : A = r * p / 2) : 
  r = 1.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_inradius_l934_93460


namespace NUMINAMATH_CALUDE_range_of_m_l934_93493

theorem range_of_m (a b m : ℝ) (h1 : 3 * a + 4 / b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : ∀ (a b : ℝ), a > 0 → b > 0 → 3 * a + 4 / b = 1 → 1 / a + 3 * b > m) :
  m < 27 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l934_93493


namespace NUMINAMATH_CALUDE_divisibility_circle_l934_93419

/-- Given seven natural numbers in a circle where each adjacent pair has a divisibility relation,
    there exists a non-adjacent pair with the same property. -/
theorem divisibility_circle (a : Fin 7 → ℕ) 
  (h : ∀ i : Fin 7, (a i ∣ a (i + 1)) ∨ (a (i + 1) ∣ a i)) :
  ∃ i j : Fin 7, i ≠ j ∧ (j ≠ i + 1) ∧ (j ≠ i - 1) ∧ ((a i ∣ a j) ∨ (a j ∣ a i)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_circle_l934_93419


namespace NUMINAMATH_CALUDE_fixed_costs_correct_l934_93400

/-- Represents the fixed monthly costs for producing electronic components -/
def fixed_monthly_costs : ℝ := 16500

/-- Represents the production cost per component -/
def production_cost_per_unit : ℝ := 80

/-- Represents the shipping cost per component -/
def shipping_cost_per_unit : ℝ := 5

/-- Represents the number of components produced and sold monthly -/
def monthly_units : ℕ := 150

/-- Represents the lowest selling price per component -/
def lowest_selling_price : ℝ := 195

/-- Theorem stating that the fixed monthly costs are correct given the conditions -/
theorem fixed_costs_correct :
  fixed_monthly_costs =
    monthly_units * lowest_selling_price -
    monthly_units * (production_cost_per_unit + shipping_cost_per_unit) := by
  sorry

#check fixed_costs_correct

end NUMINAMATH_CALUDE_fixed_costs_correct_l934_93400


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l934_93414

def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) :
  geometric_sequence a r → a 1 = 1 → r = -2 →
  a 1 + |a 2| + |a 3| + a 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l934_93414


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l934_93486

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_equals_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l934_93486


namespace NUMINAMATH_CALUDE_enlarged_parallelepiped_volume_equals_l934_93430

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped with dimensions 4 by 5 by 6 units -/
def enlarged_parallelepiped_volume : ℝ := sorry

/-- The dimensions of the original parallelepiped -/
def original_dimensions : Fin 3 → ℕ
| 0 => 4
| 1 => 5
| 2 => 6
| _ => 0

theorem enlarged_parallelepiped_volume_equals : 
  enlarged_parallelepiped_volume = (1884 + 139 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_enlarged_parallelepiped_volume_equals_l934_93430


namespace NUMINAMATH_CALUDE_larger_number_l934_93448

theorem larger_number (a b : ℝ) (h1 : a - b = 6) (h2 : a + b = 40) : max a b = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l934_93448


namespace NUMINAMATH_CALUDE_g_2000_divisors_l934_93473

/-- g(n) is the smallest power of 5 such that 1/g(n) has exactly n digits after the decimal point -/
def g (n : ℕ) : ℕ := 5^n

/-- The number of positive integer divisors of x -/
def num_divisors (x : ℕ) : ℕ := sorry

theorem g_2000_divisors : num_divisors (g 2000) = 2001 := by sorry

end NUMINAMATH_CALUDE_g_2000_divisors_l934_93473


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l934_93423

/-- Given that 3 dozen apples cost $23.40, prove that 5 dozen apples at the same rate cost $39.00 -/
theorem apple_cost_calculation (cost_three_dozen : ℝ) (h1 : cost_three_dozen = 23.40) :
  let cost_per_dozen : ℝ := cost_three_dozen / 3
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 39.00 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l934_93423


namespace NUMINAMATH_CALUDE_expression_evaluation_l934_93427

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 2
  5 * x^y + 2 * y^x + x^2 * y^2 = 97 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l934_93427


namespace NUMINAMATH_CALUDE_circle_radius_proof_l934_93476

theorem circle_radius_proof (a : ℝ) : 
  a > 0 ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = 2*x^2 - 27 ∧ (x - a)^2 + (y - (2*a^2 - 27))^2 = a^2) ∧
  a^2 = (4*a - 3*(2*a^2 - 27))^2 / (4^2 + 3^2) →
  a = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l934_93476


namespace NUMINAMATH_CALUDE_complex_modulus_range_l934_93470

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-1/2) (1/2) := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l934_93470
