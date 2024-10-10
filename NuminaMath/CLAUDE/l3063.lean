import Mathlib

namespace isabellas_house_paintable_area_l3063_306356

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable wall area in all bedrooms -/
def totalPaintableArea (dimensions : BedroomDimensions) (numBedrooms : ℕ) (nonPaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableAreaPerRoom := wallArea - nonPaintableArea
  numBedrooms * paintableAreaPerRoom

/-- Theorem stating that the total paintable wall area in Isabella's house is 1194 square feet -/
theorem isabellas_house_paintable_area :
  let dimensions : BedroomDimensions := { length := 15, width := 11, height := 9 }
  let numBedrooms : ℕ := 3
  let nonPaintableArea : ℝ := 70
  totalPaintableArea dimensions numBedrooms nonPaintableArea = 1194 := by
  sorry


end isabellas_house_paintable_area_l3063_306356


namespace annual_cost_difference_l3063_306349

/-- Calculates the annual cost difference between combined piano, violin, and singing lessons
    and clarinet lessons, given the hourly rates and weekly hours for each lesson type. -/
theorem annual_cost_difference
  (clarinet_rate : ℕ) (clarinet_hours : ℕ)
  (piano_rate : ℕ) (piano_hours : ℕ)
  (violin_rate : ℕ) (violin_hours : ℕ)
  (singing_rate : ℕ) (singing_hours : ℕ)
  (h1 : clarinet_rate = 40)
  (h2 : clarinet_hours = 3)
  (h3 : piano_rate = 28)
  (h4 : piano_hours = 5)
  (h5 : violin_rate = 35)
  (h6 : violin_hours = 2)
  (h7 : singing_rate = 45)
  (h8 : singing_hours = 1)
  : (piano_rate * piano_hours + violin_rate * violin_hours + singing_rate * singing_hours) * 52 -
    (clarinet_rate * clarinet_hours) * 52 = 7020 := by
  sorry

#eval (28 * 5 + 35 * 2 + 45 * 1) * 52 - (40 * 3) * 52

end annual_cost_difference_l3063_306349


namespace min_minutes_for_plan_c_l3063_306340

/-- Represents the cost of a cell phone plan in cents -/
def PlanCost (flatFee minutes perMinute : ℕ) : ℕ := flatFee * 100 + minutes * perMinute

/-- Checks if Plan C is cheaper than both Plan A and Plan B for a given number of minutes -/
def IsPlanCCheaper (minutes : ℕ) : Prop :=
  PlanCost 15 minutes 10 < PlanCost 0 minutes 15 ∧ 
  PlanCost 15 minutes 10 < PlanCost 25 minutes 8

theorem min_minutes_for_plan_c : ∀ m : ℕ, m ≥ 301 → IsPlanCCheaper m ∧ ∀ n : ℕ, n < 301 → ¬IsPlanCCheaper n := by
  sorry

end min_minutes_for_plan_c_l3063_306340


namespace train_length_l3063_306327

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 7 → ∃ length : ℝ, 
  (length ≥ 116.68 ∧ length ≤ 116.70) ∧ 
  length = speed * 1000 / 3600 * time :=
sorry

end train_length_l3063_306327


namespace max_value_fraction_l3063_306388

theorem max_value_fraction (x : ℝ) :
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ 15 ∧
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 := by
  sorry

end max_value_fraction_l3063_306388


namespace tangent_power_equality_l3063_306358

open Complex

theorem tangent_power_equality (α : ℝ) (n : ℕ) :
  ((1 + I * Real.tan α) / (1 - I * Real.tan α)) ^ n = 
  (1 + I * Real.tan (n * α)) / (1 - I * Real.tan (n * α)) := by
  sorry

end tangent_power_equality_l3063_306358


namespace negation_of_existence_l3063_306303

theorem negation_of_existence (p : Prop) :
  (¬∃ (n : ℕ), n > 1 ∧ n^2 > 2^n) ↔ (∀ (n : ℕ), n > 1 → n^2 ≤ 2^n) :=
sorry

end negation_of_existence_l3063_306303


namespace unknown_number_l3063_306319

theorem unknown_number (x n : ℝ) : 
  (5 * x + n = 10 * x - 17) → (x = 4) → (n = 3) := by
  sorry

end unknown_number_l3063_306319


namespace constant_point_of_quadratic_l3063_306389

/-- The quadratic function f(x) that depends on a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

/-- The theorem stating that (2, 13) is the unique constant point for f(x) -/
theorem constant_point_of_quadratic :
  ∃! p : ℝ × ℝ, ∀ m : ℝ, f m p.1 = p.2 ∧ p = (2, 13) :=
sorry

end constant_point_of_quadratic_l3063_306389


namespace systematic_sample_theorem_l3063_306350

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the number of sampled elements within the given interval -/
def elements_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) * s.sample_size + s.population - 1) / s.population

/-- The main theorem stating that for the given scenario, 12 people fall within the interval -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population = 840)
  (h2 : s.sample_size = 42)
  (h3 : s.interval_start = 481)
  (h4 : s.interval_end = 720) :
  elements_in_interval s = 12 := by
  sorry

end systematic_sample_theorem_l3063_306350


namespace eggs_left_for_breakfast_l3063_306391

def total_eggs : ℕ := 3 * 12

def eggs_for_crepes : ℕ := total_eggs / 3

def eggs_after_crepes : ℕ := total_eggs - eggs_for_crepes

def eggs_for_cupcakes : ℕ := (eggs_after_crepes * 3) / 5

def eggs_left : ℕ := eggs_after_crepes - eggs_for_cupcakes

theorem eggs_left_for_breakfast : eggs_left = 10 := by
  sorry

end eggs_left_for_breakfast_l3063_306391


namespace wilson_sledding_l3063_306336

theorem wilson_sledding (tall_hills small_hills tall_runs small_runs : ℕ) 
  (h1 : tall_hills = 2)
  (h2 : small_hills = 3)
  (h3 : tall_runs = 4)
  (h4 : small_runs = tall_runs / 2)
  : tall_hills * tall_runs + small_hills * small_runs = 14 := by
  sorry

end wilson_sledding_l3063_306336


namespace complex_magnitude_squared_l3063_306393

theorem complex_magnitude_squared (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_magnitude_squared_l3063_306393


namespace divisibility_by_seven_l3063_306359

theorem divisibility_by_seven (a b : ℕ) : 
  (7 ∣ (a * b)) → (7 ∣ a) ∨ (7 ∣ b) := by
  sorry

end divisibility_by_seven_l3063_306359


namespace no_two_obtuse_angles_l3063_306347

-- Define a triangle as a structure with three angles
structure Triangle where
  a : Real
  b : Real
  c : Real
  angle_sum : a + b + c = 180
  positive_angles : 0 < a ∧ 0 < b ∧ 0 < c

-- Theorem: A triangle cannot have two obtuse angles
theorem no_two_obtuse_angles (t : Triangle) : ¬(t.a > 90 ∧ t.b > 90) ∧ ¬(t.a > 90 ∧ t.c > 90) ∧ ¬(t.b > 90 ∧ t.c > 90) := by
  sorry

end no_two_obtuse_angles_l3063_306347


namespace parabola_curve_intersection_l3063_306380

/-- The parabola defined by y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The curve defined by y = k/x where k > 0 -/
def curve (k x y : ℝ) : Prop := k > 0 ∧ y = k / x

/-- The focus of the parabola y = (1/4)x^2 -/
def focus : ℝ × ℝ := (0, 1)

/-- A point is on both the parabola and the curve -/
def intersection_point (k x y : ℝ) : Prop :=
  parabola x y ∧ curve k x y

/-- The line from a point to the focus is perpendicular to the y-axis -/
def perpendicular_to_y_axis (x y : ℝ) : Prop :=
  x = (focus.1 - x)

theorem parabola_curve_intersection (k : ℝ) :
  (∃ x y : ℝ, intersection_point k x y ∧ perpendicular_to_y_axis x y) →
  k = 2 :=
sorry

end parabola_curve_intersection_l3063_306380


namespace six_balls_three_boxes_l3063_306376

/-- The number of ways to put n indistinguishable balls into k distinguishable boxes -/
def ball_distribution (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to put 6 indistinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : ball_distribution 6 3 = 28 := by
  sorry

end six_balls_three_boxes_l3063_306376


namespace machine_value_depletion_rate_l3063_306330

/-- Proves that the annual value depletion rate is 0.1 for a machine with given initial and final values over 2 years -/
theorem machine_value_depletion_rate
  (initial_value : ℝ)
  (final_value : ℝ)
  (time_period : ℝ)
  (h1 : initial_value = 900)
  (h2 : final_value = 729)
  (h3 : time_period = 2)
  : ∃ (rate : ℝ), rate = 0.1 ∧ final_value = initial_value * (1 - rate) ^ time_period :=
sorry

end machine_value_depletion_rate_l3063_306330


namespace sum_in_base8_l3063_306346

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def base8ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: natToBase8 (n / 8)

theorem sum_in_base8 :
  let a := base8ToNat [3, 5, 6]  -- 653₈
  let b := base8ToNat [4, 7, 2]  -- 274₈
  let c := base8ToNat [7, 6, 1]  -- 167₈
  natToBase8 (a + b + c) = [6, 5, 3, 1] := by
  sorry

end sum_in_base8_l3063_306346


namespace complex_equation_solution_l3063_306371

theorem complex_equation_solution (i : ℂ) (x : ℂ) (h1 : i * i = -1) (h2 : i * x = 1 + i) : x = 1 - i := by
  sorry

end complex_equation_solution_l3063_306371


namespace fraction_to_decimal_l3063_306366

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l3063_306366


namespace negative_a_squared_cubed_div_negative_a_squared_l3063_306333

theorem negative_a_squared_cubed_div_negative_a_squared (a : ℝ) :
  (-a^2)^3 / (-a)^2 = -a^4 := by
  sorry

end negative_a_squared_cubed_div_negative_a_squared_l3063_306333


namespace jaylen_vegetables_l3063_306351

theorem jaylen_vegetables (x y z g k h : ℕ) : 
  x = 5 → 
  y = 2 → 
  z = 2 * k → 
  g = (h / 2) - 3 → 
  k = 2 → 
  h = 20 → 
  x + y + z + g = 18 := by
sorry

end jaylen_vegetables_l3063_306351


namespace side_significant_digits_l3063_306338

-- Define the area of the square
def square_area : Real := 3.0625

-- Define the precision of the area measurement
def area_precision : Real := 0.001

-- Define a function to calculate the number of significant digits
def count_significant_digits (x : Real) : Nat :=
  sorry

-- Theorem statement
theorem side_significant_digits :
  let side := Real.sqrt square_area
  count_significant_digits side = 3 :=
sorry

end side_significant_digits_l3063_306338


namespace chloe_dimes_needed_l3063_306329

/-- Represents the minimum number of dimes needed to purchase a hoodie -/
def min_dimes_needed (hoodie_cost : ℚ) (ten_dollar_bills : ℕ) (quarters : ℕ) (one_dollar_coins : ℕ) : ℕ :=
  let current_money : ℚ := 10 * ten_dollar_bills + 0.25 * quarters + one_dollar_coins
  ⌈(hoodie_cost - current_money) / 0.1⌉₊

/-- Theorem stating that Chloe needs 0 additional dimes to buy the hoodie -/
theorem chloe_dimes_needed : 
  min_dimes_needed 45.50 4 10 3 = 0 := by
  sorry

#eval min_dimes_needed 45.50 4 10 3

end chloe_dimes_needed_l3063_306329


namespace golden_ratio_less_than_one_l3063_306334

theorem golden_ratio_less_than_one : (Real.sqrt 5 - 1) / 2 < 1 := by
  sorry

end golden_ratio_less_than_one_l3063_306334


namespace subset_proof_l3063_306353

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem subset_proof : M ⊆ N := by sorry

end subset_proof_l3063_306353


namespace window_pane_length_l3063_306317

theorem window_pane_length 
  (num_panes : ℕ) 
  (pane_width : ℝ) 
  (total_area : ℝ) : ℝ :=
  let pane_area := total_area / num_panes
  let pane_length := pane_area / pane_width
  have h1 : num_panes = 8 := by sorry
  have h2 : pane_width = 8 := by sorry
  have h3 : total_area = 768 := by sorry
  have h4 : pane_length = 12 := by sorry
  pane_length

#check window_pane_length

end window_pane_length_l3063_306317


namespace people_in_room_l3063_306370

/-- Proves that given the conditions in the problem, the number of people in the room is 67 -/
theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs →  -- Three-fifths of people are seated in four-fifths of chairs
  chairs - (4 : ℚ) / 5 * chairs = 10 →           -- 10 chairs are empty
  people = 67 := by
sorry


end people_in_room_l3063_306370


namespace product_of_one_plus_roots_l3063_306360

theorem product_of_one_plus_roots (u v w : ℝ) : 
  u^3 - 15*u^2 + 25*u - 12 = 0 ∧ 
  v^3 - 15*v^2 + 25*v - 12 = 0 ∧ 
  w^3 - 15*w^2 + 25*w - 12 = 0 → 
  (1 + u) * (1 + v) * (1 + w) = 29 := by
sorry

end product_of_one_plus_roots_l3063_306360


namespace kevin_cards_problem_l3063_306341

/-- Given that Kevin finds 47 cards and ends up with 54 cards, prove that he started with 7 cards. -/
theorem kevin_cards_problem (found_cards : ℕ) (total_cards : ℕ) (h1 : found_cards = 47) (h2 : total_cards = 54) :
  total_cards - found_cards = 7 := by
sorry

end kevin_cards_problem_l3063_306341


namespace solve_for_q_l3063_306325

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/60)
  (h2 : 5/6 = (m - n)/66)
  (h3 : 5/6 = (q - m)/150) : 
  q = 230 := by
sorry

end solve_for_q_l3063_306325


namespace ant_return_probability_l3063_306326

/-- Represents the probability of an ant returning to its starting vertex
    after n steps on a regular tetrahedron. -/
def P (n : ℕ) : ℚ :=
  1/4 - 1/4 * (-1/3)^(n-1)

/-- The probability of an ant returning to its starting vertex
    after 6 steps on a regular tetrahedron with edge length 1. -/
theorem ant_return_probability :
  P 6 = 61/243 :=
sorry

end ant_return_probability_l3063_306326


namespace stating_sticks_at_100th_stage_l3063_306342

/-- 
Given a sequence where:
- The first term is 4
- Each subsequent term increases by 4
This function calculates the nth term of the sequence
-/
def sticksAtStage (n : ℕ) : ℕ := 4 + 4 * (n - 1)

/-- 
Theorem stating that the 100th stage of the stick pattern contains 400 sticks
-/
theorem sticks_at_100th_stage : sticksAtStage 100 = 400 := by sorry

end stating_sticks_at_100th_stage_l3063_306342


namespace mini_football_betting_strategy_l3063_306385

theorem mini_football_betting_strategy :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
    x₁ + x₂ + x₃ + x₄ = 1 ∧
    3 * x₁ ≥ 1 ∧
    4 * x₂ ≥ 1 ∧
    5 * x₃ ≥ 1 ∧
    8 * x₄ ≥ 1 :=
by sorry

end mini_football_betting_strategy_l3063_306385


namespace count_valid_digits_l3063_306396

theorem count_valid_digits : 
  let is_valid (A : ℕ) := 0 ≤ A ∧ A ≤ 9 ∧ 571 * 10 + A < 5716
  (Finset.filter is_valid (Finset.range 10)).card = 6 := by
sorry

end count_valid_digits_l3063_306396


namespace binary_1101_is_13_l3063_306363

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end binary_1101_is_13_l3063_306363


namespace one_hundred_fiftieth_term_l3063_306372

/-- An arithmetic sequence with first term 2 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 2 + (n - 1) * 5

theorem one_hundred_fiftieth_term :
  arithmeticSequence 150 = 747 := by
  sorry

end one_hundred_fiftieth_term_l3063_306372


namespace gcd_91_49_l3063_306369

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end gcd_91_49_l3063_306369


namespace f_of_3_equals_neg_9_l3063_306305

/-- Given a function f(x) = 2x^7 - 3x^3 + 4x - 6 where f(-3) = -3, prove that f(3) = -9 -/
theorem f_of_3_equals_neg_9 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2*x^7 - 3*x^3 + 4*x - 6)
  (h2 : f (-3) = -3) : 
  f 3 = -9 := by
sorry


end f_of_3_equals_neg_9_l3063_306305


namespace unique_rearrangement_difference_l3063_306324

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    max a (max b c) * 100 + (a + b + c - max a (max b c) - min a (min b c)) * 10 + min a (min b c) -
    (min a (min b c) * 100 + (a + b + c - max a (max b c) - min a (min b c)) * 10 + max a (max b c)) = n

theorem unique_rearrangement_difference :
  ∃! n : ℕ, is_valid_number n :=
by sorry

end unique_rearrangement_difference_l3063_306324


namespace layer_cake_frosting_usage_l3063_306337

/-- Represents the amount of frosting in cans used for different types of baked goods. -/
structure FrostingUsage where
  single_cake : ℚ
  pan_brownies : ℚ
  dozen_cupcakes : ℚ
  layer_cake : ℚ

/-- Represents the quantities of different baked goods to be frosted. -/
structure BakedGoods where
  layer_cakes : ℕ
  dozen_cupcakes : ℕ
  single_cakes : ℕ
  pans_brownies : ℕ

/-- Calculates the total number of cans of frosting needed for a given set of baked goods and frosting usage. -/
def total_frosting (usage : FrostingUsage) (goods : BakedGoods) : ℚ :=
  usage.layer_cake * goods.layer_cakes +
  usage.dozen_cupcakes * goods.dozen_cupcakes +
  usage.single_cake * goods.single_cakes +
  usage.pan_brownies * goods.pans_brownies

/-- The main theorem stating that the amount of frosting used for a layer cake is 1 can. -/
theorem layer_cake_frosting_usage
  (usage : FrostingUsage)
  (goods : BakedGoods)
  (h1 : usage.single_cake = 1/2)
  (h2 : usage.pan_brownies = 1/2)
  (h3 : usage.dozen_cupcakes = 1/2)
  (h4 : goods.layer_cakes = 3)
  (h5 : goods.dozen_cupcakes = 6)
  (h6 : goods.single_cakes = 12)
  (h7 : goods.pans_brownies = 18)
  (h8 : total_frosting usage goods = 21)
  : usage.layer_cake = 1 := by
  sorry

end layer_cake_frosting_usage_l3063_306337


namespace not_proportional_D_l3063_306384

-- Define the equations
def equation_A (x y : ℝ) : Prop := x + y = 5
def equation_B (x y : ℝ) : Prop := 4 * x * y = 12
def equation_C (x y : ℝ) : Prop := x = 3 * y
def equation_D (x y : ℝ) : Prop := 4 * x + 2 * y = 8
def equation_E (x y : ℝ) : Prop := x / y = 4

-- Define direct and inverse proportionality
def directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Theorem statement
theorem not_proportional_D :
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_A x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_B x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_C x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_E x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  ¬(∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_D x y ↔ y = f x) ∧
                 (directly_proportional f ∨ inversely_proportional f)) :=
by sorry

end not_proportional_D_l3063_306384


namespace intersection_fixed_point_l3063_306312

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (k n x y : ℝ) : Prop := y = k * x + n

-- Define the intersection points
def intersection (k n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k n x₁ y₁ ∧ line k n x₂ y₂

-- Main theorem
theorem intersection_fixed_point (k n x₁ y₁ x₂ y₂ : ℝ) 
  (hk : k ≠ 0)
  (h_int : intersection k n x₁ y₁ x₂ y₂)
  (h_slope : 3 * (y₁ / x₁ + y₂ / x₂) = 8 * k) :
  n = 1/2 ∨ n = -1/2 := by
  sorry

end

end intersection_fixed_point_l3063_306312


namespace woman_birth_year_l3063_306315

theorem woman_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 - x ≥ 1950) (h3 : x^2 - x < 2000) (h4 : x^2 ≥ 2000) : x^2 - x = 1980 := by
  sorry

end woman_birth_year_l3063_306315


namespace mike_gave_ten_books_l3063_306331

/-- The number of books Mike gave to Lily -/
def books_from_mike : ℕ := sorry

/-- The number of books Corey gave to Lily -/
def books_from_corey : ℕ := sorry

/-- The total number of books Lily received -/
def total_books : ℕ := 35

theorem mike_gave_ten_books :
  (books_from_mike = 10) ∧
  (books_from_corey = books_from_mike + 15) ∧
  (books_from_mike + books_from_corey = total_books) :=
sorry

end mike_gave_ten_books_l3063_306331


namespace tangent_lines_and_intersection_l3063_306311

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (-3, 2)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = -3
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y + 1 = 0

-- Define the circle with diameter PC
def circle_PC (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - 2*y - 1 = 0

theorem tangent_lines_and_intersection (x y : ℝ) :
  (∀ x y, circle_C x y → (tangent_line_1 x ∨ tangent_line_2 x y)) ∧
  (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    circle_PC A.1 A.2 ∧ circle_PC B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (8*Real.sqrt 5 / 5)^2) :=
by sorry

end tangent_lines_and_intersection_l3063_306311


namespace area_triangle_dbc_l3063_306368

/-- Given a triangle ABC with vertices A(0,8), B(0,0), C(10,0), and midpoints D of AB and E of BC,
    the area of triangle DBC is 20. -/
theorem area_triangle_dbc (A B C D E : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (1 / 2 : ℝ) * 10 * 4 = 20 := by
sorry

end area_triangle_dbc_l3063_306368


namespace real_part_of_complex_difference_times_i_l3063_306397

theorem real_part_of_complex_difference_times_i :
  let z₁ : ℂ := 4 + 29 * Complex.I
  let z₂ : ℂ := 6 + 9 * Complex.I
  (z₁ - z₂) * Complex.I |>.re = 20 := by
sorry

end real_part_of_complex_difference_times_i_l3063_306397


namespace unique_seventh_digit_l3063_306314

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def digit_sum (n : ℕ) : ℕ := sorry

theorem unique_seventh_digit (a b : ℕ) (h1 : is_seven_digit a) (h2 : b = digit_sum a) 
  (h3 : is_seven_digit (a - b)) (h4 : ∃ (d : Fin 7 → ℕ), 
    (∀ i, d i ∈ ({1, 2, 3, 4, 6, 7} : Set ℕ)) ∧ 
    (∃ j, d j ∉ ({1, 2, 3, 4, 6, 7} : Set ℕ)) ∧
    (a - b = d 0 * 1000000 + d 1 * 100000 + d 2 * 10000 + d 3 * 1000 + d 4 * 100 + d 5 * 10 + d 6)) :
  ∃! x, x ∉ ({1, 2, 3, 4, 6, 7} : Set ℕ) ∧ x < 10 ∧ 
    (a - b = x * 1000000 + 1 * 100000 + 2 * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + x * 100000 + 2 * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + x * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + x * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + x * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + 6 * 100 + x * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + 6 * 100 + 7 * 10 + x) :=
by sorry

end unique_seventh_digit_l3063_306314


namespace power_equality_implies_exponent_l3063_306309

theorem power_equality_implies_exponent (n : ℕ) : 4^8 = 16^n → n = 4 := by
  sorry

end power_equality_implies_exponent_l3063_306309


namespace integral_2x_plus_exp_x_l3063_306398

open Real MeasureTheory Interval

theorem integral_2x_plus_exp_x : ∫ x in (-1)..(1), (2 * x + Real.exp x) = Real.exp 1 - Real.exp (-1) := by
  sorry

end integral_2x_plus_exp_x_l3063_306398


namespace sum_of_roots_l3063_306365

theorem sum_of_roots (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) := by
sorry

end sum_of_roots_l3063_306365


namespace pentagon_area_l3063_306323

/-- Given integers p and q where 0 < q < p, and points P, Q, R, S, T defined by reflections,
    if the area of pentagon PQRST is 700, then 5pq - q² = 700 -/
theorem pentagon_area (p q : ℤ) (h1 : 0 < q) (h2 : q < p) 
  (h3 : (5 * p * q - q^2 : ℤ) = 700) : True := by
  sorry

end pentagon_area_l3063_306323


namespace seven_digit_palindromes_count_l3063_306343

/-- A function that counts the number of seven-digit palindromes with leading digit 1 or 2 -/
def count_seven_digit_palindromes : ℕ :=
  let leading_digits := 2  -- Number of choices for the leading digit (1 or 2)
  let middle_digits := 10 * 10 * 10  -- Number of choices for the middle three digits
  leading_digits * middle_digits

/-- Theorem stating that the number of seven-digit palindromes with leading digit 1 or 2 is 2000 -/
theorem seven_digit_palindromes_count : count_seven_digit_palindromes = 2000 := by
  sorry

end seven_digit_palindromes_count_l3063_306343


namespace product_division_theorem_l3063_306302

theorem product_division_theorem :
  ∃ x : ℕ, (400 * 7000 : ℕ) = x * (100^1) ∧ x = 28000 := by sorry

end product_division_theorem_l3063_306302


namespace geometric_sequence_common_ratio_l3063_306332

theorem geometric_sequence_common_ratio 
  (a : ℝ) : 
  let seq := λ (n : ℕ) => a + Real.log 3 / Real.log (2^(2^n))
  ∃ (q : ℝ), q = 1/3 ∧ ∀ (n : ℕ), seq (n+1) / seq n = q :=
by
  sorry

end geometric_sequence_common_ratio_l3063_306332


namespace complex_square_real_implies_zero_l3063_306373

theorem complex_square_real_implies_zero (x : ℝ) :
  (Complex.I + x)^2 ∈ Set.range Complex.ofReal → x = 0 := by
  sorry

end complex_square_real_implies_zero_l3063_306373


namespace remainder_problem_l3063_306387

theorem remainder_problem (d r : ℤ) : 
  d > 1 → 
  1122 % d = r → 
  1540 % d = r → 
  2455 % d = r → 
  d - r = 1 := by
sorry

end remainder_problem_l3063_306387


namespace factor_proof_l3063_306375

theorem factor_proof : 
  (∃ n : ℤ, 65 = 5 * n) ∧ (∃ m : ℤ, 144 = 9 * m) := by sorry

end factor_proof_l3063_306375


namespace total_gifts_needed_l3063_306352

/-- The number of teams participating in the world cup -/
def num_teams : ℕ := 12

/-- The number of invited members per team who receive a gift -/
def members_per_team : ℕ := 4

/-- Theorem stating the total number of gifts needed for the event -/
theorem total_gifts_needed : num_teams * members_per_team = 48 := by
  sorry

end total_gifts_needed_l3063_306352


namespace g_value_l3063_306378

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^3 - 2*x - 2
axiom sum_eq : ∀ x, f x + g x = -2 + x

-- State the theorem
theorem g_value : g = fun x ↦ -x^3 + 3*x := by sorry

end g_value_l3063_306378


namespace fraction_to_decimal_l3063_306313

theorem fraction_to_decimal : (7 : ℚ) / 125 = (56 : ℚ) / 1000 := by sorry

end fraction_to_decimal_l3063_306313


namespace dog_bones_total_l3063_306307

theorem dog_bones_total (initial_bones dug_up_bones : ℕ) 
  (h1 : initial_bones = 493) 
  (h2 : dug_up_bones = 367) : 
  initial_bones + dug_up_bones = 860 := by
  sorry

end dog_bones_total_l3063_306307


namespace min_h_21_l3063_306399

-- Define a tenuous function
def Tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → h x + h y > y^2

-- Define the sum of h from 1 to 30
def SumH (h : ℕ → ℤ) : ℤ :=
  (List.range 30).map (λ i => h (i + 1)) |>.sum

-- Theorem statement
theorem min_h_21 (h : ℕ → ℤ) (hTenuous : Tenuous h) (hMinSum : ∀ g : ℕ → ℤ, Tenuous g → SumH g ≥ SumH h) :
  h 21 ≥ 312 := by
  sorry

end min_h_21_l3063_306399


namespace police_emergency_number_prime_divisor_l3063_306382

/-- A police emergency number is a positive integer that ends with 133 in decimal representation -/
def IsPoliceEmergencyNumber (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7 -/
theorem police_emergency_number_prime_divisor (n : ℕ) (h : IsPoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end police_emergency_number_prime_divisor_l3063_306382


namespace right_to_left_grouping_equivalence_l3063_306300

/-- Right-to-left grouping evaluation function -/
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ := a - b * (c + d)

/-- Standard algebraic notation evaluation function -/
noncomputable def standardEval (a b c d : ℝ) : ℝ := a - b * (c + d)

/-- Theorem stating that right-to-left grouping of a - b × c + d
    is equivalent to a - b(c + d) in standard algebraic notation -/
theorem right_to_left_grouping_equivalence (a b c d : ℝ) :
  rightToLeftEval a b c d = standardEval a b c d := by
  sorry

end right_to_left_grouping_equivalence_l3063_306300


namespace fraction_of_powers_equals_500_l3063_306394

theorem fraction_of_powers_equals_500 : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by sorry

end fraction_of_powers_equals_500_l3063_306394


namespace missing_number_in_set_l3063_306357

theorem missing_number_in_set (x : ℝ) (a : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 := by
sorry

end missing_number_in_set_l3063_306357


namespace odd_sum_probability_in_4x4_grid_l3063_306377

theorem odd_sum_probability_in_4x4_grid : 
  let n : ℕ := 16
  let grid_size : ℕ := 4
  let total_arrangements : ℕ := n.factorial
  let valid_arrangements : ℕ := (Nat.choose grid_size 2)^2 * (n/2).factorial * (n/2).factorial
  (valid_arrangements : ℚ) / total_arrangements = 1 / 360 := by
sorry

end odd_sum_probability_in_4x4_grid_l3063_306377


namespace combined_savings_difference_l3063_306301

/-- Regular price of a window -/
def regular_price : ℕ := 120

/-- Number of windows needed to get one free -/
def windows_for_free : ℕ := 5

/-- Discount per window for large purchases -/
def bulk_discount : ℕ := 10

/-- Minimum number of windows for bulk discount -/
def bulk_min : ℕ := 10

/-- Number of windows Alice needs -/
def alice_windows : ℕ := 9

/-- Number of windows Bob needs -/
def bob_windows : ℕ := 12

/-- Calculate the cost of windows with discounts applied -/
def calculate_cost (num_windows : ℕ) : ℕ :=
  let free_windows := num_windows / windows_for_free
  let paid_windows := num_windows - free_windows
  let price_per_window := if num_windows > bulk_min then regular_price - bulk_discount else regular_price
  paid_windows * price_per_window

/-- Calculate savings compared to regular price -/
def calculate_savings (num_windows : ℕ) : ℕ :=
  num_windows * regular_price - calculate_cost num_windows

/-- Theorem: Combined savings minus individual savings equals 300 -/
theorem combined_savings_difference : 
  calculate_savings (alice_windows + bob_windows) - 
  (calculate_savings alice_windows + calculate_savings bob_windows) = 300 := by
  sorry

end combined_savings_difference_l3063_306301


namespace apples_on_tree_l3063_306379

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 9

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := initial_apples - picked_apples

theorem apples_on_tree : remaining_apples = 7 := by
  sorry

end apples_on_tree_l3063_306379


namespace vector_at_t_4_l3063_306381

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfies the conditions -/
def given_line : ParameterizedLine :=
  { point := sorry }

theorem vector_at_t_4 (line : ParameterizedLine) 
  (h1 : line.point (-2) = (2, 6, 16)) 
  (h2 : line.point 1 = (-1, -5, -10)) :
  line.point 4 = (0, -4, -4) := by
  sorry

end vector_at_t_4_l3063_306381


namespace junk_mail_distribution_l3063_306328

theorem junk_mail_distribution (total_mail : ℕ) (num_blocks : ℕ) 
  (h1 : total_mail = 192) 
  (h2 : num_blocks = 4) :
  total_mail / num_blocks = 48 := by
  sorry

end junk_mail_distribution_l3063_306328


namespace inequality_problem_l3063_306355

theorem inequality_problem (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + a < 0) → 
  a ≤ 9 := by
  sorry

end inequality_problem_l3063_306355


namespace geometry_propositions_l3063_306348

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (intersect : Line → Line → Prop)

-- Theorem statement
theorem geometry_propositions 
  (a b c : Line) (α β : Plane) :
  (∀ (α β : Plane) (c : Line), 
    parallel_plane α β → perpendicular_plane c α → perpendicular_plane c β) ∧
  (∀ (a b c : Line),
    perpendicular a c → perpendicular b c → 
    (parallel a b ∨ skew a b ∨ intersect a b)) :=
by sorry

end geometry_propositions_l3063_306348


namespace intersection_of_A_and_B_l3063_306318

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | 2*x - 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 3/2} := by sorry

end intersection_of_A_and_B_l3063_306318


namespace joan_can_buy_5_apples_l3063_306308

/-- Represents the grocery shopping problem --/
def grocery_problem (total_money : ℕ) (hummus_price : ℕ) (hummus_quantity : ℕ)
  (chicken_price : ℕ) (bacon_price : ℕ) (vegetable_price : ℕ) (apple_price : ℕ) : Prop :=
  let remaining_money := total_money - (hummus_price * hummus_quantity + chicken_price + bacon_price + vegetable_price)
  remaining_money / apple_price = 5

/-- Theorem stating that Joan can buy 5 apples with her remaining money --/
theorem joan_can_buy_5_apples :
  grocery_problem 60 5 2 20 10 10 2 := by
  sorry

#check joan_can_buy_5_apples

end joan_can_buy_5_apples_l3063_306308


namespace sum_of_two_numbers_l3063_306392

theorem sum_of_two_numbers (x y : ℤ) : x = 18 ∧ y = 2 * x - 3 → x + y = 51 := by
  sorry

end sum_of_two_numbers_l3063_306392


namespace intersection_complement_equals_set_l3063_306395

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement_equals_set : S ∩ (U \ T) = {1, 2, 4} := by sorry

end intersection_complement_equals_set_l3063_306395


namespace product_greater_than_sum_l3063_306354

theorem product_greater_than_sum (a b : ℝ) (ha : a > 2) (hb : b > 2) : a * b > a + b := by
  sorry

end product_greater_than_sum_l3063_306354


namespace no_pentagon_decagon_tiling_l3063_306390

/-- The interior angle of a regular pentagon in degrees -/
def pentagon_angle : ℝ := 108

/-- The interior angle of a regular decagon in degrees -/
def decagon_angle : ℝ := 144

/-- The sum of angles at a vertex in a tiling -/
def vertex_angle_sum : ℝ := 360

/-- Theorem stating the impossibility of tiling with regular pentagons and decagons -/
theorem no_pentagon_decagon_tiling : 
  ¬ ∃ (p d : ℕ), p * pentagon_angle + d * decagon_angle = vertex_angle_sum :=
sorry

end no_pentagon_decagon_tiling_l3063_306390


namespace sqrt_2_2801_eq_1_51_square_diff_16_2_16_1_square_diff_less_than_3_01_l3063_306321

-- Define the square function
def square (x : ℝ) : ℝ := x * x

-- Statement 1: √2.2801 = 1.51
theorem sqrt_2_2801_eq_1_51 : Real.sqrt 2.2801 = 1.51 := by sorry

-- Statement 2: 16.2² - 16.1² = 3.23
theorem square_diff_16_2_16_1 : square 16.2 - square 16.1 = 3.23 := by sorry

-- Statement 3: For any x where 0 < x < 15, (x + 0.1)² - x² < 3.01
theorem square_diff_less_than_3_01 (x : ℝ) (h1 : 0 < x) (h2 : x < 15) :
  square (x + 0.1) - square x < 3.01 := by sorry

end sqrt_2_2801_eq_1_51_square_diff_16_2_16_1_square_diff_less_than_3_01_l3063_306321


namespace two_different_color_chips_probability_l3063_306374

/-- The probability of selecting two chips of different colors from a bag with replacement -/
theorem two_different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ)
  (h_blue : blue = 6)
  (h_red : red = 5)
  (h_yellow : yellow = 4) :
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow = 148 / 225 := by
  sorry

end two_different_color_chips_probability_l3063_306374


namespace two_out_of_three_accurate_l3063_306306

/-- The probability of an accurate forecast -/
def p_accurate : ℝ := 0.9

/-- The probability of an inaccurate forecast -/
def p_inaccurate : ℝ := 1 - p_accurate

/-- The probability of exactly 2 out of 3 forecasts being accurate -/
def p_two_accurate : ℝ := 3 * (p_accurate ^ 2 * p_inaccurate)

theorem two_out_of_three_accurate :
  p_two_accurate = 0.243 := by sorry

end two_out_of_three_accurate_l3063_306306


namespace evaluate_expression_l3063_306364

theorem evaluate_expression : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end evaluate_expression_l3063_306364


namespace factor_expression_l3063_306316

theorem factor_expression (x : ℝ) : 63 * x + 28 = 7 * (9 * x + 4) := by
  sorry

end factor_expression_l3063_306316


namespace initial_markup_percentage_l3063_306335

theorem initial_markup_percentage (C : ℝ) (M : ℝ) : 
  C > 0 →
  let S₁ := C * (1 + M)
  let S₂ := S₁ * 1.25
  let S₃ := S₂ * 0.94
  S₃ = C * 1.41 →
  M = 0.2 := by sorry

end initial_markup_percentage_l3063_306335


namespace watch_cost_price_l3063_306361

theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 21/100 →
  gain_percentage = 4/100 →
  price_difference = 140 →
  ∃ (cost_price : ℚ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 560 :=
by sorry

end watch_cost_price_l3063_306361


namespace max_edges_l3063_306386

/-- A square partitioned into convex polygons -/
structure PartitionedSquare where
  n : ℕ  -- number of polygons
  v : ℕ  -- number of vertices
  e : ℕ  -- number of edges

/-- Euler's theorem for partitioned square -/
axiom euler_theorem (ps : PartitionedSquare) : ps.v - ps.e + ps.n = 1

/-- The degree of each vertex is at least 2, except for at most 4 corner vertices -/
axiom vertex_degree (ps : PartitionedSquare) : 2 * ps.e ≥ 3 * ps.v - 4

/-- Theorem: Maximum number of edges in a partitioned square -/
theorem max_edges (ps : PartitionedSquare) : ps.e ≤ 3 * ps.n + 1 := by
  sorry

end max_edges_l3063_306386


namespace fraction_inequality_l3063_306310

theorem fraction_inequality (x : ℝ) (h : x ≠ -5) :
  x / (x + 5) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iic (-5) :=
sorry

end fraction_inequality_l3063_306310


namespace candy_mixture_l3063_306345

theorem candy_mixture :
  ∀ (x y : ℝ),
  x + y = 100 →
  18 * x + 10 * y = 15 * 100 →
  x = 62.5 ∧ y = 37.5 :=
by
  sorry

end candy_mixture_l3063_306345


namespace stone_68_is_10_l3063_306344

/-- The number of stones in the circle -/
def n : ℕ := 15

/-- The length of a full cycle (clockwise + counterclockwise) -/
def cycle_length : ℕ := n + (n - 1)

/-- The stone number corresponding to a given count -/
def stone_number (count : ℕ) : ℕ :=
  let effective_count := count % cycle_length
  if effective_count ≤ n then effective_count else n - (effective_count - n)

theorem stone_68_is_10 : stone_number 68 = 10 := by sorry

end stone_68_is_10_l3063_306344


namespace expression_value_l3063_306320

theorem expression_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c) (h2 : a^2 + b^2 = c^2 + d^2) :
  (a*b + c*d) / (a*d + b*c) = Real.sqrt 3 / 2 := by sorry

end expression_value_l3063_306320


namespace transform_range_transform_uniform_l3063_306304

/-- A uniform random variable in the interval [0,1] -/
def uniform_01 : Type := {x : ℝ // 0 ≤ x ∧ x ≤ 1}

/-- The transformation function -/
def transform (a₁ : uniform_01) : ℝ := a₁.val * 5 - 2

/-- Theorem stating that the transformation maps [0,1] to [-2,3] -/
theorem transform_range :
  ∀ (a₁ : uniform_01), -2 ≤ transform a₁ ∧ transform a₁ ≤ 3 := by
  sorry

/-- Theorem stating that the transformation preserves uniformity -/
theorem transform_uniform :
  uniform_01 → {x : ℝ // -2 ≤ x ∧ x ≤ 3} := by
  sorry

end transform_range_transform_uniform_l3063_306304


namespace arithmetic_sequence_60th_term_l3063_306339

/-- Given an arithmetic sequence with first term 7 and fifteenth term 35,
    prove that the sixtieth term is 125. -/
theorem arithmetic_sequence_60th_term
  (a : ℕ → ℤ)  -- The arithmetic sequence
  (h1 : a 1 = 7)  -- First term is 7
  (h2 : a 15 = 35)  -- Fifteenth term is 35
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Arithmetic sequence property
  : a 60 = 125 := by
  sorry

end arithmetic_sequence_60th_term_l3063_306339


namespace sum_of_four_numbers_l3063_306367

theorem sum_of_four_numbers : 8765 + 7658 + 6587 + 5876 = 28868 := by
  sorry

end sum_of_four_numbers_l3063_306367


namespace total_juice_boxes_for_school_year_l3063_306322

-- Define the structure for a child
structure Child where
  name : String
  juiceBoxesPerWeek : ℕ
  schoolWeeks : ℕ

-- Define Peyton's children
def john : Child := { name := "John", juiceBoxesPerWeek := 10, schoolWeeks := 16 }
def samantha : Child := { name := "Samantha", juiceBoxesPerWeek := 5, schoolWeeks := 14 }
def heather : Child := { name := "Heather", juiceBoxesPerWeek := 11, schoolWeeks := 15 }

def children : List Child := [john, samantha, heather]

-- Function to calculate total juice boxes for a child
def totalJuiceBoxes (child : Child) : ℕ :=
  child.juiceBoxesPerWeek * child.schoolWeeks

-- Theorem to prove
theorem total_juice_boxes_for_school_year :
  (children.map totalJuiceBoxes).sum = 395 := by
  sorry

end total_juice_boxes_for_school_year_l3063_306322


namespace fraction_equation_solution_l3063_306383

theorem fraction_equation_solution (a b : ℝ) (h1 : a ≠ b) (h2 : b = 1) 
  (h3 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4) : 
  a / b = (17 + Real.sqrt 269) / 10 := by
sorry

end fraction_equation_solution_l3063_306383


namespace smallest_perfect_square_divisible_by_5_and_6_l3063_306362

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define the property we want to prove
def is_smallest_divisible_by_5_and_6 (n : ℕ) : Prop :=
  is_perfect_square n ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧
  ∀ m : ℕ, m < n → ¬(is_perfect_square m ∧ m % 5 = 0 ∧ m % 6 = 0)

-- State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 :
  is_smallest_divisible_by_5_and_6 900 :=
sorry

end smallest_perfect_square_divisible_by_5_and_6_l3063_306362
