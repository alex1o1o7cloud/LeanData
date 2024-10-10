import Mathlib

namespace abs_sum_inequality_range_l2327_232737

theorem abs_sum_inequality_range :
  {x : ℝ | |x + 1| + |x| < 2} = Set.Ioo (-3/2 : ℝ) (1/2 : ℝ) := by
  sorry

end abs_sum_inequality_range_l2327_232737


namespace dog_food_weight_l2327_232738

/-- Proves that given the conditions, each sack of dog food weighs 50 kilograms -/
theorem dog_food_weight 
  (num_dogs : ℕ) 
  (meals_per_day : ℕ) 
  (food_per_meal : ℕ) 
  (num_sacks : ℕ) 
  (days_lasting : ℕ) 
  (h1 : num_dogs = 4)
  (h2 : meals_per_day = 2)
  (h3 : food_per_meal = 250)
  (h4 : num_sacks = 2)
  (h5 : days_lasting = 50) :
  (num_dogs * meals_per_day * food_per_meal * days_lasting) / (1000 * num_sacks) = 50 := by
  sorry

#check dog_food_weight

end dog_food_weight_l2327_232738


namespace parking_arrangements_l2327_232780

def parking_spaces : ℕ := 7
def car_models : ℕ := 3
def consecutive_empty : ℕ := 3

theorem parking_arrangements :
  (car_models.factorial) *
  (parking_spaces - car_models).choose 2 *
  ((parking_spaces - car_models - consecutive_empty + 1).factorial) = 72 := by
  sorry

end parking_arrangements_l2327_232780


namespace sum_of_reciprocals_is_one_tenth_l2327_232702

/-- A line passing through (10, 0) intersecting y = x^2 -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (10, 0) -/
  line_eq : ∀ x y : ℝ, y = k * (x - 10)
  /-- The line intersects y = x^2 at two distinct points -/
  intersects_parabola : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = k * (x₁ - 10) ∧ x₂^2 = k * (x₂ - 10)

/-- The sum of reciprocals of intersection x-coordinates is 1/10 -/
theorem sum_of_reciprocals_is_one_tenth (L : IntersectingLine) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 = L.k * (x₁ - 10) ∧ 
    x₂^2 = L.k * (x₂ - 10) ∧
    1 / x₁ + 1 / x₂ = 1 / 10 := by
  sorry

end sum_of_reciprocals_is_one_tenth_l2327_232702


namespace inequality_solution_set_l2327_232761

theorem inequality_solution_set (x : ℝ) :
  (4 * x^4 + x^2 + 4*x - 5 * x^2 * |x + 2| + 4 ≥ 0) ↔ 
  (x ≤ -1 ∨ ((1 - Real.sqrt 33) / 8 ≤ x ∧ x ≤ (1 + Real.sqrt 33) / 8) ∨ x ≥ 2) := by
  sorry

end inequality_solution_set_l2327_232761


namespace not_always_true_converse_l2327_232798

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the lines and planes
variable (a b c : Line) (α β : Plane)

-- State the theorem
theorem not_always_true_converse
  (h1 : contained_in b α)
  (h2 : ¬ contained_in c α) :
  ¬ (∀ (α β : Plane), plane_perpendicular α β → perpendicular b β) :=
sorry

end not_always_true_converse_l2327_232798


namespace koala_fiber_consumption_l2327_232752

/-- Given a koala that absorbs 30% of the fiber it eats and absorbed 15 ounces of fiber in one day,
    prove that the total amount of fiber eaten is 50 ounces. -/
theorem koala_fiber_consumption (absorption_rate : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) : 
  absorption_rate = 0.30 →
  absorbed_fiber = 15 →
  total_fiber * absorption_rate = absorbed_fiber →
  total_fiber = 50 := by
  sorry


end koala_fiber_consumption_l2327_232752


namespace total_sales_l2327_232782

def candy_bar_sales (max_sales seth_sales emma_sales : ℕ) : Prop :=
  (max_sales = 24) ∧
  (seth_sales = 3 * max_sales + 6) ∧
  (emma_sales = seth_sales / 2 + 5)

theorem total_sales (max_sales seth_sales emma_sales : ℕ) :
  candy_bar_sales max_sales seth_sales emma_sales →
  seth_sales + emma_sales = 122 := by
  sorry

end total_sales_l2327_232782


namespace sean_train_track_length_l2327_232729

theorem sean_train_track_length 
  (ruth_piece_length : ℕ) 
  (total_length : ℕ) 
  (ruth_pieces : ℕ) 
  (sean_pieces : ℕ) :
  ruth_piece_length = 18 →
  total_length = 72 →
  ruth_pieces * ruth_piece_length = total_length →
  sean_pieces = ruth_pieces →
  sean_pieces * (total_length / sean_pieces) = total_length →
  total_length / sean_pieces = 18 :=
by
  sorry

#check sean_train_track_length

end sean_train_track_length_l2327_232729


namespace arithmetic_sequence_sum_mod_17_l2327_232787

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_mod_17 :
  arithmetic_sequence_sum 4 6 100 % 17 = 0 := by
  sorry

end arithmetic_sequence_sum_mod_17_l2327_232787


namespace prism_volume_l2327_232724

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches is 30√3 cubic inches. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 10) 
  (h2 : b * c = 15) 
  (h3 : c * a = 18) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end prism_volume_l2327_232724


namespace larger_number_proof_l2327_232705

/-- Given two positive integers with specific h.c.f. and l.c.m., prove the larger number is 391 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 23)
  (lcm : Nat.lcm a b = 23 * 13 * 17) :
  max a b = 391 := by
  sorry

end larger_number_proof_l2327_232705


namespace mac_loss_is_three_dollars_l2327_232755

-- Define the values of coins in cents
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

-- Define the number of coins in each trade
def dimes_per_trade : ℕ := 3
def nickels_per_trade : ℕ := 7

-- Define the number of trades
def dime_trades : ℕ := 20
def nickel_trades : ℕ := 20

-- Calculate the loss per trade in cents
def dime_trade_loss : ℕ := dimes_per_trade * dime_value - quarter_value
def nickel_trade_loss : ℕ := nickels_per_trade * nickel_value - quarter_value

-- Calculate the total loss in cents
def total_loss_cents : ℕ := dime_trade_loss * dime_trades + nickel_trade_loss * nickel_trades

-- Convert cents to dollars
def cents_to_dollars (cents : ℕ) : ℚ := (cents : ℚ) / 100

-- Theorem: Mac's total loss is $3.00
theorem mac_loss_is_three_dollars :
  cents_to_dollars total_loss_cents = 3 := by
  sorry

end mac_loss_is_three_dollars_l2327_232755


namespace existence_of_close_multiple_l2327_232734

theorem existence_of_close_multiple (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * a - m| ≤ 1 / n := by
  sorry

end existence_of_close_multiple_l2327_232734


namespace decagon_triangles_l2327_232794

theorem decagon_triangles : 
  let n : ℕ := 10  -- number of vertices in a regular decagon
  let k : ℕ := 3   -- number of vertices needed to form a triangle
  Nat.choose n k = 120 := by
sorry

end decagon_triangles_l2327_232794


namespace circle_inequality_l2327_232701

/-- Given three circles with centers P, Q, R and radii p, q, r respectively,
    where p > q > r, prove that p + q + r ≠ dist P Q + dist Q R -/
theorem circle_inequality (P Q R : EuclideanSpace ℝ (Fin 2))
    (p q r : ℝ) (hp : p > q) (hq : q > r) :
    p + q + r ≠ dist P Q + dist Q R := by
  sorry

end circle_inequality_l2327_232701


namespace square_of_complex_number_l2327_232786

theorem square_of_complex_number (z : ℂ) (i : ℂ) :
  z = 5 + 2 * i →
  i^2 = -1 →
  z^2 = 21 + 20 * i :=
by sorry

end square_of_complex_number_l2327_232786


namespace womens_average_age_l2327_232715

/-- Represents the problem of finding the average age of two women -/
theorem womens_average_age 
  (n : ℕ) 
  (initial_total_age : ℝ) 
  (age_increase : ℝ) 
  (man1_age man2_age : ℝ) :
  n = 10 →
  age_increase = 6 →
  man1_age = 18 →
  man2_age = 22 →
  (initial_total_age / n + age_increase) * n = initial_total_age - man1_age - man2_age + 2 * ((initial_total_age / n + age_increase) * n - initial_total_age + man1_age + man2_age) / 2 →
  ((initial_total_age / n + age_increase) * n - initial_total_age + man1_age + man2_age) / 2 = 50 :=
by sorry

end womens_average_age_l2327_232715


namespace apple_lemon_equivalence_l2327_232743

/-- Represents the value of fruits in terms of a common unit -/
structure FruitValue where
  apple : ℚ
  lemon : ℚ

/-- Given that 3/4 of 14 apples are worth 9 lemons, 
    prove that 5/7 of 7 apples are worth 30/7 lemons -/
theorem apple_lemon_equivalence (v : FruitValue) 
  (h : (3/4 : ℚ) * 14 * v.apple = 9 * v.lemon) :
  (5/7 : ℚ) * 7 * v.apple = (30/7 : ℚ) * v.lemon := by
  sorry

#check apple_lemon_equivalence

end apple_lemon_equivalence_l2327_232743


namespace miriam_homework_time_l2327_232797

theorem miriam_homework_time (laundry_time bathroom_time room_time total_time : ℕ) 
  (h1 : laundry_time = 30)
  (h2 : bathroom_time = 15)
  (h3 : room_time = 35)
  (h4 : total_time = 120) :
  total_time - (laundry_time + bathroom_time + room_time) = 40 := by
  sorry

end miriam_homework_time_l2327_232797


namespace sin_sum_to_product_l2327_232790

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end sin_sum_to_product_l2327_232790


namespace star_specific_value_l2327_232768

/-- Custom binary operation star -/
def star (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b + 2 * b^2

/-- Theorem: Given the custom operation star and specific values for a and b,
    prove that the result equals 113 -/
theorem star_specific_value : star 3 5 = 113 := by
  sorry

end star_specific_value_l2327_232768


namespace mechanic_parts_cost_l2327_232777

theorem mechanic_parts_cost (hourly_rate : ℕ) (daily_hours : ℕ) (work_days : ℕ) (total_cost : ℕ) : 
  hourly_rate = 60 →
  daily_hours = 8 →
  work_days = 14 →
  total_cost = 9220 →
  total_cost - (hourly_rate * daily_hours * work_days) = 2500 := by
sorry

end mechanic_parts_cost_l2327_232777


namespace election_winning_probability_l2327_232714

/-- Represents the number of voters in the election -/
def total_voters : ℕ := 2019

/-- Represents the number of initial votes for the leading candidate -/
def initial_leading_votes : ℕ := 2

/-- Represents the number of initial votes for the trailing candidate -/
def initial_trailing_votes : ℕ := 1

/-- Represents the number of undecided voters -/
def undecided_voters : ℕ := total_voters - initial_leading_votes - initial_trailing_votes

/-- Calculates the probability of a candidate winning given their initial vote advantage -/
def winning_probability (initial_advantage : ℕ) : ℚ :=
  (1513 : ℚ) / 2017

/-- Theorem stating the probability of the leading candidate winning the election -/
theorem election_winning_probability :
  winning_probability (initial_leading_votes - initial_trailing_votes) = 1513 / 2017 := by
  sorry

end election_winning_probability_l2327_232714


namespace color_natural_numbers_l2327_232713

theorem color_natural_numbers :
  ∃ (f : ℕ → Fin 2009),
    (∀ c : Fin 2009, Set.Infinite {n : ℕ | f n = c}) ∧
    (∀ x y z : ℕ, f x ≠ f y → f y ≠ f z → f x ≠ f z → x * y ≠ z) :=
by sorry

end color_natural_numbers_l2327_232713


namespace unique_values_a_k_l2327_232709

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, 3, k}
def B (a : ℕ) : Set ℕ := {4, 7, a^4, a^2 + 3*a}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- Theorem statement
theorem unique_values_a_k :
  ∃! (a k : ℕ), 
    a > 0 ∧ 
    (∀ x ∈ A k, ∃ y ∈ B a, f x = y) ∧ 
    (∀ y ∈ B a, ∃ x ∈ A k, f x = y) ∧
    a = 2 ∧ k = 5 := by
  sorry

end unique_values_a_k_l2327_232709


namespace abs_x_minus_two_plus_three_min_l2327_232772

theorem abs_x_minus_two_plus_three_min (x : ℝ) : 
  ∃ (min : ℝ), (∀ x, |x - 2| + 3 ≥ min) ∧ (∃ x, |x - 2| + 3 = min) := by
  sorry

end abs_x_minus_two_plus_three_min_l2327_232772


namespace finite_decimal_fraction_l2327_232799

theorem finite_decimal_fraction (n : ℕ) : 
  (∃ (k m : ℕ), n * (2 * n - 1) = 2^k * 5^m) ↔ n = 1 :=
sorry

end finite_decimal_fraction_l2327_232799


namespace salmon_trip_count_l2327_232733

/-- The number of male salmon that returned to their rivers -/
def male_salmon : ℕ := 712261

/-- The number of female salmon that returned to their rivers -/
def female_salmon : ℕ := 259378

/-- The total number of salmon that made the trip -/
def total_salmon : ℕ := male_salmon + female_salmon

theorem salmon_trip_count : total_salmon = 971639 := by
  sorry

end salmon_trip_count_l2327_232733


namespace min_value_problem_1_l2327_232747

theorem min_value_problem_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / (1 + y) ≥ 9 / 2 := by
  sorry

end min_value_problem_1_l2327_232747


namespace sum_of_fractions_equals_three_l2327_232730

theorem sum_of_fractions_equals_three (a b c x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c := by sorry

end sum_of_fractions_equals_three_l2327_232730


namespace sector_central_angle_central_angle_is_two_l2327_232720

theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 1) :
  ∃ (r : ℝ), r > 0 ∧ area = 1/2 * r * arc_length ∧ arc_length = 2 * r := by
  sorry

theorem central_angle_is_two (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 1) :
  let r := (2 * area) / arc_length
  arc_length / r = 2 := by
  sorry

end sector_central_angle_central_angle_is_two_l2327_232720


namespace cubic_difference_l2327_232735

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7)
  (h2 : a^2 + b^2 = 65)
  (h3 : a + b = 6) :
  a^3 - b^3 = 432.25 := by
sorry

end cubic_difference_l2327_232735


namespace cuboid_height_calculation_l2327_232775

/-- The surface area of a cuboid given its length, width, and height. -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The height of a cuboid with surface area 2400 cm², length 15 cm, and width 10 cm is 42 cm. -/
theorem cuboid_height_calculation (sa l w h : ℝ) 
  (h_sa : sa = 2400)
  (h_l : l = 15)
  (h_w : w = 10)
  (h_surface_area : surfaceArea l w h = sa) : h = 42 := by
  sorry

#check cuboid_height_calculation

end cuboid_height_calculation_l2327_232775


namespace subset_existence_l2327_232763

theorem subset_existence (X : Finset ℕ) (hX : X.card = 20) :
  ∀ (f : Finset ℕ → ℕ),
  (∀ S : Finset ℕ, S ⊆ X → S.card = 9 → f S ∈ X) →
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Y.card = 10 ∧
  ∀ k ∈ Y, f (Y \ {k}) ≠ k :=
by sorry

end subset_existence_l2327_232763


namespace product_of_integers_with_given_sum_and_difference_l2327_232741

theorem product_of_integers_with_given_sum_and_difference :
  ∀ x y : ℕ+, 
    (x : ℤ) + (y : ℤ) = 72 → 
    (x : ℤ) - (y : ℤ) = 18 → 
    (x : ℤ) * (y : ℤ) = 1215 := by
  sorry

end product_of_integers_with_given_sum_and_difference_l2327_232741


namespace x_needs_seven_days_l2327_232759

/-- The number of days X needs to finish the remaining work after Y leaves -/
def days_for_x_to_finish (x_days y_days y_worked_days : ℕ) : ℚ :=
  let x_rate : ℚ := 1 / x_days
  let y_rate : ℚ := 1 / y_days
  let work_done_by_y : ℚ := y_rate * y_worked_days
  let remaining_work : ℚ := 1 - work_done_by_y
  remaining_work / x_rate

/-- Theorem stating that X needs 7 days to finish the remaining work -/
theorem x_needs_seven_days :
  days_for_x_to_finish 21 15 10 = 7 := by
  sorry

end x_needs_seven_days_l2327_232759


namespace stamp_problem_solution_l2327_232773

/-- Represents the number of stamps of each type -/
structure StampCounts where
  twopenny : ℕ
  penny : ℕ
  twohalfpenny : ℕ

/-- Calculates the total value of stamps in pence -/
def total_value (s : StampCounts) : ℕ :=
  2 * s.twopenny + s.penny + (5 * s.twohalfpenny) / 2

/-- Checks if the number of penny stamps is six times the number of twopenny stamps -/
def penny_constraint (s : StampCounts) : Prop :=
  s.penny = 6 * s.twopenny

/-- The main theorem stating the unique solution to the stamp problem -/
theorem stamp_problem_solution :
  ∃! s : StampCounts,
    total_value s = 60 ∧
    penny_constraint s ∧
    s.twopenny = 5 ∧
    s.penny = 30 ∧
    s.twohalfpenny = 8 := by
  sorry

end stamp_problem_solution_l2327_232773


namespace polynomial_equality_l2327_232769

theorem polynomial_equality (g : ℝ → ℝ) : 
  (∀ x, 5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5) →
  (∀ x, g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3) :=
by sorry

end polynomial_equality_l2327_232769


namespace max_y_over_x_l2327_232796

/-- Given that x and y satisfy (x-2)^2 + y^2 = 1, the maximum value of y/x is √3/3 -/
theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 3 ∧ ∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 1 → |y' / x'| ≤ max := by
  sorry

end max_y_over_x_l2327_232796


namespace nail_fraction_sum_l2327_232750

theorem nail_fraction_sum : 
  let size_2d : ℚ := 1/6
  let size_3d : ℚ := 2/15
  let size_4d : ℚ := 3/20
  let size_5d : ℚ := 1/10
  let size_6d : ℚ := 1/4
  let size_7d : ℚ := 1/12
  let size_8d : ℚ := 1/8
  let size_9d : ℚ := 1/30
  size_2d + size_3d + size_5d + size_8d = 21/40 := by
  sorry

end nail_fraction_sum_l2327_232750


namespace foci_of_hyperbola_l2327_232740

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(0, -Real.sqrt 29), (0, Real.sqrt 29)}

/-- Theorem: The given coordinates are the foci of the hyperbola -/
theorem foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates :=
sorry

end foci_of_hyperbola_l2327_232740


namespace geometric_sequence_property_l2327_232781

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∀ n : ℕ, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 16 →
  a 2 + a 4 = 4 := by
  sorry

end geometric_sequence_property_l2327_232781


namespace leftover_value_is_five_fifty_l2327_232718

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of leftover coins in dollars --/
def leftoverValue (rollSize : RollSize) (james : CoinCount) (lindsay : CoinCount) : ℚ :=
  let totalQuarters := james.quarters + lindsay.quarters
  let totalDimes := james.dimes + lindsay.dimes
  let leftoverQuarters := totalQuarters % rollSize.quarters
  let leftoverDimes := totalDimes % rollSize.dimes
  (leftoverQuarters : ℚ) * (1 / 4) + (leftoverDimes : ℚ) * (1 / 10)

theorem leftover_value_is_five_fifty :
  let rollSize : RollSize := { quarters := 40, dimes := 50 }
  let james : CoinCount := { quarters := 83, dimes := 159 }
  let lindsay : CoinCount := { quarters := 129, dimes := 266 }
  leftoverValue rollSize james lindsay = 11/2 := by
  sorry

end leftover_value_is_five_fifty_l2327_232718


namespace mixture_composition_l2327_232753

theorem mixture_composition (alcohol_water_ratio : ℚ) (alcohol_fraction : ℚ) :
  alcohol_water_ratio = 1/2 →
  alcohol_fraction = 1/7 →
  1 - alcohol_fraction = 2/7 :=
by
  sorry

end mixture_composition_l2327_232753


namespace maci_pen_cost_l2327_232731

/-- The cost of Maci's pens given the number and prices of blue and red pens. -/
def cost_of_pens (blue_pens : ℕ) (red_pens : ℕ) (blue_pen_cost : ℚ) : ℚ :=
  let red_pen_cost := 2 * blue_pen_cost
  blue_pens * blue_pen_cost + red_pens * red_pen_cost

/-- Theorem stating that Maci pays $4.00 for her pens. -/
theorem maci_pen_cost : cost_of_pens 10 15 (10 / 100) = 4 := by
  sorry

#eval cost_of_pens 10 15 (10 / 100)

end maci_pen_cost_l2327_232731


namespace polynomial_product_l2327_232722

theorem polynomial_product (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
sorry

end polynomial_product_l2327_232722


namespace compound_mass_percentage_sum_l2327_232767

/-- Given a compound with two parts, where one part's mass percentage is known,
    prove that the sum of both parts' mass percentages is 100%. -/
theorem compound_mass_percentage_sum (part1_percentage : ℝ) :
  part1_percentage = 80.12 →
  100 - part1_percentage = 19.88 := by
  sorry

end compound_mass_percentage_sum_l2327_232767


namespace mixture_cost_ratio_l2327_232783

/-- Given the conditions of the mixture problem, prove that the ratio of nut cost to raisin cost is 3:1 -/
theorem mixture_cost_ratio (R N : ℝ) (h1 : R > 0) (h2 : N > 0) : 
  3 * R = 0.25 * (3 * R + 3 * N) → N / R = 3 := by
  sorry

end mixture_cost_ratio_l2327_232783


namespace total_birds_l2327_232758

def geese : ℕ := 58
def ducks : ℕ := 37

theorem total_birds : geese + ducks = 95 := by
  sorry

end total_birds_l2327_232758


namespace marquita_garden_count_l2327_232716

/-- The number of gardens Mancino is tending -/
def mancino_gardens : ℕ := 3

/-- The length of each of Mancino's gardens in feet -/
def mancino_garden_length : ℕ := 16

/-- The width of each of Mancino's gardens in feet -/
def mancino_garden_width : ℕ := 5

/-- The length of each of Marquita's gardens in feet -/
def marquita_garden_length : ℕ := 8

/-- The width of each of Marquita's gardens in feet -/
def marquita_garden_width : ℕ := 4

/-- The total area of all gardens combined in square feet -/
def total_garden_area : ℕ := 304

/-- Theorem stating the number of gardens Marquita is tilling -/
theorem marquita_garden_count : 
  ∃ n : ℕ, n * (marquita_garden_length * marquita_garden_width) = 
    total_garden_area - mancino_gardens * (mancino_garden_length * mancino_garden_width) ∧ 
    n = 2 := by
  sorry

end marquita_garden_count_l2327_232716


namespace change_percentage_l2327_232776

-- Define the prices of the items
def price1 : ℚ := 15.50
def price2 : ℚ := 3.25
def price3 : ℚ := 6.75

-- Define the amount paid
def amountPaid : ℚ := 50.00

-- Define the total price of items
def totalPrice : ℚ := price1 + price2 + price3

-- Define the change received
def change : ℚ := amountPaid - totalPrice

-- Define the percentage of change
def percentageChange : ℚ := (change / amountPaid) * 100

-- Theorem statement
theorem change_percentage : percentageChange = 49 := by
  sorry

end change_percentage_l2327_232776


namespace function_properties_l2327_232736

def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

theorem function_properties (b : ℝ) :
  f b 0 = f b 4 →
  (b = 4 ∧
   (∀ x, f b x = 0 ↔ x = 1 ∨ x = 3) ∧
   (∀ x, f b x < 0 ↔ 1 < x ∧ x < 3) ∧
   (∀ x ∈ Set.Icc 0 3, f b x ≥ -1) ∧
   (∃ x ∈ Set.Icc 0 3, f b x = -1) ∧
   (∀ x ∈ Set.Icc 0 3, f b x ≤ 3) ∧
   (∃ x ∈ Set.Icc 0 3, f b x = 3)) :=
by sorry

end function_properties_l2327_232736


namespace like_terms_exponent_l2327_232710

/-- 
Given two terms -3x^(2m)y^3 and 2x^4y^n are like terms,
prove that m^n = 8
-/
theorem like_terms_exponent (m n : ℕ) : 
  (∃ (x y : ℝ), -3 * x^(2*m) * y^3 = 2 * x^4 * y^n) → m^n = 8 := by
sorry

end like_terms_exponent_l2327_232710


namespace infinite_primes_4k_plus_3_l2327_232739

theorem infinite_primes_4k_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 3) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4*m + 3) ∧ q ∉ S :=
by sorry

end infinite_primes_4k_plus_3_l2327_232739


namespace exam_average_problem_l2327_232779

theorem exam_average_problem (total_students : ℕ) (high_score_students : ℕ) (high_score : ℝ) (total_average : ℝ) :
  total_students = 25 →
  high_score_students = 10 →
  high_score = 90 →
  total_average = 84 →
  ∃ (low_score_students : ℕ) (low_score : ℝ),
    low_score_students + high_score_students = total_students ∧
    low_score = 80 ∧
    (low_score_students * low_score + high_score_students * high_score) / total_students = total_average ∧
    low_score_students = 15 :=
by sorry

end exam_average_problem_l2327_232779


namespace science_fair_ratio_l2327_232784

/-- Represents the number of adults and children at the science fair -/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Calculates the total fee collected given an attendance -/
def totalFee (a : Attendance) : ℕ := 30 * a.adults + 15 * a.children

/-- Calculates the ratio of adults to children -/
def ratio (a : Attendance) : ℚ := a.adults / a.children

theorem science_fair_ratio : 
  ∃ (a : Attendance), 
    a.adults ≥ 1 ∧ 
    a.children ≥ 1 ∧ 
    totalFee a = 2250 ∧ 
    ∀ (b : Attendance), 
      b.adults ≥ 1 → 
      b.children ≥ 1 → 
      totalFee b = 2250 → 
      |ratio a - 2| ≤ |ratio b - 2| := by
  sorry

end science_fair_ratio_l2327_232784


namespace imaginary_part_of_i_squared_is_zero_l2327_232764

theorem imaginary_part_of_i_squared_is_zero :
  Complex.im (Complex.I ^ 2) = 0 := by
  sorry

end imaginary_part_of_i_squared_is_zero_l2327_232764


namespace simplify_product_of_radicals_l2327_232748

theorem simplify_product_of_radicals (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (54 * x) * Real.sqrt (20 * x) * Real.sqrt (14 * x) = 12 * Real.sqrt (105 * x) := by
  sorry

end simplify_product_of_radicals_l2327_232748


namespace inscribed_square_side_length_l2327_232765

/-- A right triangle with sides 12, 16, and 20 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  is_right : de^2 + ef^2 = df^2
  de_eq : de = 12
  ef_eq : ef = 16
  df_eq : df = 20

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.df
  on_other_sides : side_length ≤ t.de ∧ side_length ≤ t.ef

/-- The side length of the inscribed square is 80/9 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 80 / 9 := by
  sorry

end inscribed_square_side_length_l2327_232765


namespace pats_picnic_dessert_l2327_232725

/-- Pat's picnic dessert problem -/
theorem pats_picnic_dessert (cookies : ℕ) (candy : ℕ) (family_size : ℕ) (dessert_per_person : ℕ) 
  (h1 : cookies = 42)
  (h2 : candy = 63)
  (h3 : family_size = 7)
  (h4 : dessert_per_person = 18) :
  family_size * dessert_per_person - (cookies + candy) = 21 := by
  sorry

end pats_picnic_dessert_l2327_232725


namespace min_draws_for_even_product_l2327_232703

theorem min_draws_for_even_product (S : Finset ℕ) : 
  S = Finset.range 16 →
  (∃ n : ℕ, n ∈ S ∧ Even n) →
  (∀ T ⊆ S, T.card = 9 → ∃ m ∈ T, Even m) ∧
  (∃ U ⊆ S, U.card = 8 ∧ ∀ k ∈ U, ¬Even k) :=
by sorry

end min_draws_for_even_product_l2327_232703


namespace binomial_20_19_l2327_232700

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_20_19_l2327_232700


namespace sequence_properties_l2327_232788

/-- Sequence type representing our 0-1 sequence --/
def Sequence := ℕ → Bool

/-- Generate the nth term of the sequence --/
def generateTerm (n : ℕ) : Sequence := sorry

/-- Check if a sequence is periodic --/
def isPeriodic (s : Sequence) : Prop := sorry

/-- Get the nth digit of the sequence --/
def nthDigit (s : Sequence) (n : ℕ) : Bool := sorry

/-- Get the position of the nth occurrence of a digit --/
def nthOccurrence (s : Sequence) (digit : Bool) (n : ℕ) : ℕ := sorry

theorem sequence_properties (s : Sequence) :
  (s = generateTerm 0) →
  (¬ isPeriodic s) ∧
  (nthDigit s 1000 = true) ∧
  (nthOccurrence s true 10000 = 21328) ∧
  (∀ n : ℕ, nthOccurrence s true n = ⌊(2 + Real.sqrt 2) * n⌋) ∧
  (∀ n : ℕ, nthOccurrence s false n = ⌊Real.sqrt 2 * n⌋) := by
  sorry

end sequence_properties_l2327_232788


namespace expenditure_ratio_l2327_232732

-- Define the monthly incomes and savings
def income_b : ℚ := 7200
def income_ratio : ℚ := 5 / 6
def savings_a : ℚ := 1800
def savings_b : ℚ := 1600

-- Define the monthly incomes
def income_a : ℚ := income_ratio * income_b

-- Define the monthly expenditures
def expenditure_a : ℚ := income_a - savings_a
def expenditure_b : ℚ := income_b - savings_b

-- Theorem to prove
theorem expenditure_ratio :
  expenditure_a / expenditure_b = 3 / 4 :=
sorry

end expenditure_ratio_l2327_232732


namespace macaron_distribution_theorem_l2327_232751

/-- The number of kids who receive macarons given the conditions of macaron production and distribution -/
def kids_receiving_macarons (mitch_total : ℕ) (mitch_burnt : ℕ) (joshua_extra : ℕ) 
  (joshua_undercooked : ℕ) (renz_burnt : ℕ) (leah_total : ℕ) (leah_undercooked : ℕ) 
  (first_kids : ℕ) (first_kids_macarons : ℕ) (remaining_kids_macarons : ℕ) : ℕ :=
  let miles_total := 2 * (mitch_total + joshua_extra)
  let renz_total := (3 * miles_total) / 4 - 1
  let total_good_macarons := (mitch_total - mitch_burnt) + 
    (mitch_total + joshua_extra - joshua_undercooked) + 
    miles_total + (renz_total - renz_burnt) + 
    (leah_total - leah_undercooked)
  let remaining_macarons := total_good_macarons - (first_kids * first_kids_macarons)
  first_kids + (remaining_macarons / remaining_kids_macarons)

theorem macaron_distribution_theorem : 
  kids_receiving_macarons 20 2 6 3 4 35 5 10 3 2 = 73 := by
  sorry

end macaron_distribution_theorem_l2327_232751


namespace train_station_wheels_l2327_232795

theorem train_station_wheels :
  let num_trains : ℕ := 6
  let carriages_per_train : ℕ := 5
  let wheel_rows_per_carriage : ℕ := 4
  let wheels_per_row : ℕ := 6
  
  num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row = 720 :=
by
  sorry

end train_station_wheels_l2327_232795


namespace hyperbola_foci_distance_l2327_232791

theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 25 - y^2 / 4 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4 * 29 :=
by sorry

end hyperbola_foci_distance_l2327_232791


namespace factor_quadratic_l2327_232792

theorem factor_quadratic (m : ℤ) : 
  let s : ℤ := 5
  m^2 - s*m - 24 = (m - 8) * (m + 3) :=
by sorry

end factor_quadratic_l2327_232792


namespace power_sum_cosine_l2327_232793

theorem power_sum_cosine (θ : Real) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end power_sum_cosine_l2327_232793


namespace tangent_line_to_circle_l2327_232707

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  (l.a * x₀ + l.b * y₀ + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

theorem tangent_line_to_circle (c : Circle) (l : Line) (p : ℝ × ℝ) :
  c.center = (0, 0) ∧ c.radius = 5 ∧
  l.a = 3 ∧ l.b = 4 ∧ l.c = -25 ∧
  p = (3, 4) →
  is_tangent l c ∧ l.contains p :=
sorry

end tangent_line_to_circle_l2327_232707


namespace item_costs_l2327_232778

/-- The cost of items in yuan -/
structure ItemCosts where
  tableLamp : ℕ
  electricFan : ℕ
  bicycle : ℕ

/-- Theorem stating the total cost of all items and the cost of lamp and fan -/
theorem item_costs (c : ItemCosts) 
  (h1 : c.tableLamp = 86)
  (h2 : c.electricFan = 185)
  (h3 : c.bicycle = 445) :
  (c.tableLamp + c.electricFan + c.bicycle = 716) ∧
  (c.tableLamp + c.electricFan = 271) := by
  sorry

#check item_costs

end item_costs_l2327_232778


namespace fence_perimeter_l2327_232742

/-- The number of posts used to enclose the garden -/
def num_posts : ℕ := 36

/-- The width of each post in inches -/
def post_width_inches : ℕ := 3

/-- The space between adjacent posts in feet -/
def post_spacing_feet : ℕ := 4

/-- Conversion factor from inches to feet -/
def inches_to_feet : ℚ := 1 / 12

/-- The width of each post in feet -/
def post_width_feet : ℚ := post_width_inches * inches_to_feet

/-- The number of posts on each side of the square garden -/
def posts_per_side : ℕ := num_posts / 4 + 1

/-- The number of spaces between posts on each side -/
def spaces_per_side : ℕ := posts_per_side - 1

/-- The length of one side of the square garden in feet -/
def side_length : ℚ := posts_per_side * post_width_feet + spaces_per_side * post_spacing_feet

/-- The outer perimeter of the fence surrounding the square garden -/
def outer_perimeter : ℚ := 4 * side_length

/-- Theorem stating that the outer perimeter of the fence is 137 feet -/
theorem fence_perimeter : outer_perimeter = 137 := by
  sorry

end fence_perimeter_l2327_232742


namespace perpendicular_line_x_intercept_l2327_232766

/-- Given a line L1 defined by 4x + 5y = 15, prove that the x-intercept of the line L2
    that is perpendicular to L1 and has a y-intercept of -3 is 12/5. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 5 * y = 15
  let m1 : ℝ := -4 / 5  -- slope of L1
  let m2 : ℝ := 5 / 4   -- slope of L2 (perpendicular to L1)
  let b2 : ℝ := -3      -- y-intercept of L2
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x + b2
  let x_intercept : ℝ := 12 / 5
  (∀ x y, L2 x y → y = 0 → x = x_intercept) :=
by sorry

end perpendicular_line_x_intercept_l2327_232766


namespace units_digit_of_fraction_l2327_232770

def numerator : ℕ := 15 * 16 * 17 * 18 * 19 * 20
def denominator : ℕ := 500

theorem units_digit_of_fraction : 
  (numerator / denominator) % 10 = 2 := by sorry

end units_digit_of_fraction_l2327_232770


namespace abs_x_minus_two_integral_l2327_232754

theorem abs_x_minus_two_integral : ∫ x in (0)..(4), |x - 2| = 4 := by sorry

end abs_x_minus_two_integral_l2327_232754


namespace penumbra_ring_area_l2327_232744

/-- Given the ratio of radii of umbra to penumbra and the radius of the umbra,
    calculate the area of the penumbra ring around the umbra. -/
theorem penumbra_ring_area (umbra_radius : ℝ) (ratio_umbra : ℝ) (ratio_penumbra : ℝ) : 
  umbra_radius = 40 →
  ratio_umbra = 2 →
  ratio_penumbra = 6 →
  (ratio_penumbra / ratio_umbra * umbra_radius)^2 * Real.pi - umbra_radius^2 * Real.pi = 12800 * Real.pi :=
by sorry

end penumbra_ring_area_l2327_232744


namespace line_segment_division_l2327_232706

/-- Given a line segment with endpoints A(3, 2) and B(12, 8) divided into three equal parts,
    prove that the coordinates of the division points are C(6, 4) and D(9, 6),
    and the length of the segment AB is √117. -/
theorem line_segment_division (A B C D : ℝ × ℝ) : 
  A = (3, 2) → 
  B = (12, 8) → 
  C = ((3 + 0.5 * 12) / 1.5, (2 + 0.5 * 8) / 1.5) → 
  D = ((3 + 2 * 12) / 3, (2 + 2 * 8) / 3) → 
  C = (6, 4) ∧ 
  D = (9, 6) ∧ 
  Real.sqrt ((12 - 3)^2 + (8 - 2)^2) = Real.sqrt 117 :=
sorry

end line_segment_division_l2327_232706


namespace min_n_for_inequality_l2327_232708

-- Define the notation x[n] for repeated exponentiation
def repeated_exp (x : ℕ) : ℕ → ℕ
| 0 => x
| n + 1 => x ^ (repeated_exp x n)

-- Define the specific case for 3[n]
def three_exp (n : ℕ) : ℕ := repeated_exp 3 n

-- Theorem statement
theorem min_n_for_inequality : 
  ∀ n : ℕ, three_exp n > 3^(2^9) ↔ n ≥ 10 :=
sorry

end min_n_for_inequality_l2327_232708


namespace counterexample_twelve_l2327_232746

theorem counterexample_twelve : ∃ n : ℕ, 
  ¬(Nat.Prime n) ∧ (n = 12) ∧ ¬(Nat.Prime (n - 1) ∧ Nat.Prime (n - 2)) :=
by sorry

end counterexample_twelve_l2327_232746


namespace sequence_well_defined_and_nonzero_l2327_232771

def x : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 * x n / ((x n)^2 - 1)

def y : ℕ → ℚ
  | 0 => 4
  | n + 1 => 2 * y n / ((y n)^2 - 1)

def z : ℕ → ℚ
  | 0 => 6/7
  | n + 1 => 2 * z n / ((z n)^2 - 1)

theorem sequence_well_defined_and_nonzero (n : ℕ) :
  (x n ≠ 1 ∧ x n ≠ -1) ∧
  (y n ≠ 1 ∧ y n ≠ -1) ∧
  (z n ≠ 1 ∧ z n ≠ -1) ∧
  x n + y n + z n ≠ 0 :=
sorry

end sequence_well_defined_and_nonzero_l2327_232771


namespace normal_distribution_two_std_dev_below_mean_l2327_232762

theorem normal_distribution_two_std_dev_below_mean 
  (μ σ : ℝ) 
  (h_mean : μ = 14.5) 
  (h_std_dev : σ = 1.5) : 
  μ - 2 * σ = 11.5 := by
  sorry

end normal_distribution_two_std_dev_below_mean_l2327_232762


namespace min_sum_of_coefficients_l2327_232704

theorem min_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, a * x + b * y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) →
  a + b ≥ 16 :=
by sorry

end min_sum_of_coefficients_l2327_232704


namespace math_chemistry_intersection_l2327_232774

/-- Represents the number of students in various groups and their intersections -/
structure StudentGroups where
  total : ℕ
  math : ℕ
  physics : ℕ
  chemistry : ℕ
  math_physics : ℕ
  physics_chemistry : ℕ
  math_chemistry : ℕ

/-- The given conditions for the student groups -/
def given_groups : StudentGroups :=
  { total := 36
  , math := 26
  , physics := 15
  , chemistry := 13
  , math_physics := 6
  , physics_chemistry := 4
  , math_chemistry := 8 }

/-- Theorem stating that the number of students in both math and chemistry is 8 -/
theorem math_chemistry_intersection (g : StudentGroups) (h : g = given_groups) :
  g.math_chemistry = 8 := by
  sorry

end math_chemistry_intersection_l2327_232774


namespace negation_of_universal_proposition_l2327_232745

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2327_232745


namespace bons_win_probability_main_theorem_l2327_232789

/-- The probability of rolling a six. -/
def prob_six : ℚ := 1/6

/-- The probability of not rolling a six. -/
def prob_not_six : ℚ := 1 - prob_six

/-- The probability that Mr. B. Bons wins the game. -/
def prob_bons_win : ℚ := 5/11

theorem bons_win_probability :
  prob_bons_win = prob_not_six * prob_six + prob_not_six * prob_not_six * prob_bons_win :=
by sorry

/-- The main theorem stating that the probability of Mr. B. Bons winning is 5/11. -/
theorem main_theorem : prob_bons_win = 5/11 :=
by sorry

end bons_win_probability_main_theorem_l2327_232789


namespace floor_length_approximately_18_78_l2327_232757

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure Floor :=
  (breadth : ℝ)
  (paintCost : ℝ)
  (paintRate : ℝ)

/-- Calculates the length of the floor given its specifications. -/
def calculateFloorLength (floor : Floor) : ℝ :=
  let length := 3 * floor.breadth
  let area := floor.paintCost / floor.paintRate
  length

/-- Theorem stating that the calculated floor length is approximately 18.78 meters. -/
theorem floor_length_approximately_18_78 (floor : Floor) 
  (h1 : floor.paintCost = 529)
  (h2 : floor.paintRate = 3) :
  ∃ ε > 0, |calculateFloorLength floor - 18.78| < ε :=
sorry

end floor_length_approximately_18_78_l2327_232757


namespace solution_set_f_leq_6_range_of_m_l2327_232721

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = Set.Icc (-1/2) (5/2) := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, 6*m^2 - 4*m < f x} = Set.Ioo (-1/3) 1 := by sorry

end solution_set_f_leq_6_range_of_m_l2327_232721


namespace sqrt_four_ninths_l2327_232712

theorem sqrt_four_ninths : Real.sqrt (4 / 9) = 2 / 3 := by
  sorry

end sqrt_four_ninths_l2327_232712


namespace daily_profit_at_35_unique_profit_600_no_profit_900_l2327_232749

/-- The daily profit function for a product -/
def P (x : ℝ) : ℝ := (x - 30) * (-2 * x + 140)

/-- The purchase price of the product -/
def purchase_price : ℝ := 30

/-- The lower bound of the selling price -/
def lower_bound : ℝ := 30

/-- The upper bound of the selling price -/
def upper_bound : ℝ := 55

theorem daily_profit_at_35 :
  P 35 = 350 := by sorry

theorem unique_profit_600 :
  ∃! x : ℝ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ P x = 600 ∧ x = 40 := by sorry

theorem no_profit_900 :
  ¬ ∃ x : ℝ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ P x = 900 := by sorry

end daily_profit_at_35_unique_profit_600_no_profit_900_l2327_232749


namespace arrangement_count_l2327_232760

def num_boxes : ℕ := 6
def num_digits : ℕ := 5

theorem arrangement_count :
  (num_boxes.factorial : ℕ) = 720 :=
sorry

end arrangement_count_l2327_232760


namespace max_M_min_N_equals_two_thirds_l2327_232726

theorem max_M_min_N_equals_two_thirds (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let M := x / (2 * x + y) + y / (x + 2 * y)
  let N := x / (x + 2 * y) + y / (2 * x + y)
  (∀ a b : ℝ, a > 0 → b > 0 → M ≤ (a / (2 * a + b) + b / (a + 2 * b))) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → N ≥ (a / (a + 2 * b) + b / (2 * a + b))) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ M = 2/3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ N = 2/3) :=
by sorry


end max_M_min_N_equals_two_thirds_l2327_232726


namespace pump_rates_determination_l2327_232717

/-- Represents the pumping rates and durations of three water pumps -/
structure PumpSystem where
  rate1 : ℝ  -- Pumping rate of the first pump
  rate2 : ℝ  -- Pumping rate of the second pump
  rate3 : ℝ  -- Pumping rate of the third pump
  time1 : ℝ  -- Working time of the first pump
  time2 : ℝ  -- Working time of the second pump
  time3 : ℝ  -- Working time of the third pump

/-- Checks if the given pump system satisfies all the conditions -/
def satisfiesConditions (p : PumpSystem) : Prop :=
  p.time1 = p.time3 ∧  -- First and third pumps finish simultaneously
  p.time2 = 2 ∧  -- Second pump works for 2 hours
  p.rate1 * p.time1 = 9 ∧  -- First pump pumps 9 m³
  p.rate2 * p.time2 + p.rate3 * p.time3 = 28 ∧  -- Second and third pumps pump 28 m³ together
  p.rate3 = p.rate1 + 3 ∧  -- Third pump pumps 3 m³ more per hour than the first
  p.rate1 + p.rate2 + p.rate3 = 14  -- Three pumps together pump 14 m³ per hour

/-- Theorem stating that the given conditions imply specific pumping rates -/
theorem pump_rates_determination (p : PumpSystem) 
  (h : satisfiesConditions p) : p.rate1 = 3 ∧ p.rate2 = 5 ∧ p.rate3 = 6 := by
  sorry


end pump_rates_determination_l2327_232717


namespace eugene_toothpick_boxes_l2327_232728

/-- Represents the number of toothpicks needed for Eugene's model house --/
def toothpicks_needed (total_cards : ℕ) (unused_cards : ℕ) (wall_toothpicks : ℕ) 
  (window_count : ℕ) (door_count : ℕ) (window_door_toothpicks : ℕ) (roof_toothpicks : ℕ) : ℕ :=
  let used_cards := total_cards - unused_cards
  let wall_total := used_cards * wall_toothpicks
  let window_door_total := used_cards * (window_count + door_count) * window_door_toothpicks
  wall_total + window_door_total + roof_toothpicks

/-- Theorem stating that Eugene used at least 7 boxes of toothpicks --/
theorem eugene_toothpick_boxes : 
  ∀ (box_capacity : ℕ),
  box_capacity = 750 →
  ∃ (n : ℕ), n ≥ 7 ∧ 
  n * box_capacity ≥ toothpicks_needed 52 23 64 3 2 12 1250 :=
by sorry

end eugene_toothpick_boxes_l2327_232728


namespace smallest_n_for_digit_rearrangement_l2327_232785

/-- Represents a natural number as a list of its digits -/
def Digits : Type := List Nat

/-- Returns true if two lists of digits represent numbers that differ by a sequence of n ones -/
def differsBy111 (a b : Digits) (n : Nat) : Prop := sorry

/-- Returns true if two lists of digits are permutations of each other -/
def isPermutation (a b : Digits) : Prop := sorry

/-- Theorem: The smallest n for which there exist two numbers A and B,
    where B is a permutation of A's digits and A - B is n ones, is 9 -/
theorem smallest_n_for_digit_rearrangement :
  ∃ (a b : Digits),
    isPermutation a b ∧
    differsBy111 a b 9 ∧
    ∀ (n : Nat), n < 9 →
      ¬∃ (x y : Digits), isPermutation x y ∧ differsBy111 x y n :=
by sorry

end smallest_n_for_digit_rearrangement_l2327_232785


namespace integral_square_le_four_integral_derivative_square_l2327_232719

open MeasureTheory Interval RealInnerProductSpace

theorem integral_square_le_four_integral_derivative_square
  (f : ℝ → ℝ) (hf : ContDiff ℝ 1 f) (h : ∃ x₀ ∈ Set.Icc 0 1, f x₀ = 0) :
  ∫ x in Set.Icc 0 1, (f x)^2 ≤ 4 * ∫ x in Set.Icc 0 1, (deriv f x)^2 :=
sorry

end integral_square_le_four_integral_derivative_square_l2327_232719


namespace t_100_gt_t_99_l2327_232756

/-- The number of ways to place n objects with weights 1 to n on a balance with equal weight in each pan. -/
def T (n : ℕ) : ℕ := sorry

/-- Theorem: T(100) is greater than T(99). -/
theorem t_100_gt_t_99 : T 100 > T 99 := by sorry

end t_100_gt_t_99_l2327_232756


namespace prob_fewer_heads_12_fair_coins_l2327_232727

/-- The number of coin flips -/
def n : ℕ := 12

/-- The probability of getting heads on a single fair coin flip -/
def p : ℚ := 1/2

/-- The probability of getting fewer heads than tails in n fair coin flips -/
def prob_fewer_heads (n : ℕ) (p : ℚ) : ℚ :=
  sorry

theorem prob_fewer_heads_12_fair_coins : 
  prob_fewer_heads n p = 793/2048 := by sorry

end prob_fewer_heads_12_fair_coins_l2327_232727


namespace sales_decrease_equation_l2327_232723

/-- Represents the monthly decrease rate as a real number between 0 and 1 -/
def monthly_decrease_rate : ℝ := sorry

/-- The initial sales amount in August -/
def initial_sales : ℝ := 42

/-- The final sales amount in October -/
def final_sales : ℝ := 27

/-- The number of months between August and October -/
def months_elapsed : ℕ := 2

theorem sales_decrease_equation :
  initial_sales * (1 - monthly_decrease_rate) ^ months_elapsed = final_sales :=
sorry

end sales_decrease_equation_l2327_232723


namespace inscribed_circle_radius_l2327_232711

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 12) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 12 / (7 + 2 * Real.sqrt 14) :=
by sorry

end inscribed_circle_radius_l2327_232711
