import Mathlib

namespace x_squared_eq_neg_one_is_quadratic_l734_73423

/-- A quadratic equation in one variable -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Check if an equation is in the form ax² + bx + c = 0 -/
def isQuadraticForm (f : ℝ → ℝ) : Prop :=
  ∃ (q : QuadraticEquation), ∀ x, f x = q.a * x^2 + q.b * x + q.c

/-- The specific equation x² = -1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: The equation x² = -1 is a quadratic equation in one variable -/
theorem x_squared_eq_neg_one_is_quadratic : isQuadraticForm f := by sorry

end x_squared_eq_neg_one_is_quadratic_l734_73423


namespace town_population_problem_l734_73486

theorem town_population_problem (original : ℕ) : 
  (((original + 1500) * 85 / 100) : ℕ) = original - 45 → original = 8800 := by
  sorry

end town_population_problem_l734_73486


namespace inverse_composition_equals_two_l734_73443

-- Define the function f
def f : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 2
| 4 => 5
| 5 => 1

-- Assume f has an inverse
axiom f_has_inverse : Function.Bijective f

-- Define f⁻¹ using the inverse of f
noncomputable def f_inv : Fin 5 → Fin 5 := Function.invFun f

-- State the theorem
theorem inverse_composition_equals_two :
  f_inv (f_inv (f_inv 3)) = 2 := by sorry

end inverse_composition_equals_two_l734_73443


namespace equation_solution_l734_73422

theorem equation_solution : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7 / 4 := by
  sorry

end equation_solution_l734_73422


namespace intersection_of_A_and_B_l734_73448

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l734_73448


namespace octal_to_decimal_l734_73403

theorem octal_to_decimal (octal_num : ℕ) : octal_num = 362 → 
  (3 * 8^2 + 6 * 8^1 + 2 * 8^0) = 242 := by
  sorry

end octal_to_decimal_l734_73403


namespace three_coins_same_probability_l734_73460

def coin_flip := Bool

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes (n : ℕ) : ℕ := 2 * 2^(n - 3)

theorem three_coins_same_probability (n : ℕ) (h : n = 6) :
  (favorable_outcomes n : ℚ) / (total_outcomes n : ℚ) = 1/4 :=
sorry

end three_coins_same_probability_l734_73460


namespace bridge_length_l734_73410

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 240 :=
by
  sorry

#check bridge_length

end bridge_length_l734_73410


namespace arithmetic_mean_property_l734_73488

def consecutive_digits_set : List Nat := [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789]

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

def digits (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 10) ((m % 10) :: acc)
    go n []

theorem arithmetic_mean_property :
  let M : Rat := arithmetic_mean consecutive_digits_set
  (M = 137174210) ∧
  (∀ d : Nat, d < 10 → (d ≠ 5 ↔ d ∈ digits M.num.toNat)) := by
  sorry

end arithmetic_mean_property_l734_73488


namespace differences_of_geometric_progression_l734_73479

/-- Given a geometric progression with first term a₁ and common ratio q,
    the sequence of differences between consecutive terms forms a geometric progression
    with first term a₁(q - 1) and common ratio q. -/
theorem differences_of_geometric_progression
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1) :
  let gp : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  let diff : ℕ → ℝ := λ n => gp (n + 1) - gp n
  ∀ n : ℕ, diff (n + 1) = q * diff n :=
by sorry

end differences_of_geometric_progression_l734_73479


namespace room_width_l734_73481

/-- Given a rectangular room with area 10 square feet and length 5 feet, prove the width is 2 feet -/
theorem room_width (area : ℝ) (length : ℝ) (width : ℝ) : 
  area = 10 → length = 5 → area = length * width → width = 2 := by
  sorry

end room_width_l734_73481


namespace a_fourth_minus_b_fourth_l734_73418

theorem a_fourth_minus_b_fourth (a b : ℝ) 
  (h1 : a - b = 1) 
  (h2 : a^2 - b^2 = -1) : 
  a^4 - b^4 = -1 := by
sorry

end a_fourth_minus_b_fourth_l734_73418


namespace ceiling_cube_fraction_plus_one_l734_73474

theorem ceiling_cube_fraction_plus_one :
  ⌈(-5/3)^3 + 1⌉ = -3 := by
  sorry

end ceiling_cube_fraction_plus_one_l734_73474


namespace second_meeting_at_six_minutes_l734_73483

/-- Represents a swimmer in the race -/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting given a race scenario -/
def secondMeetingTime (race : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the second meeting occurs at 6 minutes -/
theorem second_meeting_at_six_minutes (race : RaceScenario) 
  (h1 : race.poolLength = 120)
  (h2 : race.swimmer1.startPosition = 0)
  (h3 : race.swimmer2.startPosition = 120)
  (h4 : race.firstMeetingTime = 1)
  (h5 : race.firstMeetingPosition = 40)
  (h6 : race.swimmer1.speed = race.firstMeetingPosition / race.firstMeetingTime)
  (h7 : race.swimmer2.speed = (race.poolLength - race.firstMeetingPosition) / race.firstMeetingTime) :
  secondMeetingTime race = 6 :=
sorry

end second_meeting_at_six_minutes_l734_73483


namespace sam_picked_42_cans_l734_73434

/-- The number of cans Sam picked up in total -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem: Sam picked up 42 cans in total -/
theorem sam_picked_42_cans :
  total_cans 4 3 6 = 42 := by
  sorry

end sam_picked_42_cans_l734_73434


namespace intercept_sum_mod_17_l734_73412

theorem intercept_sum_mod_17 :
  ∃! (x₀ y₀ : ℕ), x₀ < 17 ∧ y₀ < 17 ∧
  (5 * x₀ ≡ 2 [MOD 17]) ∧
  (3 * y₀ + 2 ≡ 0 [MOD 17]) ∧
  x₀ + y₀ = 19 :=
by sorry

end intercept_sum_mod_17_l734_73412


namespace sixth_term_is_twelve_l734_73425

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first four terms is 20
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 20
  -- Fifth term is 10
  fifth_term : a + 4*d = 10

/-- The sixth term of the arithmetic sequence is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) : seq.a + 5*seq.d = 12 := by
  sorry

end sixth_term_is_twelve_l734_73425


namespace sqrt_abs_sum_eq_one_l734_73462

theorem sqrt_abs_sum_eq_one (a : ℝ) (h : 1 < a ∧ a < 2) :
  Real.sqrt ((a - 2)^2) + |a - 1| = 1 := by
  sorry

end sqrt_abs_sum_eq_one_l734_73462


namespace tax_discount_order_invariance_l734_73491

theorem tax_discount_order_invariance 
  (price : ℝ) 
  (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) 
  (discount_rate_pos : 0 < discount_rate) :
  price * (1 + tax_rate) * (1 - discount_rate) = 
  price * (1 - discount_rate) * (1 + tax_rate) :=
sorry

end tax_discount_order_invariance_l734_73491


namespace extremum_at_zero_l734_73463

/-- Given a function f(x) = e^x - ax with an extremum at x = 0, prove that a = 1 -/
theorem extremum_at_zero (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x - a * x) ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x ≤ f 0 ∨ f x ≥ f 0)) → 
  a = 1 := by
  sorry

end extremum_at_zero_l734_73463


namespace base12_addition_l734_73426

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Digit12 to its decimal (base 10) value --/
def toDecimal (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11
  | Digit12.C => 12

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal (base 10) value --/
def base12ToDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => toDecimal d + 12 * acc) 0

/-- The main theorem to prove --/
theorem base12_addition :
  base12ToDecimal [Digit12.D3, Digit12.C, Digit12.D5] +
  base12ToDecimal [Digit12.D2, Digit12.A, Digit12.B] =
  base12ToDecimal [Digit12.D6, Digit12.D3, Digit12.D4] := by
  sorry


end base12_addition_l734_73426


namespace triangle_area_rational_l734_73465

theorem triangle_area_rational (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ) :
  ∃ (q : ℚ), q = (1/2) * |((x₁ + (1/2 : ℚ)) * (y₂ - y₃) + 
                           (x₂ + (1/2 : ℚ)) * (y₃ - y₁) + 
                           (x₃ + (1/2 : ℚ)) * (y₁ - y₂))| := by
  sorry

end triangle_area_rational_l734_73465


namespace grass_field_width_l734_73433

/-- Proves that given a rectangular grass field with length 75 m, surrounded by a 2.5 m wide path,
    if the cost of constructing the path is Rs. 6750 at Rs. 10 per sq m,
    then the width of the grass field is 55 m. -/
theorem grass_field_width (field_length : ℝ) (path_width : ℝ) (path_cost : ℝ) (cost_per_sqm : ℝ) :
  field_length = 75 →
  path_width = 2.5 →
  path_cost = 6750 →
  cost_per_sqm = 10 →
  ∃ (field_width : ℝ),
    field_width = 55 ∧
    path_cost = cost_per_sqm * (
      (field_length + 2 * path_width) * (field_width + 2 * path_width) -
      field_length * field_width
    ) := by
  sorry

end grass_field_width_l734_73433


namespace arun_speed_doubling_l734_73489

/-- Proves that Arun takes 1 hour less than Anil when he doubles his speed -/
theorem arun_speed_doubling (distance : ℝ) (arun_speed : ℝ) (anil_time : ℝ) :
  distance = 30 →
  arun_speed = 5 →
  distance / arun_speed = anil_time + 2 →
  distance / (2 * arun_speed) = anil_time - 1 := by
  sorry

#check arun_speed_doubling

end arun_speed_doubling_l734_73489


namespace sum_of_150_consecutive_integers_l734_73429

def is_sum_of_consecutive_integers (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : ℕ, n = (k * (2 * a + k - 1)) / 2

theorem sum_of_150_consecutive_integers :
  is_sum_of_consecutive_integers 4692583675 150 ∧
  ¬ is_sum_of_consecutive_integers 1627386425 150 ∧
  ¬ is_sum_of_consecutive_integers 2345680925 150 ∧
  ¬ is_sum_of_consecutive_integers 3579113450 150 ∧
  ¬ is_sum_of_consecutive_integers 5815939525 150 :=
by sorry

end sum_of_150_consecutive_integers_l734_73429


namespace min_sum_position_max_product_position_l734_73498

/-- Represents the special number with 1991 nines between two ones -/
def specialNumber : ℕ := 1 * 10^1992 + 1

/-- Calculates the sum when splitting the number at position m -/
def sumAtPosition (m : ℕ) : ℕ := 
  (2 * 10^m - 1) + (10^(1992 - m) - 9)

/-- Calculates the product when splitting the number at position m -/
def productAtPosition (m : ℕ) : ℕ := 
  (2 * 10^m - 1) * (10^(1992 - m) - 9)

theorem min_sum_position : 
  ∀ m : ℕ, m ≠ 996 → m ≠ 997 → sumAtPosition m > sumAtPosition 996 :=
sorry

theorem max_product_position : 
  ∀ m : ℕ, m ≠ 995 → m ≠ 996 → productAtPosition m < productAtPosition 995 :=
sorry

end min_sum_position_max_product_position_l734_73498


namespace zeros_after_decimal_for_one_over_twelve_to_twelve_l734_73431

-- Define the function to count zeros after decimal point
def count_zeros_after_decimal (x : ℚ) : ℕ :=
  sorry

-- Theorem statement
theorem zeros_after_decimal_for_one_over_twelve_to_twelve :
  count_zeros_after_decimal (1 / (12^12)) = 11 :=
sorry

end zeros_after_decimal_for_one_over_twelve_to_twelve_l734_73431


namespace colleen_paid_more_than_joy_l734_73457

/-- The amount of money Colleen paid more than Joy for pencils -/
def extra_cost (joy_pencils colleen_pencils price_per_pencil : ℕ) : ℕ :=
  (colleen_pencils - joy_pencils) * price_per_pencil

/-- Proof that Colleen paid $80 more than Joy for pencils -/
theorem colleen_paid_more_than_joy :
  extra_cost 30 50 4 = 80 := by
  sorry

end colleen_paid_more_than_joy_l734_73457


namespace alok_veggie_plates_l734_73478

/-- Represents the order and payment details of Alok's meal -/
structure MealOrder where
  chapatis : Nat
  rice_plates : Nat
  ice_cream_cups : Nat
  chapati_cost : Nat
  rice_cost : Nat
  veggie_cost : Nat
  total_paid : Nat

/-- Calculates the number of mixed vegetable plates ordered -/
def veggie_plates_ordered (order : MealOrder) : Nat :=
  let known_cost := order.chapatis * order.chapati_cost + order.rice_plates * order.rice_cost
  let veggie_total_cost := order.total_paid - known_cost
  veggie_total_cost / order.veggie_cost

/-- Theorem stating that Alok ordered 11 plates of mixed vegetable -/
theorem alok_veggie_plates (order : MealOrder) 
        (h1 : order.chapatis = 16)
        (h2 : order.rice_plates = 5)
        (h3 : order.ice_cream_cups = 6)
        (h4 : order.chapati_cost = 6)
        (h5 : order.rice_cost = 45)
        (h6 : order.veggie_cost = 70)
        (h7 : order.total_paid = 1111) :
        veggie_plates_ordered order = 11 := by
  sorry

end alok_veggie_plates_l734_73478


namespace sqrt_inequality_l734_73447

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end sqrt_inequality_l734_73447


namespace multiples_4_or_9_less_than_201_l734_73458

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max divisor : ℕ) : ℕ :=
  (max - 1) / divisor

def count_either_not_both (max a b : ℕ) : ℕ :=
  count_multiples max a + count_multiples max b - 2 * count_multiples max (lcm a b)

theorem multiples_4_or_9_less_than_201 :
  count_either_not_both 201 4 9 = 62 := by sorry

end multiples_4_or_9_less_than_201_l734_73458


namespace essay_competition_probability_l734_73427

/-- The number of topics in the essay competition -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from a pool of 6 topics is 5/6 -/
theorem essay_competition_probability :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics :=
sorry

end essay_competition_probability_l734_73427


namespace negative_two_x_plus_two_positive_l734_73401

theorem negative_two_x_plus_two_positive (x : ℝ) : x < 1 → -2*x + 2 > 0 := by
  sorry

end negative_two_x_plus_two_positive_l734_73401


namespace sum_reciprocal_inequality_l734_73436

theorem sum_reciprocal_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) ≥ 16 := by
  sorry

end sum_reciprocal_inequality_l734_73436


namespace oak_trees_cut_down_problem_l734_73435

/-- The number of oak trees cut down in a park --/
def oak_trees_cut_down (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

/-- Theorem: Given 9 initial oak trees and 7 remaining after cutting, 2 oak trees were cut down --/
theorem oak_trees_cut_down_problem : oak_trees_cut_down 9 7 = 2 := by
  sorry

end oak_trees_cut_down_problem_l734_73435


namespace craftsman_production_l734_73444

/-- The number of parts manufactured by a master craftsman during a shift -/
def total_parts : ℕ := by sorry

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 1.5

theorem craftsman_production :
  let initial_rate := first_hour_parts
  let new_rate := initial_rate + rate_increase
  let remaining_parts := total_parts - first_hour_parts
  (remaining_parts : ℚ) / initial_rate - (remaining_parts : ℚ) / new_rate = time_saved →
  total_parts = 210 := by sorry

end craftsman_production_l734_73444


namespace min_value_expression_l734_73469

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^2 + 4*x*y + 9*y^2 + 8*y*z + 3*z^2 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 8*y₀*z₀ + 3*z₀^2 = 18 :=
by sorry

end min_value_expression_l734_73469


namespace post_office_distance_l734_73476

theorem post_office_distance (outbound_speed inbound_speed : ℝ) 
  (total_time : ℝ) (distance : ℝ) : 
  outbound_speed = 12.5 →
  inbound_speed = 2 →
  total_time = 5.8 →
  distance / outbound_speed + distance / inbound_speed = total_time →
  distance = 10 := by
sorry

end post_office_distance_l734_73476


namespace cube_volume_problem_l734_73466

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (3 * a) * (a / 2) * a - a^3 = 2 * a^2 → a^3 = 64 := by
sorry

end cube_volume_problem_l734_73466


namespace sum_of_roots_l734_73438

-- Define the quadratic equation
def quadratic_equation (x m n : ℝ) : Prop := 2 * x^2 + m * x + n = 0

-- State the theorem
theorem sum_of_roots (m n : ℝ) 
  (hm : quadratic_equation m m n) 
  (hn : quadratic_equation n m n) : 
  m + n = -m / 2 := by
  sorry

end sum_of_roots_l734_73438


namespace total_highlighters_l734_73455

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : pink = 10) (h2 : yellow = 15) (h3 : blue = 8) : 
  pink + yellow + blue = 33 := by
  sorry

end total_highlighters_l734_73455


namespace five_balls_three_boxes_l734_73430

/-- Represents the number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 18 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 18 := by
  sorry

end five_balls_three_boxes_l734_73430


namespace janous_conjecture_l734_73492

theorem janous_conjecture (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 := by
  sorry

end janous_conjecture_l734_73492


namespace number_of_ways_to_turn_off_lights_l734_73467

/-- The number of streetlights --/
def total_lights : ℕ := 12

/-- The number of lights that can be turned off --/
def lights_off : ℕ := 3

/-- The number of positions where lights can be turned off --/
def eligible_positions : ℕ := total_lights - 2 - lights_off + 1

/-- Theorem stating the number of ways to turn off lights --/
theorem number_of_ways_to_turn_off_lights : 
  Nat.choose eligible_positions lights_off = 56 := by sorry

end number_of_ways_to_turn_off_lights_l734_73467


namespace complex_equation_solution_l734_73473

theorem complex_equation_solution (a : ℝ) : (Complex.I * a + 1) * (a - Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l734_73473


namespace equation_one_solution_equation_two_no_solution_l734_73442

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  (9 / x = 8 / (x - 1)) ↔ x = 9 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), ((x - 8) / (x - 7) - 8 = 1 / (7 - x)) :=
sorry

end equation_one_solution_equation_two_no_solution_l734_73442


namespace no_bounded_integral_exists_l734_73421

/-- Base 2 representation of x in [0, 1) -/
def base2Rep (x : ℝ) : ℕ → Fin 2 :=
  sorry

/-- Function f_n as defined in the problem -/
def f_n (n : ℕ) (x : ℝ) : ℤ :=
  sorry

/-- The main theorem -/
theorem no_bounded_integral_exists :
  ∀ (φ : ℝ → ℝ),
    (∀ y, 0 ≤ φ y) →
    (∀ M, ∃ N, ∀ x, N ≤ x → M ≤ φ x) →
    (∀ B, ∃ n : ℕ, B < ∫ x in (0 : ℝ)..1, φ (|f_n n x|)) :=
  sorry

end no_bounded_integral_exists_l734_73421


namespace log_ten_seven_in_terms_of_pqr_l734_73416

theorem log_ten_seven_in_terms_of_pqr (p q r : ℝ) 
  (hp : Real.log 3 / Real.log 8 = p)
  (hq : Real.log 5 / Real.log 3 = q)
  (hr : Real.log 7 / Real.log 4 = r) :
  Real.log 7 / Real.log 10 = 2 * r / (1 + 4 * q * p) := by
  sorry

end log_ten_seven_in_terms_of_pqr_l734_73416


namespace car_speed_first_hour_l734_73452

/-- Given a car's speed over two hours, prove its speed in the first hour -/
theorem car_speed_first_hour (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 30 →
  average_speed = 65 →
  ∃ speed_first_hour : ℝ,
    speed_first_hour = 100 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end car_speed_first_hour_l734_73452


namespace stating_course_selection_schemes_l734_73494

/-- Represents the number of elective courses -/
def num_courses : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of courses with no students -/
def courses_with_no_students : ℕ := 2

/-- Represents the number of courses with students -/
def courses_with_students : ℕ := num_courses - courses_with_no_students

/-- 
  Theorem stating that the number of ways to distribute students among courses
  under the given conditions is 18
-/
theorem course_selection_schemes : 
  (num_courses.choose courses_with_students) * 
  ((num_students.choose courses_with_students) / 2) = 18 := by
  sorry

end stating_course_selection_schemes_l734_73494


namespace number_divisibility_l734_73482

theorem number_divisibility (N : ℕ) : 
  N % 7 = 0 ∧ N % 11 = 2 → N / 7 = 5 := by
  sorry

end number_divisibility_l734_73482


namespace truck_toll_calculation_l734_73454

/-- Calculates the number of axles for a truck given the total number of wheels,
    number of wheels on the front axle, and number of wheels on other axles -/
def calculateAxles (totalWheels : Nat) (frontAxleWheels : Nat) (otherAxleWheels : Nat) : Nat :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck given the number of axles -/
def calculateToll (axles : Nat) : Real :=
  1.50 + 1.50 * (axles - 2)

theorem truck_toll_calculation :
  let totalWheels : Nat := 18
  let frontAxleWheels : Nat := 2
  let otherAxleWheels : Nat := 4
  let axles := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  calculateToll axles = 6.00 := by
  sorry

end truck_toll_calculation_l734_73454


namespace min_sum_squares_l734_73464

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 7, 9}

theorem min_sum_squares (p q r s t u v w x : Int) 
  (hp : p ∈ S) (hq : q ∈ S) (hr : r ∈ S) (hs : s ∈ S) 
  (ht : t ∈ S) (hu : u ∈ S) (hv : v ∈ S) (hw : w ∈ S) (hx : x ∈ S)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ p ≠ x ∧
               q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧ q ≠ x ∧
               r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ r ≠ x ∧
               s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ s ≠ x ∧
               t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ t ≠ x ∧
               u ≠ v ∧ u ≠ w ∧ u ≠ x ∧
               v ≠ w ∧ v ≠ x ∧
               w ≠ x) :
  (p + q + r + s)^2 + (t + u + v + w + x)^2 ≥ 18 := by
sorry

end min_sum_squares_l734_73464


namespace decimal_168_equals_binary_10101000_l734_73477

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_168_equals_binary_10101000 :
  toBinary 168 = [false, false, false, true, false, true, false, true] ∧
  fromBinary [false, false, false, true, false, true, false, true] = 168 :=
by sorry

end decimal_168_equals_binary_10101000_l734_73477


namespace problem_solution_l734_73409

theorem problem_solution : 
  |Real.sqrt 3 - 2| + (27 : ℝ) ^ (1/3 : ℝ) - Real.sqrt 16 + (-1) ^ 2023 = -Real.sqrt 3 := by
  sorry

end problem_solution_l734_73409


namespace ellipse_circle_tangent_perpendicular_l734_73446

/-- Ellipse M with focal length 2√3 and eccentricity √3/2 -/
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Circle N with radius r -/
def circle_N (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

/-- Tangent line l with slope k -/
def line_l (x y k m : ℝ) : Prop :=
  y = k * x + m

/-- P and Q are intersection points of line l and ellipse M -/
def intersection_points (P Q : ℝ × ℝ) (k m : ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  ellipse_M x₁ y₁ ∧ ellipse_M x₂ y₂ ∧
  line_l x₁ y₁ k m ∧ line_l x₂ y₂ k m

/-- OP and OQ are perpendicular -/
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_circle_tangent_perpendicular (k m r : ℝ) (P Q : ℝ × ℝ) :
  m^2 = r^2 * (k^2 + 1) →
  intersection_points P Q k m →
  (perpendicular P Q ↔ r = 2 * Real.sqrt 5 / 5) :=
sorry

end ellipse_circle_tangent_perpendicular_l734_73446


namespace isosceles_triangle_base_length_l734_73451

/-- An isosceles triangle with perimeter 13 and one side length 3 -/
structure IsoscelesTriangle where
  -- The length of two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The triangle is isosceles
  isIsosceles : side ≠ base
  -- The perimeter is 13
  perimeterIs13 : side + side + base = 13
  -- One side length is 3
  oneSideIs3 : side = 3 ∨ base = 3

/-- The base of an isosceles triangle with perimeter 13 and one side 3 must be 3 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.base = 3 := by
  sorry

end isosceles_triangle_base_length_l734_73451


namespace no_integer_points_between_l734_73420

def point_C : ℤ × ℤ := (2, 3)
def point_D : ℤ × ℤ := (101, 200)

def is_between (a b c : ℤ) : Prop := a < b ∧ b < c

theorem no_integer_points_between :
  ¬ ∃ (x y : ℤ), 
    (is_between point_C.1 x point_D.1) ∧ 
    (is_between point_C.2 y point_D.2) ∧ 
    (y - point_C.2) * (point_D.1 - point_C.1) = (point_D.2 - point_C.2) * (x - point_C.1) :=
sorry

end no_integer_points_between_l734_73420


namespace parade_tricycles_l734_73400

theorem parade_tricycles :
  ∀ (w b t : ℕ),
    w + b + t = 10 →
    2 * b + 3 * t = 25 →
    t = 5 :=
by
  sorry

end parade_tricycles_l734_73400


namespace line_segment_ratio_l734_73496

theorem line_segment_ratio (x y z s : ℝ) 
  (h1 : x < y ∧ y < z)
  (h2 : x / y = y / z)
  (h3 : x + y + z = s)
  (h4 : x + y = z) :
  x / y = (-1 + Real.sqrt 5) / 2 := by
sorry

end line_segment_ratio_l734_73496


namespace shekar_average_marks_l734_73424

def shekar_scores : List ℝ := [92, 78, 85, 67, 89, 74, 81, 95, 70, 88]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℝ) = 81.9 := by
  sorry

end shekar_average_marks_l734_73424


namespace ten_point_circle_triangles_l734_73428

/-- The number of ways to choose 3 points from n points to form a triangle -/
def triangles_from_points (n : ℕ) : ℕ := n.choose 3

/-- Given 10 points on a circle, the number of inscribed triangles is 360 -/
theorem ten_point_circle_triangles :
  triangles_from_points 10 = 360 := by
  sorry

end ten_point_circle_triangles_l734_73428


namespace sequence_eventually_periodic_l734_73402

/-- Two sequences of positive integers satisfying the given conditions -/
def SequencePair : Type :=
  { pair : (ℕ → ℕ) × (ℕ → ℕ) //
    (∀ n, pair.1 n > 0 ∧ pair.2 n > 0) ∧
    pair.1 0 ≥ 2 ∧ pair.2 0 ≥ 2 ∧
    (∀ n, pair.1 (n + 1) = Nat.gcd (pair.1 n) (pair.2 n) + 1) ∧
    (∀ n, pair.2 (n + 1) = Nat.lcm (pair.1 n) (pair.2 n) - 1) }

/-- The sequence a_n is eventually periodic -/
theorem sequence_eventually_periodic (seq : SequencePair) :
  ∃ (N t : ℕ), t > 0 ∧ ∀ n ≥ N, seq.1.1 (n + t) = seq.1.1 n :=
sorry

end sequence_eventually_periodic_l734_73402


namespace people_in_line_l734_73487

theorem people_in_line (total : ℕ) (left : ℕ) (right : ℕ) : 
  total = 11 → left = 5 → right = total - left - 1 :=
by
  sorry

end people_in_line_l734_73487


namespace bisection_termination_condition_l734_73468

/-- The bisection method termination condition -/
def is_termination_condition (x₁ x₂ e : ℝ) : Prop :=
  |x₁ - x₂| < e

/-- Theorem stating that the correct termination condition for the bisection method is |x₁ - x₂| < e -/
theorem bisection_termination_condition (x₁ x₂ e : ℝ) (h : e > 0) :
  is_termination_condition x₁ x₂ e ↔ |x₁ - x₂| < e := by sorry

end bisection_termination_condition_l734_73468


namespace tracy_candies_problem_l734_73450

theorem tracy_candies_problem (x : ℕ) : 
  (x % 4 = 0) →  -- x is divisible by 4
  (x % 2 = 0) →  -- x is divisible by 2
  (∃ y : ℕ, 2 ≤ y ∧ y ≤ 6 ∧ x / 2 - 20 - y = 5) →  -- sister took between 2 to 6 candies, leaving 5
  x = 60 :=
by
  sorry

end tracy_candies_problem_l734_73450


namespace range_of_a_l734_73439

/-- The function g(x) = ax + 2 where a > 0 -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, g a x₁ = f x₀) →
  a ∈ Set.Ioo 0 1 :=
sorry

end range_of_a_l734_73439


namespace adjacent_complementary_implies_complementary_l734_73499

/-- Two angles are adjacent if they share a common vertex and a common side. -/
def adjacent_angles (α β : Real) : Prop := sorry

/-- Two angles are complementary if their measures add up to 90 degrees. -/
def complementary_angles (α β : Real) : Prop := α + β = 90

theorem adjacent_complementary_implies_complementary 
  (α β : Real) (h1 : adjacent_angles α β) (h2 : complementary_angles α β) : 
  complementary_angles α β :=
sorry

end adjacent_complementary_implies_complementary_l734_73499


namespace otimes_two_three_l734_73413

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem to prove
theorem otimes_two_three : otimes 2 3 = 23 := by
  sorry

end otimes_two_three_l734_73413


namespace car_distances_theorem_l734_73495

/-- Represents the distance traveled by a car -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distances traveled by two cars under given conditions -/
theorem car_distances_theorem (distance_AB : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ) 
  (h1 : distance_AB = 70)
  (h2 : speed_car1 = 30)
  (h3 : speed_car2 = 40)
  (h4 : speed_car1 + speed_car2 > 0) -- Ensure division by zero is avoided
  : ∃ (time : ℝ), 
    distance speed_car1 time = 150 ∧ 
    distance speed_car2 time = 200 ∧
    time * (speed_car1 + speed_car2) = 5 * distance_AB :=
sorry

end car_distances_theorem_l734_73495


namespace pumpkin_weight_sum_total_pumpkin_weight_l734_73404

theorem pumpkin_weight_sum : ℝ → ℝ → ℝ
  | weight1, weight2 => weight1 + weight2

theorem total_pumpkin_weight :
  let weight1 : ℝ := 4
  let weight2 : ℝ := 8.7
  pumpkin_weight_sum weight1 weight2 = 12.7 := by
  sorry

end pumpkin_weight_sum_total_pumpkin_weight_l734_73404


namespace not_both_count_l734_73471

/-- The number of students taking both chemistry and physics -/
def both : ℕ := 15

/-- The total number of students in the chemistry class -/
def chem_total : ℕ := 30

/-- The number of students taking only physics -/
def physics_only : ℕ := 18

/-- The number of students taking chemistry or physics but not both -/
def not_both : ℕ := (chem_total - both) + physics_only

theorem not_both_count : not_both = 33 := by
  sorry

end not_both_count_l734_73471


namespace triangle_segment_length_l734_73440

theorem triangle_segment_length (a b c h x : ℝ) : 
  a = 24 → b = 45 → c = 51 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 40 := by
  sorry

end triangle_segment_length_l734_73440


namespace angle_B_value_max_perimeter_max_perimeter_achievable_l734_73437

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition (2a-c)cos B = b cos C -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/-- Theorem stating that B = π/3 -/
theorem angle_B_value (t : Triangle) (h : triangle_condition t) : t.B = π / 3 :=
sorry

/-- Theorem stating that when b = 2, the maximum perimeter is 6 -/
theorem max_perimeter (t : Triangle) (h : triangle_condition t) (hb : t.b = 2) :
  t.a + t.b + t.c ≤ 6 :=
sorry

/-- Theorem stating that the maximum perimeter of 6 is achievable -/
theorem max_perimeter_achievable : ∃ t : Triangle, triangle_condition t ∧ t.b = 2 ∧ t.a + t.b + t.c = 6 :=
sorry

end angle_B_value_max_perimeter_max_perimeter_achievable_l734_73437


namespace work_completion_time_l734_73461

/-- 
Given a piece of work that can be completed by 9 laborers in 15 days, 
this theorem proves that it would take 9 days for 15 laborers to complete the same work.
-/
theorem work_completion_time 
  (total_laborers : ℕ) 
  (available_laborers : ℕ) 
  (actual_days : ℕ) 
  (h1 : total_laborers = 15)
  (h2 : available_laborers = total_laborers - 6)
  (h3 : actual_days = 15)
  : (available_laborers * actual_days) / total_laborers = 9 := by
  sorry

#check work_completion_time

end work_completion_time_l734_73461


namespace rectangle_division_theorem_l734_73411

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Checks if a point is inside a rectangle -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  r.bottomLeft.x ≤ p.x ∧ p.x ≤ r.topRight.x ∧
  r.bottomLeft.y ≤ p.y ∧ p.y ≤ r.topRight.y

/-- Theorem: Given a rectangle with 4 points, it can be divided into 4 equal rectangles, each containing one point -/
theorem rectangle_division_theorem 
  (r : Rectangle) 
  (p1 p2 p3 p4 : Point) 
  (h1 : pointInRectangle p1 r)
  (h2 : pointInRectangle p2 r)
  (h3 : pointInRectangle p3 r)
  (h4 : pointInRectangle p4 r)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :
  ∃ (r1 r2 r3 r4 : Rectangle),
    -- The four rectangles are equal in area
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r2.topRight.x - r2.bottomLeft.x) * (r2.topRight.y - r2.bottomLeft.y) ∧
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r3.topRight.x - r3.bottomLeft.x) * (r3.topRight.y - r3.bottomLeft.y) ∧
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r4.topRight.x - r4.bottomLeft.x) * (r4.topRight.y - r4.bottomLeft.y) ∧
    -- Each smaller rectangle contains exactly one point
    (pointInRectangle p1 r1 ∧ pointInRectangle p2 r2 ∧ pointInRectangle p3 r3 ∧ pointInRectangle p4 r4) ∧
    -- The union of the smaller rectangles is the original rectangle
    (r1.bottomLeft = r.bottomLeft) ∧ (r4.topRight = r.topRight) ∧
    (r1.topRight.x = r2.bottomLeft.x) ∧ (r2.topRight.x = r.topRight.x) ∧
    (r1.topRight.y = r3.bottomLeft.y) ∧ (r3.topRight.y = r.topRight.y) :=
by
  sorry

end rectangle_division_theorem_l734_73411


namespace geometric_series_ratio_l734_73407

theorem geometric_series_ratio (a r : ℝ) : 
  (∃ (S : ℕ → ℝ), (∀ n, S n = a * r^n) ∧ 
   (∑' n, S n) = 18 ∧ 
   (∑' n, S (2*n + 1)) = 8) →
  r = 4/5 := by
sorry

end geometric_series_ratio_l734_73407


namespace square_tiles_count_l734_73497

/-- Represents a collection of triangular and square tiles. -/
structure TileCollection where
  triangles : ℕ
  squares : ℕ
  total_tiles : ℕ
  total_edges : ℕ
  tiles_sum : triangles + squares = total_tiles
  edges_sum : 3 * triangles + 4 * squares = total_edges

/-- Theorem stating that in a collection of 32 tiles with 110 edges, there are 14 square tiles. -/
theorem square_tiles_count (tc : TileCollection) 
  (h1 : tc.total_tiles = 32) 
  (h2 : tc.total_edges = 110) : 
  tc.squares = 14 := by
  sorry

#check square_tiles_count

end square_tiles_count_l734_73497


namespace decimal_insertion_sum_l734_73441

-- Define a function to represent the possible ways to insert a decimal point in 2016
def insert_decimal (n : ℕ) : List ℝ :=
  [2.016, 20.16, 201.6]

-- Define the problem statement
theorem decimal_insertion_sum :
  ∃ (a b c d e f : ℝ),
    (a ∈ insert_decimal 2016) ∧
    (b ∈ insert_decimal 2016) ∧
    (c ∈ insert_decimal 2016) ∧
    (d ∈ insert_decimal 2016) ∧
    (e ∈ insert_decimal 2016) ∧
    (f ∈ insert_decimal 2016) ∧
    (a + b + c + d + e + f = 46368 / 100) :=
by
  sorry

end decimal_insertion_sum_l734_73441


namespace g_of_one_equals_fifteen_l734_73445

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_one_equals_fifteen :
  (∀ x : ℝ, g (2 * x - 3) = 3 * x + 9) →
  g 1 = 15 := by sorry

end g_of_one_equals_fifteen_l734_73445


namespace wall_width_proof_l734_73405

theorem wall_width_proof (wall_height : ℝ) (painting_width : ℝ) (painting_height : ℝ) (painting_percentage : ℝ) :
  wall_height = 5 →
  painting_width = 2 →
  painting_height = 4 →
  painting_percentage = 0.16 →
  ∃ (wall_width : ℝ), 
    wall_width = 10 ∧
    painting_width * painting_height = painting_percentage * (wall_height * wall_width) :=
by
  sorry

end wall_width_proof_l734_73405


namespace smallest_divisible_by_14_15_18_l734_73417

theorem smallest_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ 14 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n ∧ ∀ m : ℕ, m > 0 → 14 ∣ m → 15 ∣ m → 18 ∣ m → n ≤ m :=
by
  use 630
  sorry

end smallest_divisible_by_14_15_18_l734_73417


namespace largest_perfect_square_factor_882_l734_73456

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_882 :
  largest_perfect_square_factor 882 = 441 := by sorry

end largest_perfect_square_factor_882_l734_73456


namespace cash_percentage_is_twenty_percent_l734_73432

def total_amount : ℝ := 137500
def raw_materials : ℝ := 80000
def machinery : ℝ := 30000

def cash : ℝ := total_amount - (raw_materials + machinery)

theorem cash_percentage_is_twenty_percent :
  (cash / total_amount) * 100 = 20 := by
  sorry

end cash_percentage_is_twenty_percent_l734_73432


namespace extremum_point_implies_a_zero_b_range_for_real_roots_l734_73406

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + x^3 - x^2 - a * x

theorem extremum_point_implies_a_zero :
  (∀ a : ℝ, (∃ ε > 0, ∀ x ∈ Set.Ioo ((2/3) - ε) ((2/3) + ε), f a x ≤ f a (2/3) ∨ f a x ≥ f a (2/3))) →
  (∃ a : ℝ, ∀ x : ℝ, f a x = f a (2/3)) :=
sorry

theorem b_range_for_real_roots :
  ∀ b : ℝ, (∃ x : ℝ, f (-1) (1 - x) - (1 - x)^3 = b) → b ∈ Set.Iic 0 :=
sorry

end extremum_point_implies_a_zero_b_range_for_real_roots_l734_73406


namespace arithmetic_sequence_sum_l734_73414

theorem arithmetic_sequence_sum (a b c : ℝ) : 
  (∃ d : ℝ, a = 3 + d ∧ b = a + d ∧ c = b + d ∧ 15 = c + d) → 
  a + b + c = 27 := by
sorry

end arithmetic_sequence_sum_l734_73414


namespace bicyclist_scooter_meeting_time_l734_73472

/-- Represents a vehicle with a constant speed -/
structure Vehicle where
  speed : ℝ

/-- Represents the time when two vehicles meet -/
def MeetingTime (v1 v2 : Vehicle) : ℝ → Prop := sorry

theorem bicyclist_scooter_meeting_time 
  (car motorcycle scooter bicycle : Vehicle)
  (h1 : MeetingTime car scooter 12)
  (h2 : MeetingTime car bicycle 14)
  (h3 : MeetingTime car motorcycle 16)
  (h4 : MeetingTime motorcycle scooter 17)
  (h5 : MeetingTime motorcycle bicycle 18) :
  MeetingTime bicycle scooter (12 + 10/3) :=
sorry

end bicyclist_scooter_meeting_time_l734_73472


namespace line_circle_intersection_range_l734_73449

/-- The range of m for a line intersecting a circle under specific conditions -/
theorem line_circle_intersection_range (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧ 
    A.1 + A.2 + m = 0 ∧ 
    B.1 + B.2 + m = 0 ∧ 
    A.1^2 + A.2^2 = 2 ∧ 
    B.1^2 + B.2^2 = 2 ∧ 
    ‖(A.1, A.2)‖ + ‖(B.1, B.2)‖ ≥ ‖(A.1 - B.1, A.2 - B.2)‖) →
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 :=
by sorry

end line_circle_intersection_range_l734_73449


namespace negation_equivalence_l734_73419

theorem negation_equivalence : 
  (¬∃ x : ℝ, x^2 - 2*x + 4 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :=
by sorry

end negation_equivalence_l734_73419


namespace min_value_sum_min_value_sum_achieved_l734_73475

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 2*y ≥ 3 + 8*Real.sqrt 2 := by
sorry

theorem min_value_sum_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 2*y₀ = 3 + 8*Real.sqrt 2 := by
sorry

end min_value_sum_min_value_sum_achieved_l734_73475


namespace subcommittee_formation_count_l734_73485

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let selected_republicans : ℕ := 4
  let selected_democrats : ℕ := 3
  (Nat.choose total_republicans selected_republicans) *
  (Nat.choose total_democrats selected_democrats) = 7350 := by
  sorry

end subcommittee_formation_count_l734_73485


namespace multiplier_problem_l734_73459

theorem multiplier_problem (n : ℝ) (m : ℝ) : 
  n = 3 → 7 * n = m * n + 12 → m = 3 := by
  sorry

end multiplier_problem_l734_73459


namespace article_cost_l734_73490

theorem article_cost (selling_price1 selling_price2 : ℝ) (percentage_diff : ℝ) :
  selling_price1 = 350 →
  selling_price2 = 340 →
  percentage_diff = 0.05 →
  (selling_price1 - selling_price2) / (selling_price2 - (selling_price1 - selling_price2) / percentage_diff) = percentage_diff →
  selling_price1 - (selling_price1 - selling_price2) / percentage_diff = 140 :=
by sorry

end article_cost_l734_73490


namespace multiples_count_l734_73408

theorem multiples_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 300 ∧ 2 ∣ n ∧ 5 ∣ n ∧ ¬(3 ∣ n) ∧ ¬(11 ∣ n)) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 300 ∧ 2 ∣ n ∧ 5 ∣ n ∧ ¬(3 ∣ n) ∧ ¬(11 ∣ n) → n ∈ S) ∧
  S.card = 18 :=
by sorry

end multiples_count_l734_73408


namespace circle_center_and_tangent_line_l734_73470

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 - 2*x + y^2 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Define the tangent line l
def tangent_line_equation (x y : ℝ) : Prop := 
  y = (Real.sqrt 3 / 3) * (x + 1) ∨ y = -(Real.sqrt 3 / 3) * (x + 1)

-- Define the point that the line passes through
def point_on_line : ℝ × ℝ := (-1, 0)

theorem circle_center_and_tangent_line :
  (∀ x y, circle_equation x y ↔ (x - 1)^2 + y^2 = 1) ∧
  (tangent_line_equation (point_on_line.1) (point_on_line.2)) ∧
  (∀ x y, tangent_line_equation x y → 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 1 → 
     (x, y) = (x, y) ∨ (x, y) = (x, y))) :=
by sorry

end circle_center_and_tangent_line_l734_73470


namespace arithmetic_sequence_problem_l734_73453

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : isArithmeticSequence a) 
  (h2 : a 7 + a 9 = 8) : 
  a 8 = 4 := by
sorry

end arithmetic_sequence_problem_l734_73453


namespace paper_distribution_l734_73484

theorem paper_distribution (num_students : ℕ) (paper_per_student : ℕ) 
  (h1 : num_students = 230) 
  (h2 : paper_per_student = 15) : 
  num_students * paper_per_student = 3450 := by
  sorry

end paper_distribution_l734_73484


namespace exponential_inequality_l734_73415

theorem exponential_inequality (a x : ℝ) : 
  a > Real.log 2 - 1 → x > 0 → Real.exp x > x^2 - 2*a*x + 1 := by sorry

end exponential_inequality_l734_73415


namespace sum_area_15_disks_on_unit_circle_l734_73493

/-- The sum of areas of 15 congruent disks covering a unit circle --/
theorem sum_area_15_disks_on_unit_circle : 
  ∃ (r : ℝ), 
    0 < r ∧ 
    (15 : ℝ) * (2 * r) = 2 * π ∧ 
    15 * (π * r^2) = π * (105 - 60 * Real.sqrt 3) := by
  sorry

end sum_area_15_disks_on_unit_circle_l734_73493


namespace king_hearts_diamonds_probability_l734_73480

/-- The number of cards in a double deck -/
def total_cards : ℕ := 104

/-- The number of King of Hearts and King of Diamonds cards in a double deck -/
def target_cards : ℕ := 4

/-- The probability of drawing a King of Hearts or King of Diamonds from a shuffled double deck -/
def probability : ℚ := target_cards / total_cards

theorem king_hearts_diamonds_probability :
  probability = 1 / 26 := by sorry

end king_hearts_diamonds_probability_l734_73480
