import Mathlib

namespace complex_power_sum_l2674_267471

open Complex

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + (1 / z)^1500 = -Real.sqrt 3 := by
  sorry

end complex_power_sum_l2674_267471


namespace cube_sum_equals_negative_eighteen_l2674_267406

theorem cube_sum_equals_negative_eighteen
  (a b c : ℝ)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 :=
by sorry

end cube_sum_equals_negative_eighteen_l2674_267406


namespace opposite_sides_line_range_l2674_267453

/-- Given that the points (-2, 1) and (1, 1) are on opposite sides of the line 3x-2y-a=0,
    prove that the range of values for a is -8 < a < 1 -/
theorem opposite_sides_line_range (a : ℝ) : 
  (∀ (x y : ℝ), 3*x - 2*y - a = 0 → 
    ((3*(-2) - 2*1 - a) * (3*1 - 2*1 - a) < 0)) → 
  -8 < a ∧ a < 1 := by
  sorry

end opposite_sides_line_range_l2674_267453


namespace gcf_2835_9150_l2674_267439

theorem gcf_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end gcf_2835_9150_l2674_267439


namespace last_number_theorem_l2674_267469

theorem last_number_theorem (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : (b + c + d) / 3 = 5)
  (h3 : a + d = 11) :
  d = 4 := by
sorry

end last_number_theorem_l2674_267469


namespace angle_C_value_l2674_267444

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)

-- Define the theorem
theorem angle_C_value (t : Triangle) 
  (h1 : t.b = Real.sqrt 2)
  (h2 : t.c = 1)
  (h3 : t.B = π / 4) : 
  t.C = π / 6 := by
  sorry

end angle_C_value_l2674_267444


namespace base_conversion_problem_l2674_267491

theorem base_conversion_problem (b : ℕ+) : 
  (b : ℝ)^5 ≤ 125 ∧ 125 < (b : ℝ)^6 ↔ b = 2 :=
sorry

end base_conversion_problem_l2674_267491


namespace measurable_eq_set_l2674_267401

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (F : MeasurableSpace Ω)
variable (ξ η : Ω → ℝ)

theorem measurable_eq_set (hξ : Measurable ξ) (hη : Measurable η) :
  MeasurableSet {ω | ξ ω = η ω} :=
by
  sorry

end measurable_eq_set_l2674_267401


namespace rachel_chocolate_sales_l2674_267414

theorem rachel_chocolate_sales (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : 
  total_bars = 13 → price_per_bar = 2 → unsold_bars = 4 → 
  (total_bars - unsold_bars) * price_per_bar = 18 := by
sorry

end rachel_chocolate_sales_l2674_267414


namespace same_weaving_rate_first_group_weavers_count_l2674_267467

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 14

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 49

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 14

/-- The rate of weaving is the same for both groups -/
theorem same_weaving_rate :
  (first_group_mats : ℚ) / (first_group_days * first_group_weavers) =
  (second_group_mats : ℚ) / (second_group_days * second_group_weavers) :=
sorry

/-- The number of weavers in the first group is 4 -/
theorem first_group_weavers_count :
  first_group_weavers = 4 :=
sorry

end same_weaving_rate_first_group_weavers_count_l2674_267467


namespace right_triangle_area_l2674_267475

/-- The area of a right triangle with hypotenuse 12 inches and one angle of 30° is 18√3 square inches. -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 → θ = 30 * π / 180 → area = 18 * Real.sqrt 3 → 
  area = (1/2) * h * h * Real.sin θ * Real.cos θ :=
sorry

end right_triangle_area_l2674_267475


namespace congruence_solution_l2674_267400

theorem congruence_solution (y : ℤ) : 
  (10 * y + 3) % 18 = 7 % 18 → y % 9 = 4 % 9 := by
sorry

end congruence_solution_l2674_267400


namespace tangent_line_equation_l2674_267443

noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x^3 - 1)

theorem tangent_line_equation :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := deriv f p.1
  (λ (x y : ℝ) ↦ m * (x - p.1) - (y - p.2)) = (λ (x y : ℝ) ↦ 3 * x + y - 3) :=
by sorry

end tangent_line_equation_l2674_267443


namespace fifth_week_consumption_l2674_267434

/-- Represents the vegetable consumption for a day -/
structure VegetableConsumption where
  asparagus : Float
  broccoli : Float
  cauliflower : Float
  spinach : Float
  kale : Float
  zucchini : Float
  carrots : Float

/-- Calculates the total vegetable consumption for a day -/
def totalConsumption (vc : VegetableConsumption) : Float :=
  vc.asparagus + vc.broccoli + vc.cauliflower + vc.spinach + vc.kale + vc.zucchini + vc.carrots

/-- Initial weekday consumption -/
def initialWeekday : VegetableConsumption := {
  asparagus := 0.25, broccoli := 0.25, cauliflower := 0.5,
  spinach := 0, kale := 0, zucchini := 0, carrots := 0
}

/-- Initial weekend consumption -/
def initialWeekend : VegetableConsumption := {
  asparagus := 0.3, broccoli := 0.4, cauliflower := 0.6,
  spinach := 0, kale := 0, zucchini := 0, carrots := 0
}

/-- Updated weekday consumption -/
def updatedWeekday : VegetableConsumption := {
  asparagus := initialWeekday.asparagus * 2,
  broccoli := initialWeekday.broccoli * 3,
  cauliflower := initialWeekday.cauliflower * 1.75,
  spinach := 0.5,
  kale := 0, zucchini := 0, carrots := 0
}

/-- Updated Saturday consumption -/
def updatedSaturday : VegetableConsumption := {
  asparagus := initialWeekend.asparagus,
  broccoli := initialWeekend.broccoli,
  cauliflower := initialWeekend.cauliflower,
  spinach := 0,
  kale := 1,
  zucchini := 0.3,
  carrots := 0
}

/-- Updated Sunday consumption -/
def updatedSunday : VegetableConsumption := {
  asparagus := initialWeekend.asparagus,
  broccoli := initialWeekend.broccoli,
  cauliflower := initialWeekend.cauliflower,
  spinach := 0,
  kale := 0,
  zucchini := 0,
  carrots := 0.5
}

/-- Theorem: The total vegetable consumption in the fifth week is 17.225 pounds -/
theorem fifth_week_consumption :
  5 * totalConsumption updatedWeekday +
  totalConsumption updatedSaturday +
  totalConsumption updatedSunday = 17.225 := by
  sorry

end fifth_week_consumption_l2674_267434


namespace sibling_pairs_count_l2674_267427

theorem sibling_pairs_count 
  (business_students : ℕ) 
  (law_students : ℕ) 
  (sibling_pair_probability : ℝ) 
  (h1 : business_students = 500) 
  (h2 : law_students = 800) 
  (h3 : sibling_pair_probability = 7.500000000000001e-05) : 
  ℕ := 
by
  sorry

#check sibling_pairs_count

end sibling_pairs_count_l2674_267427


namespace power_of_point_theorem_l2674_267426

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a plane -/
def Point := ℝ × ℝ

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Determine if a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

theorem power_of_point_theorem 
  (c : Circle) (A B C D E : Point) 
  (hB : onCircle B c) (hC : onCircle C c) (hD : onCircle D c) (hE : onCircle E c)
  (hAB : distance A B = 7)
  (hBC : distance B C = 7)
  (hAD : distance A D = 10) :
  distance D E = 0.2 := by sorry

end power_of_point_theorem_l2674_267426


namespace shuffleboard_games_l2674_267487

/-- The number of games won by Jerry -/
def jerry_wins : ℕ := 7

/-- The number of games won by Dave -/
def dave_wins : ℕ := jerry_wins + 3

/-- The number of games won by Ken -/
def ken_wins : ℕ := dave_wins + 5

/-- The number of games won by Larry -/
def larry_wins : ℕ := 2 * jerry_wins

/-- The total number of ties -/
def total_ties : ℕ := jerry_wins

/-- The total number of games played -/
def total_games : ℕ := ken_wins + dave_wins + jerry_wins + larry_wins + total_ties

theorem shuffleboard_games :
  (∀ player : ℕ, player ∈ [ken_wins, dave_wins, jerry_wins, larry_wins] → player ≥ 5) →
  total_games = 53 := by
  sorry

end shuffleboard_games_l2674_267487


namespace pizza_eating_l2674_267433

theorem pizza_eating (n : ℕ) (initial_pizza : ℚ) : 
  initial_pizza = 1 →
  (let eat_fraction := 1/3
   let remaining_fraction := 1 - eat_fraction
   let total_eaten := (1 - remaining_fraction^n) / (1 - remaining_fraction)
   n = 6 →
   total_eaten = 665/729) := by
sorry

end pizza_eating_l2674_267433


namespace equation_solution_l2674_267449

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 := by
  sorry

end equation_solution_l2674_267449


namespace white_triangle_pairs_count_l2674_267422

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : Nat
  blue_blue : Nat
  red_white : Nat
  white_white : Nat

/-- The main theorem statement -/
theorem white_triangle_pairs_count 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.red = 3)
  (h2 : counts.blue = 5)
  (h3 : counts.white = 8)
  (h4 : pairs.red_red = 2)
  (h5 : pairs.blue_blue = 3)
  (h6 : pairs.red_white = 2) :
  pairs.white_white = 5 := by
  sorry

end white_triangle_pairs_count_l2674_267422


namespace sin_285_degrees_l2674_267497

theorem sin_285_degrees : 
  Real.sin (285 * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end sin_285_degrees_l2674_267497


namespace factory_scrap_rate_l2674_267478

/-- The overall scrap rate of a factory with two machines -/
def overall_scrap_rate (output_a output_b scrap_rate_a scrap_rate_b : ℝ) : ℝ :=
  output_a * scrap_rate_a + output_b * scrap_rate_b

theorem factory_scrap_rate :
  overall_scrap_rate 0.45 0.55 0.02 0.03 = 0.0255 := by
  sorry

end factory_scrap_rate_l2674_267478


namespace input_is_input_command_l2674_267445

-- Define the type for programming commands
inductive ProgrammingCommand
  | PRINT
  | INPUT
  | THEN
  | END

-- Define a function to check if a command is used for input
def isInputCommand (cmd : ProgrammingCommand) : Prop :=
  match cmd with
  | ProgrammingCommand.INPUT => True
  | _ => False

-- Theorem: INPUT is the only command used for receiving user input
theorem input_is_input_command :
  ∀ (cmd : ProgrammingCommand),
    isInputCommand cmd ↔ cmd = ProgrammingCommand.INPUT :=
  sorry

end input_is_input_command_l2674_267445


namespace opposite_silver_is_yellow_l2674_267437

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Silver | Violet

-- Define the faces of the cube
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube as a function from Face to Color
def Cube := Face → Color

-- Define the three views
def view1 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Yellow ∧ c Face.Right = Color.Orange

def view2 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Black ∧ c Face.Right = Color.Orange

def view3 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Violet ∧ c Face.Right = Color.Orange

-- Define the theorem
theorem opposite_silver_is_yellow (c : Cube) :
  view1 c → view2 c → view3 c →
  (∃ f : Face, c f = Color.Silver) →
  (∃ f : Face, c f = Color.Yellow) →
  c Face.Front = Color.Yellow :=
by sorry

end opposite_silver_is_yellow_l2674_267437


namespace ferris_wheel_capacity_l2674_267472

/-- The number of people that can be seated in one seat of the Ferris wheel -/
def people_per_seat : ℕ := 9

/-- The number of seats on the Ferris wheel -/
def number_of_seats : ℕ := 2

/-- The total number of people that can ride the Ferris wheel at the same time -/
def total_riders : ℕ := people_per_seat * number_of_seats

theorem ferris_wheel_capacity : total_riders = 18 := by
  sorry

end ferris_wheel_capacity_l2674_267472


namespace find_y_l2674_267496

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(3*y - 1) = 8) (h2 : x = 2) : y = 4/3 := by
  sorry

end find_y_l2674_267496


namespace linear_function_increasing_l2674_267452

/-- Given a linear function f(x) = (k^2 + 1)x - 5, prove that f(-3) < f(4) for any real k -/
theorem linear_function_increasing (k : ℝ) :
  let f : ℝ → ℝ := λ x => (k^2 + 1) * x - 5
  f (-3) < f 4 := by sorry

end linear_function_increasing_l2674_267452


namespace cube_volume_problem_l2674_267420

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a + 2) * (a + 1) * (a - 1) = a^3 - 6 → 
  a^3 = 8 := by
sorry

end cube_volume_problem_l2674_267420


namespace repeating_decimal_35_equals_fraction_l2674_267438

/-- The repeating decimal 0.3535... is equal to 35/99 -/
theorem repeating_decimal_35_equals_fraction : ∃ (x : ℚ), x = 35 / 99 ∧ 100 * x - x = 35 := by
  sorry

end repeating_decimal_35_equals_fraction_l2674_267438


namespace notebook_purchase_problem_l2674_267481

theorem notebook_purchase_problem :
  ∀ (price_A price_B : ℝ) (quantity_A quantity_B : ℕ),
  -- Conditions
  (price_B = price_A + 1) →
  (110 / price_A = 120 / price_B) →
  (quantity_A + quantity_B = 100) →
  (quantity_B ≤ 3 * quantity_A) →
  -- Conclusions
  (price_A = 11) ∧
  (price_B = 12) ∧
  (∀ (total_cost : ℝ),
    total_cost = price_A * quantity_A + price_B * quantity_B →
    total_cost ≥ 1100) :=
by sorry

end notebook_purchase_problem_l2674_267481


namespace wilson_hamburgers_l2674_267474

/-- The number of hamburgers Wilson bought -/
def num_hamburgers : ℕ := 2

/-- The price of each hamburger in dollars -/
def hamburger_price : ℕ := 5

/-- The number of cola bottles -/
def num_cola : ℕ := 3

/-- The price of each cola bottle in dollars -/
def cola_price : ℕ := 2

/-- The discount amount in dollars -/
def discount : ℕ := 4

/-- The total amount Wilson paid in dollars -/
def total_paid : ℕ := 12

theorem wilson_hamburgers :
  num_hamburgers * hamburger_price + num_cola * cola_price - discount = total_paid :=
by sorry

end wilson_hamburgers_l2674_267474


namespace no_integer_solution_l2674_267403

theorem no_integer_solution : ¬∃ (m n : ℤ), m^2 = n^2 + 1954 := by
  sorry

end no_integer_solution_l2674_267403


namespace sam_dimes_l2674_267431

/-- The number of dimes Sam has after receiving more from his dad -/
def total_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Proof that Sam has 16 dimes after receiving more from his dad -/
theorem sam_dimes : total_dimes 9 7 = 16 := by
  sorry

end sam_dimes_l2674_267431


namespace sum_of_w_and_z_l2674_267416

theorem sum_of_w_and_z (w x y z : ℤ) 
  (eq1 : w + x = 45)
  (eq2 : x + y = 51)
  (eq3 : y + z = 28) :
  w + z = 22 := by
  sorry

end sum_of_w_and_z_l2674_267416


namespace largest_consecutive_nonprime_less_than_30_l2674_267440

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem largest_consecutive_nonprime_less_than_30 :
  ∃ (n : ℕ),
    isTwoDigit n ∧
    n < 30 ∧
    (∀ k : ℕ, k ∈ List.range 5 → ¬isPrime (n - k)) ∧
    (∀ m : ℕ, m > n → ¬(isTwoDigit m ∧ m < 30 ∧ (∀ k : ℕ, k ∈ List.range 5 → ¬isPrime (m - k)))) ∧
    n = 28 := by sorry

end largest_consecutive_nonprime_less_than_30_l2674_267440


namespace cubic_quadratic_system_solution_l2674_267419

theorem cubic_quadratic_system_solution :
  ∀ a b c : ℕ,
    a^3 - b^3 - c^3 = 3*a*b*c →
    a^2 = 2*(a + b + c) →
    ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end cubic_quadratic_system_solution_l2674_267419


namespace monomial_properties_l2674_267409

/-- Represents a monomial with coefficient and exponents for variables a, b, and c -/
structure Monomial where
  coeff : ℤ
  a_exp : ℕ
  b_exp : ℕ
  c_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.a_exp + m.b_exp + m.c_exp

/-- The given monomial -7a³b⁴c -/
def given_monomial : Monomial :=
  { coeff := -7
    a_exp := 3
    b_exp := 4
    c_exp := 1 }

theorem monomial_properties :
  given_monomial.coeff = -7 ∧ degree given_monomial = 8 := by
  sorry

end monomial_properties_l2674_267409


namespace k_range_theorem_l2674_267428

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

/-- A function f has a maximum value of a and a minimum value of b on the interval [0, k] -/
def HasMaxMinOn (f : ℝ → ℝ) (a b k : ℝ) : Prop :=
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ k → f x ≤ a) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ f x = a) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ k → b ≤ f x) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ f x = b)

theorem k_range_theorem (k : ℝ) :
  let p := IsIncreasing (λ x : ℝ => k * x + 1)
  let q := HasMaxMinOn (λ x : ℝ => x^2 - 2*x + 3) 3 2 k
  (¬(p ∧ q)) ∧ (p ∨ q) → k ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
by sorry

end k_range_theorem_l2674_267428


namespace prob_two_consecutive_accurate_value_l2674_267488

/-- The accuracy rate of the weather forecast for each day -/
def accuracy_rate : ℝ := 0.8

/-- The probability of having at least two consecutive days with accurate forecasts
    out of three days, given the accuracy rate for each day -/
def prob_two_consecutive_accurate (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

/-- Theorem stating that the probability of having at least two consecutive days
    with accurate forecasts out of three days, given an accuracy rate of 0.8,
    is equal to 0.768 -/
theorem prob_two_consecutive_accurate_value :
  prob_two_consecutive_accurate accuracy_rate = 0.768 := by
  sorry


end prob_two_consecutive_accurate_value_l2674_267488


namespace range_of_a_l2674_267466

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on [0, +∞) if f(x) ≥ f(y) for all 0 ≤ x ≤ y -/
def IsDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≥ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
    (h_even : IsEven f)
    (h_decreasing : IsDecreasingOnNonnegative f)
    (h_inequality : ∀ x, x ∈ Set.Ici 1 ∩ Set.Iio 3 → 
      f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) :
    a ∈ Set.Icc (Real.exp (-1)) ((2 + Real.log 3) / 3) := by
  sorry


end range_of_a_l2674_267466


namespace lateral_surface_area_of_rotated_square_l2674_267468

theorem lateral_surface_area_of_rotated_square (Q : ℝ) (h : Q > 0) :
  let side_length := Real.sqrt Q
  let radius := side_length
  let height := side_length
  let lateral_surface_area := 2 * Real.pi * radius * height
  lateral_surface_area = 2 * Real.pi * Q :=
by sorry

end lateral_surface_area_of_rotated_square_l2674_267468


namespace f_three_intersections_iff_a_in_range_l2674_267417

/-- The function f(x) = √(ax + 4) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x + 4)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) / a

/-- Predicate for f and f_inv having exactly three distinct intersection points -/
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = f_inv a x₁ ∧ f a x₂ = f_inv a x₂ ∧ f a x₃ = f_inv a x₃ ∧
    ∀ x : ℝ, f a x = f_inv a x → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem f_three_intersections_iff_a_in_range (a : ℝ) :
  a ≠ 0 → (has_three_intersections a ↔ -4 * Real.sqrt 3 / 3 < a ∧ a ≤ -2) :=
sorry

end f_three_intersections_iff_a_in_range_l2674_267417


namespace divisibility_equivalence_l2674_267430

theorem divisibility_equivalence (a b : ℤ) : 
  (13 ∣ (2*a + 3*b)) ↔ (13 ∣ (2*b - 3*a)) := by
  sorry

end divisibility_equivalence_l2674_267430


namespace daily_step_goal_l2674_267486

def sunday_steps : ℕ := 9400
def monday_steps : ℕ := 9100
def tuesday_steps : ℕ := 8300
def wednesday_steps : ℕ := 9200
def thursday_steps : ℕ := 8900
def friday_saturday_avg : ℕ := 9050
def days_in_week : ℕ := 7

theorem daily_step_goal :
  (sunday_steps + monday_steps + tuesday_steps + wednesday_steps + thursday_steps + 
   2 * friday_saturday_avg) / days_in_week = 9000 := by
  sorry

end daily_step_goal_l2674_267486


namespace derivative_problems_l2674_267448

open Real

theorem derivative_problems :
  (∀ x : ℝ, x > 0 → deriv (λ x => x * log x) x = log x + 1) ∧
  (∀ x : ℝ, x ≠ 0 → deriv (λ x => sin x / x) x = (x * cos x - sin x) / x^2) :=
by sorry

end derivative_problems_l2674_267448


namespace polynomial_equality_l2674_267485

-- Define the polynomial Q
def Q (a b c d : ℝ) (x : ℝ) : ℝ := a + b * x + c * x^2 + d * x^3

-- State the theorem
theorem polynomial_equality (a b c d : ℝ) :
  (Q a b c d (-1) = 2) →
  (∀ x, Q a b c d x = 2 + x^2 - x^3) :=
by
  sorry

end polynomial_equality_l2674_267485


namespace cubic_sum_values_l2674_267456

def N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z],
    ![y, z, x],
    ![z, x, y]]

theorem cubic_sum_values (x y z : ℂ) :
  N x y z ^ 2 = 1 →
  x * y * z = 2 →
  x^3 + y^3 + z^3 = 5 ∨ x^3 + y^3 + z^3 = 7 := by
  sorry

end cubic_sum_values_l2674_267456


namespace square_intersection_perimeter_ratio_l2674_267404

/-- Given a square with vertices at (-2b, -2b), (2b, -2b), (-2b, 2b), and (2b, 2b),
    intersected by the line y = bx, the ratio of the perimeter of one of the
    resulting quadrilaterals to b is equal to 12 + 4√2. -/
theorem square_intersection_perimeter_ratio (b : ℝ) (b_pos : b > 0) :
  let square_vertices := [(-2*b, -2*b), (2*b, -2*b), (-2*b, 2*b), (2*b, 2*b)]
  let intersecting_line := fun x => b * x
  let quadrilateral_perimeter := 12 * b + 4 * b * Real.sqrt 2
  quadrilateral_perimeter / b = 12 + 4 * Real.sqrt 2 :=
by sorry

end square_intersection_perimeter_ratio_l2674_267404


namespace total_sections_after_admission_l2674_267451

/-- Proves that the total number of sections after admitting new students is 16 -/
theorem total_sections_after_admission (
  initial_students_per_section : ℕ) 
  (new_sections : ℕ)
  (students_per_section_after : ℕ)
  (new_students : ℕ)
  (h1 : initial_students_per_section = 24)
  (h2 : new_sections = 3)
  (h3 : students_per_section_after = 21)
  (h4 : new_students = 24) :
  ∃ (initial_sections : ℕ),
    (initial_sections + new_sections) * students_per_section_after = 
    initial_sections * initial_students_per_section + new_students ∧
    initial_sections + new_sections = 16 := by
  sorry

end total_sections_after_admission_l2674_267451


namespace y_satisfies_conditions_l2674_267405

/-- The function we want to prove satisfies the given conditions -/
def y (t : ℝ) : ℝ := t^3 - t^2 + t + 19

/-- The derivative of y(t) -/
def y_derivative (t : ℝ) : ℝ := 3*t^2 - 2*t + 1

theorem y_satisfies_conditions :
  (∀ t, (deriv y) t = y_derivative t) ∧ y 2 = 25 := by
  sorry

end y_satisfies_conditions_l2674_267405


namespace bus_seating_capacity_l2674_267464

theorem bus_seating_capacity : ∀ (x : ℕ),
  (4 * x + 30 = 5 * x - 10) → x = 40 := by
  sorry

end bus_seating_capacity_l2674_267464


namespace infinite_essentially_different_solutions_l2674_267461

/-- Two integer triples are essentially different if they are not scalar multiples of each other -/
def EssentiallyDifferent (a b c a₁ b₁ c₁ : ℤ) : Prop :=
  ∀ r : ℚ, ¬(a₁ = r * a ∧ b₁ = r * b ∧ c₁ = r * c)

/-- The set of solutions for the equation x^2 = y^2 + k·z^2 -/
def SolutionSet (k : ℕ) : Set (ℤ × ℤ × ℤ) :=
  {(x, y, z) | x^2 = y^2 + k * z^2}

/-- The theorem stating that there are infinitely many essentially different solutions -/
theorem infinite_essentially_different_solutions (k : ℕ) :
  ∃ S : Set (ℤ × ℤ × ℤ),
    (∀ (x y z : ℤ), (x, y, z) ∈ S → (x, y, z) ∈ SolutionSet k) ∧
    (∀ (x y z x₁ y₁ z₁ : ℤ), (x, y, z) ∈ S → (x₁, y₁, z₁) ∈ S → (x, y, z) ≠ (x₁, y₁, z₁) →
      EssentiallyDifferent x y z x₁ y₁ z₁) ∧
    Set.Infinite S :=
  sorry

end infinite_essentially_different_solutions_l2674_267461


namespace problem_1_problem_2_l2674_267465

theorem problem_1 (a b : ℚ) (h1 : a = -1/2) (h2 : b = -1) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3/4 := by
  sorry

theorem problem_2 (x y : ℝ) (h : |2*x - 1| + (3*y + 2)^2 = 0) :
  5 * x^2 - (2*x*y - 3 * (1/3 * x*y + 2) + 5 * x^2) = 19/3 := by
  sorry

end problem_1_problem_2_l2674_267465


namespace parallel_vectors_m_value_l2674_267425

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (3, 2)
  are_parallel a b → m = 3/2 := by
sorry

end parallel_vectors_m_value_l2674_267425


namespace fraction_equality_l2674_267489

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39/37 := by
  sorry

end fraction_equality_l2674_267489


namespace salt_solution_volume_l2674_267463

/-- Proves that for a salt solution with given conditions, the initial volume is 56 gallons -/
theorem salt_solution_volume 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_concentration = 0.10)
  (h2 : final_concentration = 0.08)
  (h3 : added_water = 14) :
  ∃ (initial_volume : ℝ), 
    initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration ∧ 
    initial_volume = 56 := by
  sorry

end salt_solution_volume_l2674_267463


namespace radical_equality_l2674_267458

theorem radical_equality (a b c : ℕ+) :
  Real.sqrt (a * (b + c)) = a * Real.sqrt (b + c) ↔ a = 1 := by sorry

end radical_equality_l2674_267458


namespace sum_of_digits_square_999999999_l2674_267436

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def repeated_nines (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem sum_of_digits_square_999999999 :
  digit_sum ((repeated_nines 9)^2) = 81 := by sorry

end sum_of_digits_square_999999999_l2674_267436


namespace total_arrangements_eq_5760_l2674_267457

/-- The number of ways to arrange n distinct objects taken k at a time -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The total number of students -/
def total_students : ℕ := 8

/-- The number of students in each row -/
def students_per_row : ℕ := 4

/-- The number of students with fixed positions (A, B, and C) -/
def fixed_students : ℕ := 3

/-- The number of ways to arrange A and B in the front row -/
def front_row_arrangements : ℕ := arrangements students_per_row 2

/-- The number of ways to arrange C in the back row -/
def back_row_arrangements : ℕ := arrangements students_per_row 1

/-- The number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := arrangements (total_students - fixed_students) (total_students - fixed_students)

theorem total_arrangements_eq_5760 :
  front_row_arrangements * back_row_arrangements * remaining_arrangements = 5760 := by
  sorry

end total_arrangements_eq_5760_l2674_267457


namespace base4_132_is_30_l2674_267484

/-- Converts a number from base 4 to decimal --/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The decimal representation of 132 in base 4 --/
def m : Nat := base4ToDecimal [2, 3, 1]

theorem base4_132_is_30 : m = 30 := by
  sorry

end base4_132_is_30_l2674_267484


namespace min_a_bound_l2674_267402

theorem min_a_bound (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) ↔ a ≥ 1/5 := by
  sorry

end min_a_bound_l2674_267402


namespace kayla_kimiko_age_ratio_l2674_267410

/-- Proves that the ratio of Kayla's age to Kimiko's age is 1:2 -/
theorem kayla_kimiko_age_ratio :
  let kimiko_age : ℕ := 26
  let min_driving_age : ℕ := 18
  let years_until_driving : ℕ := 5
  let kayla_age : ℕ := min_driving_age - years_until_driving
  (kayla_age : ℚ) / kimiko_age = 1 / 2 := by
  sorry

end kayla_kimiko_age_ratio_l2674_267410


namespace frozen_fruit_sold_l2674_267483

/-- Given an orchard's fruit sales, calculate the amount of frozen fruit sold. -/
theorem frozen_fruit_sold (total_fruit : ℕ) (fresh_fruit : ℕ) (h1 : total_fruit = 9792) (h2 : fresh_fruit = 6279) :
  total_fruit - fresh_fruit = 3513 := by
  sorry

end frozen_fruit_sold_l2674_267483


namespace katy_made_65_brownies_l2674_267421

/-- The number of brownies Katy made and ate over four days --/
def brownies_problem (total : ℕ) : Prop :=
  ∃ (mon tue wed thu_before thu_after : ℕ),
    -- Monday's consumption
    mon = 5 ∧
    -- Tuesday's consumption
    tue = 2 * mon ∧
    -- Wednesday's consumption
    wed = 3 * tue ∧
    -- Remaining brownies before sharing on Thursday
    thu_before = total - (mon + tue + wed) ∧
    -- Brownies left after sharing on Thursday
    thu_after = thu_before / 2 ∧
    -- Brownies left after sharing equals Tuesday's consumption
    thu_after = tue ∧
    -- All brownies are gone after Thursday
    mon + tue + wed + thu_before = total

/-- The total number of brownies Katy made is 65 --/
theorem katy_made_65_brownies : brownies_problem 65 := by
  sorry

end katy_made_65_brownies_l2674_267421


namespace circle_center_radius_sum_l2674_267413

/-- Given a circle D with equation x^2 - 4y - 34 = -y^2 + 12x + 74,
    prove that its center (c,d) and radius r satisfy c + d + r = 4 + 2√17 -/
theorem circle_center_radius_sum (x y c d r : ℝ) : 
  (∀ x y, x^2 - 4*y - 34 = -y^2 + 12*x + 74) →
  ((x - c)^2 + (y - d)^2 = r^2) →
  (c + d + r = 4 + 2 * Real.sqrt 17) := by
  sorry


end circle_center_radius_sum_l2674_267413


namespace prove_certain_number_l2674_267415

def w : ℕ := 468
def certain_number : ℕ := 2028

theorem prove_certain_number :
  (∃ (n : ℕ), (2^4 ∣ n * w) ∧ (3^3 ∣ n * w) ∧ (13^3 ∣ n * w)) ∧
  (∀ (x : ℕ), x < w → ¬(∃ (m : ℕ), (2^4 ∣ m * x) ∧ (3^3 ∣ m * x) ∧ (13^3 ∣ m * x))) →
  certain_number * w % 2^4 = 0 ∧
  certain_number * w % 3^3 = 0 ∧
  certain_number * w % 13^3 = 0 ∧
  (∀ (y : ℕ), y < certain_number →
    (y * w % 2^4 ≠ 0 ∨ y * w % 3^3 ≠ 0 ∨ y * w % 13^3 ≠ 0)) :=
by sorry

end prove_certain_number_l2674_267415


namespace store_purchase_count_l2674_267476

def num_cookie_flavors : ℕ := 6
def num_milk_flavors : ℕ := 4

def gamma_purchase_options (n : ℕ) : ℕ :=
  Nat.choose (num_cookie_flavors + num_milk_flavors) n

def delta_cookie_options (n : ℕ) : ℕ :=
  if n = 1 then
    num_cookie_flavors
  else if n = 2 then
    Nat.choose num_cookie_flavors 2 + num_cookie_flavors
  else if n = 3 then
    Nat.choose num_cookie_flavors 3 + num_cookie_flavors * (num_cookie_flavors - 1) + num_cookie_flavors
  else
    0

def total_purchase_options : ℕ :=
  gamma_purchase_options 3 +
  gamma_purchase_options 2 * delta_cookie_options 1 +
  gamma_purchase_options 1 * delta_cookie_options 2 +
  delta_cookie_options 3

theorem store_purchase_count : total_purchase_options = 656 := by
  sorry

end store_purchase_count_l2674_267476


namespace locus_of_midpoints_l2674_267450

/-- The locus of midpoints theorem -/
theorem locus_of_midpoints 
  (A : ℝ × ℝ) 
  (h_A : A = (4, -2))
  (B : ℝ × ℝ → Prop)
  (h_B : ∀ x y, B (x, y) ↔ x^2 + y^2 = 4)
  (P : ℝ × ℝ)
  (h_P : ∃ x y, B (x, y) ∧ P = ((A.1 + x) / 2, (A.2 + y) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 := by
  sorry

end locus_of_midpoints_l2674_267450


namespace value_of_a_l2674_267495

/-- Two circles centered at the origin with given properties -/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ → ℝ × ℝ
  -- Distance between Q and R on x-axis
  QR_distance : ℝ
  -- Conditions
  center_origin : True
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : ∀ a, (S a).1^2 + (S a).2^2 = r^2
  S_on_diagonal : ∀ a, (S a).1 = (S a).2
  QR_is_4 : QR_distance = 4
  R_eq_sqrt_104 : R = Real.sqrt 104
  r_eq_R_minus_4 : r = R - 4

/-- The theorem stating the value of a -/
theorem value_of_a (c : TwoCircles) : 
  ∃ a, c.S a = (a, a) ∧ a = Real.sqrt (60 - 4 * Real.sqrt 104) :=
sorry

end value_of_a_l2674_267495


namespace lcm_three_consecutive_naturals_l2674_267480

theorem lcm_three_consecutive_naturals (n : ℕ) :
  let lcm := Nat.lcm (Nat.lcm n (n + 1)) (n + 2)
  lcm = if Even (n + 1) then n * (n + 1) * (n + 2)
        else (n * (n + 1) * (n + 2)) / 2 := by
  sorry

end lcm_three_consecutive_naturals_l2674_267480


namespace cubic_properties_l2674_267460

theorem cubic_properties :
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x < 1 → x^3 < x) :=
by sorry

end cubic_properties_l2674_267460


namespace bag_problem_l2674_267411

theorem bag_problem (total_slips : ℕ) (value1 value2 : ℝ) (expected_value : ℝ) :
  total_slips = 12 →
  value1 = 2 →
  value2 = 7 →
  expected_value = 3.25 →
  ∃ (slips_with_value1 : ℕ),
    slips_with_value1 ≤ total_slips ∧
    (slips_with_value1 : ℝ) / total_slips * value1 +
    ((total_slips - slips_with_value1) : ℝ) / total_slips * value2 = expected_value ∧
    slips_with_value1 = 9 :=
by sorry

end bag_problem_l2674_267411


namespace tablet_consumption_time_l2674_267418

theorem tablet_consumption_time (num_tablets : ℕ) (interval : ℕ) : num_tablets = 10 ∧ interval = 25 → (num_tablets - 1) * interval = 225 := by
  sorry

end tablet_consumption_time_l2674_267418


namespace arithmetic_mean_inequality_negative_l2674_267455

theorem arithmetic_mean_inequality_negative (m n : ℝ) (h1 : m < n) (h2 : n < 0) : n / m + m / n > 2 := by
  sorry

end arithmetic_mean_inequality_negative_l2674_267455


namespace garden_area_this_year_l2674_267432

/-- Represents the garden and its contents over two years --/
structure Garden where
  cabbage_area : ℝ  -- Area taken by one cabbage
  tomato_area : ℝ   -- Area taken by one tomato plant
  last_year_cabbages : ℕ
  last_year_tomatoes : ℕ
  cabbage_increase : ℕ
  tomato_decrease : ℕ

/-- Calculates the total area of the garden --/
def garden_area (g : Garden) : ℝ :=
  let this_year_cabbages := g.last_year_cabbages + g.cabbage_increase
  let this_year_tomatoes := max (g.last_year_tomatoes - g.tomato_decrease) 0
  g.cabbage_area * this_year_cabbages + g.tomato_area * this_year_tomatoes

/-- The theorem stating the area of the garden this year --/
theorem garden_area_this_year (g : Garden) 
  (h1 : g.cabbage_area = 1)
  (h2 : g.tomato_area = 0.5)
  (h3 : g.last_year_cabbages = 72)
  (h4 : g.last_year_tomatoes = 36)
  (h5 : g.cabbage_increase = 193)
  (h6 : g.tomato_decrease = 50) :
  garden_area g = 265 := by
  sorry

#eval garden_area { 
  cabbage_area := 1, 
  tomato_area := 0.5, 
  last_year_cabbages := 72, 
  last_year_tomatoes := 36, 
  cabbage_increase := 193, 
  tomato_decrease := 50 
}

end garden_area_this_year_l2674_267432


namespace bridge_building_time_l2674_267424

theorem bridge_building_time
  (workers₁ workers₂ : ℕ)
  (days₁ : ℕ)
  (h_workers₁ : workers₁ = 60)
  (h_workers₂ : workers₂ = 30)
  (h_days₁ : days₁ = 6)
  (h_positive : workers₁ > 0 ∧ workers₂ > 0 ∧ days₁ > 0)
  (h_same_rate : ∀ w : ℕ, w > 0 → ∃ r : ℚ, r > 0 ∧ w * r * days₁ = 1) :
  ∃ days₂ : ℕ, days₂ = 12 ∧ workers₂ * days₂ = workers₁ * days₁ :=
by sorry


end bridge_building_time_l2674_267424


namespace opposite_sides_range_l2674_267447

def line_equation (x y a : ℝ) : ℝ := x - y + a

theorem opposite_sides_range (a : ℝ) : 
  (line_equation 0 0 a) * (line_equation 1 (-1) a) < 0 ↔ -2 < a ∧ a < 0 := by
  sorry

end opposite_sides_range_l2674_267447


namespace motorcycles_in_anytown_l2674_267408

/-- Given the ratio of vehicles in Anytown and the number of sedans, 
    prove the number of motorcycles. -/
theorem motorcycles_in_anytown 
  (truck_ratio : ℕ) 
  (sedan_ratio : ℕ) 
  (motorcycle_ratio : ℕ) 
  (num_sedans : ℕ) 
  (h1 : truck_ratio = 3)
  (h2 : sedan_ratio = 7)
  (h3 : motorcycle_ratio = 2)
  (h4 : num_sedans = 9100) : 
  (num_sedans / sedan_ratio) * motorcycle_ratio = 2600 := by
  sorry

end motorcycles_in_anytown_l2674_267408


namespace line_equivalence_l2674_267470

/-- Given a line in the form (3, -7) · ((x, y) - (2, 8)) = 0, prove it's equivalent to y = (3/7)x + 50/7 -/
theorem line_equivalence (x y : ℝ) :
  (3 : ℝ) * (x - 2) + (-7 : ℝ) * (y - 8) = 0 ↔ y = (3/7)*x + 50/7 :=
by sorry

end line_equivalence_l2674_267470


namespace rectangular_box_area_product_l2674_267473

theorem rectangular_box_area_product (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  (x * z) * (x * y) * (y * z) = (x * y * z)^2 := by sorry

end rectangular_box_area_product_l2674_267473


namespace cubic_equation_roots_l2674_267412

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 38 := by
sorry

end cubic_equation_roots_l2674_267412


namespace only_odd_solution_is_one_l2674_267442

theorem only_odd_solution_is_one :
  ∀ y : ℤ, ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1 ∧ Odd y → y = 1 :=
by sorry

end only_odd_solution_is_one_l2674_267442


namespace tree_height_average_l2674_267441

def tree_heights (n : ℕ) : Type := Fin n → ℕ

def valid_heights (h : tree_heights 7) : Prop :=
  h 1 = 16 ∧
  ∀ i : Fin 6, (h i = 2 * h i.succ ∨ 2 * h i = h i.succ)

def average_height (h : tree_heights 7) : ℚ :=
  (h 0 + h 1 + h 2 + h 3 + h 4 + h 5 + h 6 : ℚ) / 7

theorem tree_height_average (h : tree_heights 7) 
  (hvalid : valid_heights h) : average_height h = 145.1 := by
  sorry

end tree_height_average_l2674_267441


namespace set_equality_l2674_267493

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l2674_267493


namespace xia_initial_stickers_l2674_267477

/-- The number of stickers Xia shared with her friends -/
def shared_stickers : ℕ := 100

/-- The number of sheets of stickers Xia had left after sharing -/
def remaining_sheets : ℕ := 5

/-- The number of stickers on each sheet -/
def stickers_per_sheet : ℕ := 10

/-- Xia's initial number of stickers -/
def initial_stickers : ℕ := shared_stickers + remaining_sheets * stickers_per_sheet

theorem xia_initial_stickers : initial_stickers = 150 := by
  sorry

end xia_initial_stickers_l2674_267477


namespace line_intersects_both_axes_l2674_267479

/-- A line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  not_both_zero : A ≠ 0 ∨ B ≠ 0

/-- Predicate for a line intersecting both coordinate axes -/
def intersects_both_axes (l : Line) : Prop :=
  ∃ x y : ℝ, (l.A * x + l.C = 0) ∧ (l.B * y + l.C = 0)

/-- Theorem stating the condition for a line to intersect both coordinate axes -/
theorem line_intersects_both_axes (l : Line) : 
  intersects_both_axes l ↔ l.A ≠ 0 ∧ l.B ≠ 0 := by
  sorry

end line_intersects_both_axes_l2674_267479


namespace number_equality_l2674_267494

theorem number_equality (x : ℝ) : (0.4 * x = 0.25 * 80) → x = 50 := by
  sorry

end number_equality_l2674_267494


namespace solution_set_f_max_integer_m_max_m_is_two_l2674_267482

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |1 - 2*x|

-- Theorem for part 1
theorem solution_set_f (x : ℝ) :
  f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem for part 2
theorem max_integer_m (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : f a = 3 * f b) :
  ∀ m : ℤ, (a^2 + b^2 > m) → m ≤ 2 :=
sorry

-- Theorem to prove that 2 is the maximum integer satisfying the condition
theorem max_m_is_two (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : f a = 3 * f b) :
  ∃ m : ℤ, (∀ n : ℤ, (a^2 + b^2 > n) → n ≤ m) ∧ m = 2 :=
sorry

end solution_set_f_max_integer_m_max_m_is_two_l2674_267482


namespace coffee_beans_remaining_l2674_267429

theorem coffee_beans_remaining (jar_weight empty_weight full_weight remaining_weight : ℝ)
  (h1 : empty_weight = 0.2 * full_weight)
  (h2 : remaining_weight = 0.6 * full_weight)
  (h3 : empty_weight > 0)
  (h4 : full_weight > empty_weight) : 
  let beans_weight := full_weight - empty_weight
  let defective_weight := 0.1 * beans_weight
  let remaining_beans := remaining_weight - empty_weight
  (remaining_beans - defective_weight) / (beans_weight - defective_weight) = 4 / 9 := by
  sorry

end coffee_beans_remaining_l2674_267429


namespace triangle_properties_l2674_267423

/-- Given a triangle ABC with specific properties, prove that A = π/3 and AB = 2 -/
theorem triangle_properties (A B C : ℝ) (AB BC AC : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.sin B * Real.cos A = Real.sin (A + C) →
  BC = 2 →
  (1/2) * AB * AC * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ AB = 2 := by
sorry

end triangle_properties_l2674_267423


namespace congruence_solution_l2674_267490

theorem congruence_solution (n : ℤ) : 11 * 21 ≡ 16 [ZMOD 43] := by sorry

end congruence_solution_l2674_267490


namespace inequality_solution_range_l2674_267446

theorem inequality_solution_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 := by
  sorry

end inequality_solution_range_l2674_267446


namespace min_value_expression_l2674_267435

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1/2) :
  a^2 + 4*a*b + 9*b^2 + 8*b*c + 3*c^2 ≥ 13.5 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1/2 ∧
    a₀^2 + 4*a₀*b₀ + 9*b₀^2 + 8*b₀*c₀ + 3*c₀^2 = 13.5 :=
by sorry

end min_value_expression_l2674_267435


namespace sum_of_special_primes_is_prime_l2674_267492

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

theorem sum_of_special_primes_is_prime :
  ∀ A B : ℕ,
    is_prime A →
    is_prime B →
    is_prime (A - B) →
    is_prime (A + B) →
    A > B →
    B = 2 →
    is_odd A →
    is_odd (A - B) →
    is_odd (A + B) →
    (∃ k : ℕ, A = (A - B) + 2*k ∧ (A + B) = A + 2*k) →
    is_prime (A + B + (A - B) + B) :=
sorry

end sum_of_special_primes_is_prime_l2674_267492


namespace smallest_number_of_eggs_smallest_number_of_eggs_is_162_l2674_267407

theorem smallest_number_of_eggs : ℕ → Prop :=
  fun n =>
    (n > 150) ∧
    (∃ c : ℕ, c > 0 ∧ n = 15 * c - 3) ∧
    (∀ m : ℕ, m > 150 ∧ (∃ d : ℕ, d > 0 ∧ m = 15 * d - 3) → m ≥ n) →
    n = 162

theorem smallest_number_of_eggs_is_162 : smallest_number_of_eggs 162 := by
  sorry

end smallest_number_of_eggs_smallest_number_of_eggs_is_162_l2674_267407


namespace megan_popsicle_consumption_l2674_267459

/-- The number of minutes between 1:00 PM and 6:20 PM -/
def time_interval : ℕ := 320

/-- The interval in minutes at which Megan eats a Popsicle -/
def popsicle_interval : ℕ := 20

/-- The number of Popsicles Megan consumes -/
def popsicles_consumed : ℕ := time_interval / popsicle_interval

theorem megan_popsicle_consumption :
  popsicles_consumed = 16 :=
by sorry

end megan_popsicle_consumption_l2674_267459


namespace same_color_probability_l2674_267462

/-- The probability of drawing balls of the same color from two bags -/
theorem same_color_probability (bagA_white bagA_red bagB_white bagB_red : ℕ) :
  bagA_white = 8 →
  bagA_red = 4 →
  bagB_white = 6 →
  bagB_red = 6 →
  (bagA_white / (bagA_white + bagA_red : ℚ)) * (bagB_white / (bagB_white + bagB_red : ℚ)) +
  (bagA_red / (bagA_white + bagA_red : ℚ)) * (bagB_red / (bagB_white + bagB_red : ℚ)) = 1/2 :=
by sorry

end same_color_probability_l2674_267462


namespace equation_solution_l2674_267454

theorem equation_solution (M N : ℕ) 
  (h1 : (4 : ℚ) / 7 = M / 63)
  (h2 : (4 : ℚ) / 7 = 84 / N) : 
  M + N = 183 := by
sorry

end equation_solution_l2674_267454


namespace compound_interest_repayment_l2674_267498

-- Define the initial loan amount in yuan
def initial_loan : ℝ := 100000

-- Define the annual interest rate
def interest_rate : ℝ := 0.07

-- Define the repayment function (in ten thousand yuan)
def repayment_amount (years : ℕ) : ℝ :=
  10 * (1 + interest_rate) ^ years

-- Define the total repayment after 5 years (in yuan)
def total_repayment_5_years : ℕ := 140255

-- Define the number of installments
def num_installments : ℕ := 5

-- Define the annual installment amount (in yuan)
def annual_installment : ℕ := 24389

theorem compound_interest_repayment :
  -- 1. Repayment function
  (∀ x : ℕ, repayment_amount x = 10 * (1 + interest_rate) ^ x) ∧
  -- 2. Total repayment after 5 years
  (repayment_amount 5 * 10000 = total_repayment_5_years) ∧
  -- 3. Annual installment calculation
  (annual_installment * (((1 + interest_rate) ^ num_installments - 1) / interest_rate) =
    total_repayment_5_years) :=
by sorry

end compound_interest_repayment_l2674_267498


namespace rice_A_more_stable_than_B_l2674_267499

/-- Represents a rice variety with its yield variance -/
structure RiceVariety where
  name : String
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when a rice variety is considered more stable than another -/
def more_stable (a b : RiceVariety) : Prop :=
  a.variance < b.variance

/-- The main theorem stating that rice variety A is more stable than B -/
theorem rice_A_more_stable_than_B (A B : RiceVariety) 
  (hA : A.name = "A" ∧ A.variance = 794)
  (hB : B.name = "B" ∧ B.variance = 958) : 
  more_stable A B := by
  sorry

end rice_A_more_stable_than_B_l2674_267499
