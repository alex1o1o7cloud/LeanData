import Mathlib

namespace hyperbola_eccentricity_range_l1314_131455

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 3/4}
  let tangent_line := {(x, y) : ℝ × ℝ | ∃ k, y = k * x ∧ k^2 = 3}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ p q : ℝ × ℝ, p ∈ tangent_line ∧ q ∈ tangent_line ∧ p ∈ hyperbola ∧ q ∈ hyperbola ∧ p ≠ q) →
  e > 2 :=
sorry

end hyperbola_eccentricity_range_l1314_131455


namespace incorrect_equation_simplification_l1314_131433

theorem incorrect_equation_simplification (x : ℝ) : 
  (1 / (x + 1) = 2 * x / (3 * x + 3) - 1) ≠ (3 = 2 * x - 3 * x + 3) :=
by sorry

end incorrect_equation_simplification_l1314_131433


namespace subtracted_value_l1314_131428

theorem subtracted_value (chosen_number : ℕ) (x : ℚ) : 
  chosen_number = 120 → (chosen_number / 6 : ℚ) - x = 5 → x = 15 := by
  sorry

end subtracted_value_l1314_131428


namespace compound_interest_approximation_l1314_131417

/-- Approximation of compound interest using Binomial Theorem -/
theorem compound_interest_approximation
  (K : ℝ) (p : ℝ) (n : ℕ) :
  let r := p / 100
  let Kn := K * (1 + r)^n
  let approx := K * (1 + n*r + (n*(n-1)/2) * r^2 + (n*(n-1)*(n-2)/6) * r^3)
  ∃ (ε : ℝ), ε > 0 ∧ |Kn - approx| < ε * Kn :=
sorry

end compound_interest_approximation_l1314_131417


namespace isosceles_triangle_conditions_l1314_131425

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that each of the following conditions implies that the triangle is isosceles. -/
theorem isosceles_triangle_conditions (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C)
  (h_angle_sum : A + B + C = Real.pi) : 
  (a * Real.cos B = b * Real.cos A → a = b ∨ b = c ∨ a = c) ∧ 
  (Real.cos B * Real.cos C = (1 - Real.cos A) / 2 → a = b ∨ b = c ∨ a = c) ∧
  (a / Real.sin B + b / Real.sin A ≤ 2 * c → a = b ∨ b = c ∨ a = c) := by
  sorry


end isosceles_triangle_conditions_l1314_131425


namespace train_length_l1314_131406

/-- The length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 82 →
  person_speed = 6 →
  passing_time = 4.499640028797696 →
  ∃ (length : ℝ), abs (length - 110) < 0.5 :=
by sorry

end train_length_l1314_131406


namespace units_digit_of_17_times_24_l1314_131469

theorem units_digit_of_17_times_24 : (17 * 24) % 10 = 8 := by
  sorry

end units_digit_of_17_times_24_l1314_131469


namespace skateboard_distance_l1314_131432

/-- The distance traveled by the skateboard in the nth second -/
def distance (n : ℕ) : ℕ := 8 + 9 * (n - 1)

/-- The total distance traveled by the skateboard after n seconds -/
def total_distance (n : ℕ) : ℕ := n * (distance 1 + distance n) / 2

theorem skateboard_distance :
  total_distance 20 = 1870 := by sorry

end skateboard_distance_l1314_131432


namespace tens_digit_equals_number_of_tens_l1314_131420

theorem tens_digit_equals_number_of_tens (n : ℕ) (h : 10 ≤ n ∧ n ≤ 999) : 
  (n / 10) % 10 = n / 10 - (n / 100) * 10 :=
sorry

end tens_digit_equals_number_of_tens_l1314_131420


namespace cosine_of_angle_l1314_131466

def a : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -3)

theorem cosine_of_angle (t : ℝ) : 
  let b : ℝ × ℝ := (3, t)
  (b.1 * c.1 + b.2 * c.2 = 0) →  -- b ⊥ c
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = Real.sqrt 2 / 10 := by
sorry

end cosine_of_angle_l1314_131466


namespace circles_symmetric_line_l1314_131499

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 7 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem circles_symmetric_line :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line_l x y :=
by sorry

end circles_symmetric_line_l1314_131499


namespace salary_raise_percentage_l1314_131492

/-- Calculates the percentage raise given the original and new salaries. -/
def percentage_raise (original : ℚ) (new : ℚ) : ℚ :=
  (new - original) / original * 100

/-- Proves that the percentage raise from $500 to $530 is 6%. -/
theorem salary_raise_percentage :
  percentage_raise 500 530 = 6 := by
  sorry

end salary_raise_percentage_l1314_131492


namespace no_cracked_seashells_l1314_131475

theorem no_cracked_seashells (tim_shells sally_shells total_shells : ℕ) 
  (h1 : tim_shells = 37)
  (h2 : sally_shells = 13)
  (h3 : total_shells = 50)
  (h4 : tim_shells + sally_shells = total_shells) :
  total_shells - (tim_shells + sally_shells) = 0 := by
  sorry

end no_cracked_seashells_l1314_131475


namespace logarithm_simplification_l1314_131444

theorem logarithm_simplification (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.log (Real.cos x * Real.tan x + 1 - 2 * Real.sin (x / 2) ^ 2) +
  Real.log (Real.sqrt 2 * Real.cos (x - Real.pi / 4)) -
  Real.log (1 + Real.sin (2 * x)) = 0 := by sorry

end logarithm_simplification_l1314_131444


namespace remaining_water_fills_glasses_l1314_131426

theorem remaining_water_fills_glasses (total_water : ℕ) (glass_5oz : ℕ) (glass_8oz : ℕ) (glass_4oz : ℕ) :
  total_water = 122 →
  glass_5oz = 6 →
  glass_8oz = 4 →
  glass_4oz * 4 = total_water - (glass_5oz * 5 + glass_8oz * 8) →
  glass_4oz = 15 := by
sorry

end remaining_water_fills_glasses_l1314_131426


namespace multiple_properties_l1314_131404

theorem multiple_properties (a b : ℤ) 
  (ha : 4 ∣ a) (hb : 8 ∣ b) : 
  (4 ∣ b) ∧ (4 ∣ (a - b)) := by
  sorry

end multiple_properties_l1314_131404


namespace unique_modular_congruence_l1314_131410

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end unique_modular_congruence_l1314_131410


namespace ursula_annual_salary_l1314_131405

/-- Calculates the annual salary given hourly wage, hours per day, and days per month -/
def annual_salary (hourly_wage : ℝ) (hours_per_day : ℝ) (days_per_month : ℝ) : ℝ :=
  hourly_wage * hours_per_day * days_per_month * 12

/-- Proves that Ursula's annual salary is $16,320 given her work conditions -/
theorem ursula_annual_salary :
  annual_salary 8.50 8 20 = 16320 := by
  sorry

end ursula_annual_salary_l1314_131405


namespace subset_cardinality_inequality_l1314_131464

theorem subset_cardinality_inequality (n m : ℕ) (A : Fin m → Finset (Fin n)) :
  (∀ i : Fin m, ¬ (30 ∣ (A i).card)) →
  (∀ i j : Fin m, i ≠ j → (30 ∣ (A i ∩ A j).card)) →
  2 * m - m / 30 ≤ 3 * n :=
by sorry

end subset_cardinality_inequality_l1314_131464


namespace machine_worked_twelve_minutes_l1314_131422

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  shirts_made_today : ℕ

/-- Calculate the number of minutes the machine worked today -/
def minutes_worked_today (machine : ShirtMachine) : ℕ :=
  machine.shirts_made_today / machine.shirts_per_minute

/-- Theorem: The machine worked for 12 minutes today -/
theorem machine_worked_twelve_minutes 
  (machine : ShirtMachine) 
  (h1 : machine.shirts_per_minute = 6)
  (h2 : machine.shirts_made_today = 72) : 
  minutes_worked_today machine = 12 := by
  sorry

#eval minutes_worked_today ⟨6, 72⟩

end machine_worked_twelve_minutes_l1314_131422


namespace conference_arrangements_l1314_131407

/-- The number of ways to arrange n distinct elements --/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct elements with k pairs having a specific order requirement --/
def arrangementsWithOrderRequirements (n : ℕ) (k : ℕ) : ℕ :=
  arrangements n / (2^k)

/-- Theorem stating that arranging 7 lecturers with 2 order requirements results in 1260 possible arrangements --/
theorem conference_arrangements : arrangementsWithOrderRequirements 7 2 = 1260 := by
  sorry

end conference_arrangements_l1314_131407


namespace quadratic_symmetry_l1314_131474

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

-- State the theorem
theorem quadratic_symmetry (a : ℝ) :
  (f a 1 = 2) → (f a (-1) = 2) := by
  sorry

end quadratic_symmetry_l1314_131474


namespace intersection_of_A_and_B_l1314_131442

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | 2*x - 3 > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 3/2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1314_131442


namespace emilys_flowers_l1314_131484

theorem emilys_flowers (flower_cost : ℕ) (total_spent : ℕ) : 
  flower_cost = 3 →
  total_spent = 12 →
  ∃ (roses daisies : ℕ), 
    roses = daisies ∧ 
    roses + daisies = total_spent / flower_cost :=
by
  sorry

end emilys_flowers_l1314_131484


namespace complex_product_modulus_l1314_131400

theorem complex_product_modulus : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_product_modulus_l1314_131400


namespace value_of_a_l1314_131462

theorem value_of_a (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = -1 := by
sorry

end value_of_a_l1314_131462


namespace bob_start_time_l1314_131424

/-- Proves that Bob started walking 1 hour after Yolanda, given the conditions of the problem. -/
theorem bob_start_time (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) (bob_distance : ℝ) :
  total_distance = 10 →
  yolanda_rate = 3 →
  bob_rate = 4 →
  bob_distance = 4 →
  ∃ (bob_start_time : ℝ),
    bob_start_time = 1 ∧
    bob_start_time * bob_rate + yolanda_rate * (bob_start_time + bob_distance / bob_rate) = total_distance :=
by sorry

end bob_start_time_l1314_131424


namespace remainder_is_18_l1314_131416

/-- A cubic polynomial p(x) with coefficients a and b. -/
def p (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + b * x + 12

/-- The theorem stating that the remainder when p(x) is divided by x-1 is 18. -/
theorem remainder_is_18 (a b : ℝ) :
  (x + 2 ∣ p a b x) → (x - 3 ∣ p a b x) → p a b 1 = 18 := by
  sorry

end remainder_is_18_l1314_131416


namespace geometry_propositions_l1314_131437

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) : 
  (p₁ ∧ p₄) ∧ ¬(p₁ ∧ p₂) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) :=
by sorry

end geometry_propositions_l1314_131437


namespace max_perfect_matchings_20gon_l1314_131495

/-- Represents a convex polygon with 2n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin (2 * n) → ℝ × ℝ

/-- Represents a triangulation of a convex polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  diagonals : Fin (2 * n - 3) → Fin (2 * n) × Fin (2 * n)

/-- Represents a perfect matching in a triangulation -/
structure PerfectMatching (n : ℕ) where
  triangulation : Triangulation n
  edges : Fin n → Fin (4 * n - 3)

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fib (n + 1) + fib n

/-- The maximum number of perfect matchings for a convex 2n-gon -/
def maxPerfectMatchings (n : ℕ) : ℕ := fib n

/-- The theorem statement -/
theorem max_perfect_matchings_20gon :
  maxPerfectMatchings 10 = 89 := by sorry

end max_perfect_matchings_20gon_l1314_131495


namespace parallel_vector_with_given_magnitude_l1314_131498

/-- Given two vectors a and b in ℝ², where a = (2,1) and b is parallel to a with magnitude 2√5,
    prove that b must be either (4,2) or (-4,-2). -/
theorem parallel_vector_with_given_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  (∃ k : ℝ, b = (k * a.1, k * a.2)) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 2 * Real.sqrt 5 →
  b = (4, 2) ∨ b = (-4, -2) := by
  sorry

#check parallel_vector_with_given_magnitude

end parallel_vector_with_given_magnitude_l1314_131498


namespace min_value_C_squared_minus_D_squared_l1314_131485

theorem min_value_C_squared_minus_D_squared :
  ∀ (x y z : ℝ), 
  x ≥ 0 → y ≥ 0 → z ≥ 0 →
  x ≤ 1 → y ≤ 2 → z ≤ 3 →
  let C := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let D := Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3)
  ∀ (C' D' : ℝ), C = C' → D = D' →
  C' ^ 2 - D' ^ 2 ≥ 36 :=
by sorry

end min_value_C_squared_minus_D_squared_l1314_131485


namespace rachel_furniture_assembly_l1314_131480

/-- The number of tables Rachel bought -/
def num_tables : ℕ := 3

theorem rachel_furniture_assembly :
  ∀ (chairs tables : ℕ) (time_per_piece total_time : ℕ),
  chairs = 7 →
  time_per_piece = 4 →
  total_time = 40 →
  total_time = time_per_piece * (chairs + tables) →
  tables = num_tables :=
by sorry

end rachel_furniture_assembly_l1314_131480


namespace inequality_theorem_l1314_131409

open Real

theorem inequality_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x, deriv f x < f x)
  (h3 : 0 < a ∧ a < 1) :
  3 * f 0 > f a ∧ f a > a * f 1 := by
  sorry

end inequality_theorem_l1314_131409


namespace speed_at_40_degrees_l1314_131403

-- Define the relationship between temperature and speed
def temperature (s : ℝ) : ℝ := 5 * s^2 + 20 * s + 15

-- Theorem statement
theorem speed_at_40_degrees : 
  ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ temperature s₁ = 40 ∧ temperature s₂ = 40 ∧ 
  ((s₁ = 1 ∧ s₂ = -5) ∨ (s₁ = -5 ∧ s₂ = 1)) := by
  sorry

end speed_at_40_degrees_l1314_131403


namespace root_order_quadratic_equations_l1314_131427

theorem root_order_quadratic_equations (m : ℝ) (a b c d : ℝ) 
  (hm : m > 0)
  (h1 : a^2 - m*a - 1 = 0)
  (h2 : b^2 - m*b - 1 = 0)
  (h3 : c^2 + m*c - 1 = 0)
  (h4 : d^2 + m*d - 1 = 0)
  (ha : a > 0)
  (hb : b < 0)
  (hc : c > 0)
  (hd : d < 0) :
  abs a > abs c ∧ abs c > abs b ∧ abs b > abs d :=
sorry

end root_order_quadratic_equations_l1314_131427


namespace mcpherson_contribution_l1314_131451

/-- Calculate Mr. McPherson's contribution to rent and expenses --/
theorem mcpherson_contribution
  (current_rent : ℝ)
  (rent_increase_rate : ℝ)
  (current_monthly_expenses : ℝ)
  (monthly_expenses_increase_rate : ℝ)
  (mrs_mcpherson_contribution_rate : ℝ)
  (h1 : current_rent = 1200)
  (h2 : rent_increase_rate = 0.05)
  (h3 : current_monthly_expenses = 100)
  (h4 : monthly_expenses_increase_rate = 0.03)
  (h5 : mrs_mcpherson_contribution_rate = 0.30) :
  ∃ (mr_mcpherson_contribution : ℝ),
    mr_mcpherson_contribution = 1747.20 ∧
    mr_mcpherson_contribution =
      (1 - mrs_mcpherson_contribution_rate) *
      (current_rent * (1 + rent_increase_rate) +
       12 * current_monthly_expenses * (1 + monthly_expenses_increase_rate)) :=
by
  sorry

end mcpherson_contribution_l1314_131451


namespace fraction_simplification_l1314_131447

theorem fraction_simplification (x : ℝ) (h : x = 10) :
  (x^6 - 100*x^3 + 2500) / (x^3 - 50) = 950 := by
  sorry

end fraction_simplification_l1314_131447


namespace moving_trips_l1314_131487

theorem moving_trips (total_time : ℕ) (fill_time : ℕ) (drive_time : ℕ) : 
  total_time = 7 * 60 ∧ fill_time = 15 ∧ drive_time = 30 →
  (total_time / (fill_time + 2 * drive_time) : ℕ) = 5 := by
sorry

end moving_trips_l1314_131487


namespace polynomial_sum_l1314_131412

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  P a b c d 1 = 2000 →
  P a b c d 2 = 4000 →
  P a b c d 3 = 6000 →
  P a b c d 9 + P a b c d (-5) = 12704 := by
  sorry

end polynomial_sum_l1314_131412


namespace simplify_and_evaluate_l1314_131443

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4)) = 5 / 13 := by
  sorry

end simplify_and_evaluate_l1314_131443


namespace jackies_tree_climbing_l1314_131401

theorem jackies_tree_climbing (h : ℝ) : 
  (1000 + 500 + 500 + h) / 4 = 800 → h - 1000 = 200 := by sorry

end jackies_tree_climbing_l1314_131401


namespace polynomial_simplification_l1314_131408

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 := by
  sorry

end polynomial_simplification_l1314_131408


namespace quadratic_minimum_l1314_131435

/-- Given a quadratic function f(x) = x^2 - 2x + m with a minimum value of -2 
    on the interval [2, +∞), prove that m = -2. -/
theorem quadratic_minimum (m : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → x^2 - 2*x + m ≥ -2) ∧ 
  (∃ x : ℝ, x ≥ 2 ∧ x^2 - 2*x + m = -2) → 
  m = -2 := by
  sorry

end quadratic_minimum_l1314_131435


namespace min_sum_of_squares_min_sum_of_squares_wire_l1314_131446

theorem min_sum_of_squares (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ 8 := by sorry

theorem min_sum_of_squares_wire :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧ x^2 + y^2 = 8 := by sorry

end min_sum_of_squares_min_sum_of_squares_wire_l1314_131446


namespace households_with_only_bike_l1314_131415

theorem households_with_only_bike (total : ℕ) (without_car_or_bike : ℕ) (with_both : ℕ) (with_car : ℕ) :
  total = 90 →
  without_car_or_bike = 11 →
  with_both = 14 →
  with_car = 44 →
  ∃ (with_only_bike : ℕ), with_only_bike = 35 ∧
    total = without_car_or_bike + with_both + (with_car - with_both) + with_only_bike :=
by sorry

end households_with_only_bike_l1314_131415


namespace bill_denomination_proof_l1314_131445

/-- Given the cost of berries, cost of peaches, and the amount of change received,
    prove that the denomination of the bill used equals the sum of these three amounts. -/
theorem bill_denomination_proof 
  (cost_berries : ℚ) 
  (cost_peaches : ℚ) 
  (change_received : ℚ) 
  (h1 : cost_berries = 719/100)
  (h2 : cost_peaches = 683/100)
  (h3 : change_received = 598/100) :
  cost_berries + cost_peaches + change_received = 20 := by
  sorry

end bill_denomination_proof_l1314_131445


namespace book_sale_gain_percentage_l1314_131497

def total_cost : ℚ := 420
def cost_loss_book : ℚ := 245
def loss_percentage : ℚ := 15 / 100

theorem book_sale_gain_percentage :
  let cost_gain_book := total_cost - cost_loss_book
  let selling_price := cost_loss_book * (1 - loss_percentage)
  let gain_percentage := (selling_price - cost_gain_book) / cost_gain_book * 100
  gain_percentage = 19 := by sorry

end book_sale_gain_percentage_l1314_131497


namespace binomial_distribution_problem_l1314_131429

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  ξ : ℝ

/-- The expectation of a binomial variable -/
def expectation (X : BinomialVariable) : ℝ := X.n * X.p

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_problem (X : BinomialVariable) 
  (h1 : expectation X = 300)
  (h2 : variance X = 200) :
  X.p = 1/3 := by
  sorry

end binomial_distribution_problem_l1314_131429


namespace absolute_value_equality_l1314_131413

theorem absolute_value_equality : |5 - 3| = -(3 - 5) := by
  sorry

end absolute_value_equality_l1314_131413


namespace complex_number_real_minus_imag_l1314_131465

theorem complex_number_real_minus_imag : 
  let z : ℂ := 5 / (-3 - Complex.I)
  let a : ℝ := z.re
  let b : ℝ := z.im
  a - b = -2 := by sorry

end complex_number_real_minus_imag_l1314_131465


namespace oliver_birthday_gift_l1314_131470

/-- The amount of money Oliver's friend gave him on his birthday --/
def friend_gift (initial_amount savings frisbee_cost puzzle_cost final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount + savings - frisbee_cost - puzzle_cost)

theorem oliver_birthday_gift :
  friend_gift 9 5 4 3 15 = 8 :=
by sorry

end oliver_birthday_gift_l1314_131470


namespace cube_sum_given_sum_and_diff_l1314_131436

theorem cube_sum_given_sum_and_diff (a b : ℝ) (h1 : a + b = 12) (h2 : a - b = 4) :
  a^3 + b^3 = 1344 := by
  sorry

end cube_sum_given_sum_and_diff_l1314_131436


namespace tangent_line_inclination_l1314_131478

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b

-- Define the derivative of f(x)
def f_prime (a x : ℝ) : ℝ := 3*x^2 - 2*a*x

-- Theorem statement
theorem tangent_line_inclination (a b : ℝ) :
  (f_prime a 1 = -1) → a = 2 := by
  sorry

end tangent_line_inclination_l1314_131478


namespace car_distance_in_yards_l1314_131476

/-- Proves the distance traveled by a car in yards over 60 minutes -/
theorem car_distance_in_yards
  (b : ℝ) (s : ℝ) (h_s_pos : s > 0) :
  let feet_per_s_seconds : ℝ := 5 * b / 12
  let seconds_in_hour : ℝ := 60 * 60
  let feet_in_yard : ℝ := 3
  let distance_feet : ℝ := feet_per_s_seconds * seconds_in_hour / s
  let distance_yards : ℝ := distance_feet / feet_in_yard
  distance_yards = 500 * b / s :=
by sorry


end car_distance_in_yards_l1314_131476


namespace f_increasing_interval_l1314_131454

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 9) / Real.log (1/3)

theorem f_increasing_interval :
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < -3 → f x₁ < f x₂ :=
by sorry

end f_increasing_interval_l1314_131454


namespace math_books_count_l1314_131441

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 90 →
  math_cost = 4 →
  history_cost = 5 →
  total_price = 396 →
  ∃ (math_books : ℕ),
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧
    math_books = 54 :=
by sorry

end math_books_count_l1314_131441


namespace perpendicular_bisector_equation_l1314_131423

/-- Given two points A and B in the plane, this theorem states that
    the equation 4x - 2y - 5 = 0 represents the perpendicular bisector
    of the line segment connecting A and B. -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) :
  A = (1, 2) →
  B = (3, 1) →
  ∀ (x y : ℝ), (4 * x - 2 * y - 5 = 0) ↔
    (((x - 1)^2 + (y - 2)^2 = (x - 3)^2 + (y - 1)^2) ∧
     ((y - 2) * (3 - 1) = -(x - 1) * (1 - 2))) :=
by sorry

end perpendicular_bisector_equation_l1314_131423


namespace smallest_m_perfect_square_and_cube_l1314_131467

theorem smallest_m_perfect_square_and_cube : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∃ (k : ℕ), 5 * m = k * k) ∧ 
  (∃ (l : ℕ), 3 * m = l * l * l) ∧ 
  (∀ (n : ℕ), n > 0 → 
    (∃ (k : ℕ), 5 * n = k * k) → 
    (∃ (l : ℕ), 3 * n = l * l * l) → 
    m ≤ n) ∧
  m = 243 :=
sorry

end smallest_m_perfect_square_and_cube_l1314_131467


namespace quadratic_equation_equivalence_l1314_131431

theorem quadratic_equation_equivalence : 
  (∀ x, x^2 - 2*(3*x - 2) + (x + 1) = 0 ↔ x^2 - 5*x + 5 = 0) := by
  sorry

end quadratic_equation_equivalence_l1314_131431


namespace potato_cost_proof_l1314_131439

/-- The original cost of one bag of potatoes from the farmer -/
def original_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The extra profit Boris made compared to Andrey -/
def extra_profit : ℝ := 1200

theorem potato_cost_proof :
  bags_bought * original_cost * andrey_increase +
  extra_profit =
  boris_first_sale * original_cost * boris_first_increase +
  boris_second_sale * original_cost * boris_first_increase * boris_second_increase :=
by sorry

end potato_cost_proof_l1314_131439


namespace distance_time_relationship_l1314_131440

/-- Represents the speed of a car in km/h -/
def speed : ℝ := 70

/-- Represents the distance traveled by the car in km -/
def distance (t : ℝ) : ℝ := speed * t

/-- Theorem stating the relationship between distance and time for the car -/
theorem distance_time_relationship (t : ℝ) : 
  distance t = speed * t ∧ 
  (∃ (S : ℝ → ℝ), S = distance ∧ (∀ x, S x = speed * x)) := by
  sorry

/-- The independent variable is time -/
def independent_variable : Type := ℝ

/-- The dependent variable is distance -/
def dependent_variable : ℝ → ℝ := distance

end distance_time_relationship_l1314_131440


namespace paolo_coconuts_l1314_131468

theorem paolo_coconuts (paolo dante : ℕ) : 
  dante = 3 * paolo →  -- Dante has thrice as many coconuts as Paolo
  dante - 10 = 32 →    -- Dante had 32 coconuts left after selling 10
  paolo = 14 :=        -- Paolo had 14 coconuts
by
  sorry

end paolo_coconuts_l1314_131468


namespace half_angle_quadrant_l1314_131459

/-- Given an angle in the second quadrant, its half angle can only be in the first or third quadrant -/
theorem half_angle_quadrant (θ : Real) :
  π / 2 < θ ∧ θ < π →
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by sorry

end half_angle_quadrant_l1314_131459


namespace viaduct_laying_speed_l1314_131457

/-- Proves that the original daily laying length is 300 meters given the conditions of the viaduct construction. -/
theorem viaduct_laying_speed 
  (total_length : ℝ) 
  (total_days : ℝ) 
  (initial_length : ℝ) 
  (h1 : total_length = 4800)
  (h2 : total_days = 9)
  (h3 : initial_length = 600)
  (h4 : ∃ (x : ℝ), (initial_length / x) + ((total_length - initial_length) / (2 * x)) = total_days)
  : ∃ (x : ℝ), x = 300 ∧ (initial_length / x) + ((total_length - initial_length) / (2 * x)) = total_days :=
sorry

end viaduct_laying_speed_l1314_131457


namespace unique_solution_trigonometric_equation_l1314_131450

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((100 : ℝ) * π / 180 - x * π / 180) =
    (Real.sin ((100 : ℝ) * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos ((100 : ℝ) * π / 180) - Real.cos (x * π / 180)) ∧
  x = 80 := by
sorry

end unique_solution_trigonometric_equation_l1314_131450


namespace probability_calculation_l1314_131472

/-- The probability of selecting exactly 2 purple and 2 orange marbles -/
def probability_two_purple_two_orange : ℚ :=
  66 / 1265

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 8

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 12

/-- The number of orange marbles in the bag -/
def orange_marbles : ℕ := 5

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles + orange_marbles

/-- The number of marbles selected -/
def selected_marbles : ℕ := 4

theorem probability_calculation :
  probability_two_purple_two_orange = 
    (Nat.choose purple_marbles 2 * Nat.choose orange_marbles 2) / 
    Nat.choose total_marbles selected_marbles :=
by
  sorry

end probability_calculation_l1314_131472


namespace f_zero_values_l1314_131411

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = 2 * f x * f y

/-- The theorem stating the possible values of f(0) -/
theorem f_zero_values (f : ℝ → ℝ) (h : FunctionalEquation f) :
    f 0 = 0 ∨ f 0 = (1 : ℝ) / 2 := by
  sorry

end f_zero_values_l1314_131411


namespace books_remaining_l1314_131496

/-- Calculates the number of books remaining in Tracy's charity book store -/
theorem books_remaining (initial_books : ℕ) (donors : ℕ) (books_per_donor : ℕ) (borrowed_books : ℕ) : 
  initial_books = 300 → 
  donors = 10 → 
  books_per_donor = 5 → 
  borrowed_books = 140 → 
  initial_books + donors * books_per_donor - borrowed_books = 210 := by
sorry

end books_remaining_l1314_131496


namespace magnitude_of_B_area_of_triangle_l1314_131490

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def triangleCondition (t : Triangle) : Prop :=
  2 * t.b * Real.sin t.B = (2 * t.a + t.c) * Real.sin t.A + (2 * t.c + t.a) * Real.sin t.C

-- Theorem for part I
theorem magnitude_of_B (t : Triangle) (h : triangleCondition t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Theorem for part II
theorem area_of_triangle (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = Real.sqrt 3) (h3 : t.A = Real.pi / 4) :
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 - Real.sqrt 3) / 4 := by
  sorry

end magnitude_of_B_area_of_triangle_l1314_131490


namespace time_to_write_rearrangements_l1314_131488

def name_length : ℕ := 5
def rearrangements_per_minute : ℕ := 20

theorem time_to_write_rearrangements :
  (Nat.factorial name_length / rearrangements_per_minute : ℚ) / 60 = 1/10 := by
  sorry

end time_to_write_rearrangements_l1314_131488


namespace product_sum_puzzle_l1314_131489

theorem product_sum_puzzle :
  ∃ (a b c : ℤ), (a * b + c = 40) ∧ (a + b ≠ 18) ∧
  (∃ (a' b' c' : ℤ), (a' * b' + c' = 40) ∧ (a' + b' ≠ 18) ∧ (c' ≠ c)) :=
by sorry

end product_sum_puzzle_l1314_131489


namespace marbles_exceed_200_l1314_131414

def marbles (n : ℕ) : ℕ := 3 * 2^(n - 1)

theorem marbles_exceed_200 :
  ∀ k : ℕ, k < 9 → marbles k ≤ 200 ∧ marbles 9 > 200 :=
by sorry

end marbles_exceed_200_l1314_131414


namespace nancy_bathroom_flooring_l1314_131483

/-- Represents the dimensions of a rectangular area -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The central area of Nancy's bathroom -/
def central_area : Rectangle := { length := 10, width := 10 }

/-- The hallway area of Nancy's bathroom -/
def hallway : Rectangle := { length := 6, width := 4 }

/-- The total area of hardwood flooring in Nancy's bathroom -/
def total_flooring_area : ℝ := area central_area + area hallway

theorem nancy_bathroom_flooring :
  total_flooring_area = 124 := by sorry

end nancy_bathroom_flooring_l1314_131483


namespace min_cut_length_40x30_paper_l1314_131481

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the problem setup -/
structure PaperCutProblem where
  paper : Rectangle
  inner_rectangle : Rectangle
  num_cuts : ℕ

/-- The minimum total length of cuts for the given problem -/
def min_cut_length (problem : PaperCutProblem) : ℕ := 
  2 * problem.paper.width + 2 * problem.paper.height

/-- Theorem stating the minimum cut length for the specific problem -/
theorem min_cut_length_40x30_paper (problem : PaperCutProblem) 
  (h1 : problem.paper = ⟨40, 30⟩) 
  (h2 : problem.inner_rectangle = ⟨10, 5⟩) 
  (h3 : problem.num_cuts = 4) : 
  min_cut_length problem = 140 := by
  sorry

#check min_cut_length_40x30_paper

end min_cut_length_40x30_paper_l1314_131481


namespace max_value_2x_plus_y_max_value_2x_plus_y_achievable_l1314_131434

theorem max_value_2x_plus_y (x y : ℝ) : 
  2 * x - y ≤ 0 → x + y ≤ 3 → x ≥ 0 → 2 * x + y ≤ 4 := by
  sorry

theorem max_value_2x_plus_y_achievable : 
  ∃ x y : ℝ, 2 * x - y ≤ 0 ∧ x + y ≤ 3 ∧ x ≥ 0 ∧ 2 * x + y = 4 := by
  sorry

end max_value_2x_plus_y_max_value_2x_plus_y_achievable_l1314_131434


namespace roberts_reading_capacity_l1314_131421

def reading_speed : ℝ := 75
def book_length : ℝ := 300
def available_time : ℝ := 9

theorem roberts_reading_capacity :
  ⌊available_time / (book_length / reading_speed)⌋ = 2 := by
  sorry

end roberts_reading_capacity_l1314_131421


namespace choose_four_from_nine_l1314_131460

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 4 → Nat.choose n k = 126 := by
  sorry

end choose_four_from_nine_l1314_131460


namespace sqrt_sum_problem_l1314_131486

theorem sqrt_sum_problem (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end sqrt_sum_problem_l1314_131486


namespace parallelogram_opposite_sides_parallel_equal_l1314_131471

-- Define a parallelogram
structure Parallelogram :=
  (vertices : Fin 4 → ℝ × ℝ)
  (is_parallelogram : 
    (vertices 0 - vertices 1 = vertices 3 - vertices 2) ∧
    (vertices 0 - vertices 3 = vertices 1 - vertices 2))

-- Define the property of having parallel and equal opposite sides
def has_parallel_equal_opposite_sides (p : Parallelogram) : Prop :=
  (p.vertices 0 - p.vertices 1 = p.vertices 3 - p.vertices 2) ∧
  (p.vertices 0 - p.vertices 3 = p.vertices 1 - p.vertices 2)

-- Theorem stating that all parallelograms have parallel and equal opposite sides
theorem parallelogram_opposite_sides_parallel_equal (p : Parallelogram) :
  has_parallel_equal_opposite_sides p :=
by
  sorry

-- Note: Rectangles, rhombuses, and squares are special cases of parallelograms,
-- so this theorem applies to them as well.

end parallelogram_opposite_sides_parallel_equal_l1314_131471


namespace solve_for_b_l1314_131418

theorem solve_for_b (y : ℝ) (b : ℝ) (h1 : y > 0) 
  (h2 : (6 * y) / b + (3 * y) / 10 = 0.60 * y) : b = 20 := by
  sorry

end solve_for_b_l1314_131418


namespace solution_set_when_a_eq_two_right_triangle_condition_l1314_131482

def f (a : ℝ) (x : ℝ) := |x + 1| - |a * x - 3|

theorem solution_set_when_a_eq_two :
  {x : ℝ | f 2 x > 1} = {x : ℝ | 1 < x ∧ x < 3} := by sorry

theorem right_triangle_condition (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, f a x = y ∧ f a y = 0 ∧ x ≠ y ∧ (x - y)^2 + (f a x)^2 = (x - y)^2 + y^2) →
  a = Real.sqrt 2 := by sorry

end solution_set_when_a_eq_two_right_triangle_condition_l1314_131482


namespace alpha_beta_range_l1314_131430

theorem alpha_beta_range (α β : ℝ) 
  (h1 : 0 < α - β) (h2 : α - β < π) 
  (h3 : 0 < α + 2*β) (h4 : α + 2*β < π) : 
  0 < α + β ∧ α + β < π :=
by sorry

end alpha_beta_range_l1314_131430


namespace trig_identity_proof_l1314_131477

theorem trig_identity_proof (α : Real) (h : Real.tan α = 2) :
  4 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 5 * (Real.cos α)^2 = 1 := by
  sorry

end trig_identity_proof_l1314_131477


namespace jonathan_book_purchase_l1314_131461

theorem jonathan_book_purchase (dictionary_cost dinosaur_book_cost cookbook_cost savings : ℕ) 
  (h1 : dictionary_cost = 11)
  (h2 : dinosaur_book_cost = 19)
  (h3 : cookbook_cost = 7)
  (h4 : savings = 8) :
  dictionary_cost + dinosaur_book_cost + cookbook_cost - savings = 29 := by
  sorry

end jonathan_book_purchase_l1314_131461


namespace not_polite_and_power_of_two_polite_or_power_of_two_l1314_131453

/-- A number is polite if it can be written as the sum of consecutive integers from m to n, where m < n. -/
def IsPolite (N : ℕ) : Prop :=
  ∃ m n : ℕ, m < n ∧ N = (n * (n + 1) - m * (m - 1)) / 2

/-- A number is a power of two if it can be written as 2^ℓ for some non-negative integer ℓ. -/
def IsPowerOfTwo (N : ℕ) : Prop :=
  ∃ ℓ : ℕ, N = 2^ℓ

/-- No number is both polite and a power of two. -/
theorem not_polite_and_power_of_two (N : ℕ) : ¬(IsPolite N ∧ IsPowerOfTwo N) := by
  sorry

/-- Every positive integer is either polite or a power of two. -/
theorem polite_or_power_of_two (N : ℕ) : N > 0 → IsPolite N ∨ IsPowerOfTwo N := by
  sorry

end not_polite_and_power_of_two_polite_or_power_of_two_l1314_131453


namespace smaller_cone_height_equals_frustum_height_l1314_131494

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  height : ℝ
  larger_base_area : ℝ
  smaller_base_area : ℝ

/-- Calculates the height of the smaller cone removed to form the frustum -/
def smaller_cone_height (f : Frustum) : ℝ :=
  f.height

/-- Theorem stating that the height of the smaller cone is equal to the frustum's height -/
theorem smaller_cone_height_equals_frustum_height (f : Frustum)
  (h1 : f.height = 18)
  (h2 : f.larger_base_area = 400 * Real.pi)
  (h3 : f.smaller_base_area = 100 * Real.pi) :
  smaller_cone_height f = f.height :=
by sorry

end smaller_cone_height_equals_frustum_height_l1314_131494


namespace rs_value_l1314_131491

theorem rs_value (r s : ℝ) (hr : r > 0) (hs : s > 0)
  (h1 : r^3 + s^3 = 1) (h2 : r^6 + s^6 = 15/16) :
  r * s = 1 / Real.rpow 48 (1/3) :=
sorry

end rs_value_l1314_131491


namespace bobby_candy_count_l1314_131449

/-- The number of candy pieces Bobby had initially -/
def initial_candy : ℕ := 22

/-- The number of candy pieces Bobby ate at the start -/
def eaten_start : ℕ := 9

/-- The number of additional candy pieces Bobby ate -/
def eaten_additional : ℕ := 5

/-- The number of candy pieces Bobby has left -/
def remaining_candy : ℕ := 8

/-- Theorem stating that Bobby's initial candy count is correct -/
theorem bobby_candy_count : 
  initial_candy = eaten_start + eaten_additional + remaining_candy :=
by sorry

end bobby_candy_count_l1314_131449


namespace some_number_value_l1314_131419

theorem some_number_value : ∃ (n : ℚ), n = 10/3 ∧ 
  (3 + 2 * (3/2 : ℚ))^5 = (1 + n * (3/2 : ℚ))^4 := by
  sorry

end some_number_value_l1314_131419


namespace vector_magnitude_proof_l1314_131402

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

theorem vector_magnitude_proof : ‖(2 • a) + b‖ = 5 := by
  sorry

end vector_magnitude_proof_l1314_131402


namespace complex_number_quadrant_l1314_131479

/-- If (a - i) / (1 + i) is a pure imaginary number where a ∈ ℝ, then 3a + 4i is in the first quadrant of the complex plane. -/
theorem complex_number_quadrant (a : ℝ) :
  (((a : ℂ) - I) / (1 + I)).im ≠ 0 ∧ (((a : ℂ) - I) / (1 + I)).re = 0 →
  (3 * a : ℝ) > 0 ∧ 4 > 0 := by
  sorry

end complex_number_quadrant_l1314_131479


namespace triangle_angle_calculation_l1314_131458

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_angle_calculation (t : Triangle) :
  t.A = 60 ∧ t.B = 2 * t.C ∧ t.A + t.B + t.C = 180 → t.B = 80 :=
by
  sorry


end triangle_angle_calculation_l1314_131458


namespace three_in_A_even_not_in_A_l1314_131438

-- Define the set A
def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- Theorem statements
theorem three_in_A : 3 ∈ A := by sorry

theorem even_not_in_A : ∀ k : ℤ, (4*k - 2) ∉ A := by sorry

end three_in_A_even_not_in_A_l1314_131438


namespace kim_thursday_sales_l1314_131493

/-- The number of boxes Kim sold on Tuesday -/
def tuesday_sales : ℕ := 4800

/-- The number of boxes Kim sold on Wednesday -/
def wednesday_sales : ℕ := tuesday_sales / 2

/-- The number of boxes Kim sold on Thursday -/
def thursday_sales : ℕ := wednesday_sales / 2

/-- Theorem stating that Kim sold 1200 boxes on Thursday -/
theorem kim_thursday_sales : thursday_sales = 1200 := by
  sorry

end kim_thursday_sales_l1314_131493


namespace projection_matrix_values_l1314_131463

def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem projection_matrix_values :
  ∀ (a c : ℚ),
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![a, 18/45; c, 27/45]
  is_projection_matrix P →
  a = 1/5 ∧ c = 2/5 := by
sorry

end projection_matrix_values_l1314_131463


namespace slow_dancers_count_l1314_131452

theorem slow_dancers_count (total_kids : ℕ) (non_slow_dancers : ℕ) : 
  total_kids = 140 → 
  non_slow_dancers = 10 → 
  (total_kids / 4 : ℕ) - non_slow_dancers = 25 := by
  sorry

end slow_dancers_count_l1314_131452


namespace min_omega_value_l1314_131473

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem min_omega_value (ω φ : ℝ) (h_ω_pos : ω > 0) 
  (h_exists : ∃ x₀ : ℝ, f ω φ (x₀ + 2) - f ω φ x₀ = 4) :
  ω ≥ Real.pi / 2 ∧ ∀ ω' > 0, (∃ x₀' : ℝ, f ω' φ (x₀' + 2) - f ω' φ x₀' = 4) → ω' ≥ Real.pi / 2 :=
sorry

end min_omega_value_l1314_131473


namespace c_work_rate_l1314_131456

def work_rate (days : ℚ) : ℚ := 1 / days

theorem c_work_rate 
  (ab_days : ℚ) 
  (bc_days : ℚ) 
  (ca_days : ℚ) 
  (hab : ab_days = 3) 
  (hbc : bc_days = 4) 
  (hca : ca_days = 6) : 
  ∃ (c_rate : ℚ), 
    c_rate = 1 / 24 ∧ 
    c_rate + work_rate ab_days = work_rate ca_days ∧
    c_rate + work_rate bc_days - work_rate ab_days = work_rate bc_days := by
  sorry

end c_work_rate_l1314_131456


namespace min_value_theorem_l1314_131448

/-- Given that f(x) = a^x - b and g(x) = x + 1, where a > 0, a ≠ 1, and b ∈ ℝ,
    if f(x) * g(x) ≤ 0 for all real x, then the minimum value of 1/a + 4/b is 4 -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  (∀ x : ℝ, (a^x - b) * (x + 1) ≤ 0) →
  (∃ m : ℝ, m = 4 ∧ ∀ a b : ℝ, a > 0 → a ≠ 1 → (∀ x : ℝ, (a^x - b) * (x + 1) ≤ 0) → 1/a + 4/b ≥ m) :=
by sorry

end min_value_theorem_l1314_131448
