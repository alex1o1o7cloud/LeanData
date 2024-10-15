import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l954_95417

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A hyperbola with foci at the vertices of a triangle -/
structure Hyperbola (t : Triangle) where
  /-- The hyperbola passes through point A of the triangle -/
  passes_through_A : Bool

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola t) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (t : Triangle) (h : Hyperbola t) :
  t.a = 4 ∧ t.b = 5 ∧ t.c = Real.sqrt 21 ∧ h.passes_through_A = true →
  eccentricity h = 5 + Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l954_95417


namespace NUMINAMATH_CALUDE_angle_triple_complement_l954_95465

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l954_95465


namespace NUMINAMATH_CALUDE_factorial_divisibility_l954_95453

theorem factorial_divisibility (k n : ℕ) (hk : 0 < k ∧ k ≤ 2020) (hn : 0 < n) :
  ¬ (3^((k-1)*n+1) ∣ ((Nat.factorial (k*n) / Nat.factorial n)^2)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l954_95453


namespace NUMINAMATH_CALUDE_smallest_shift_l954_95427

/-- A function with period 30 -/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

/-- The shift property for g(x/4) -/
def shift_property (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 4) = g (x / 4)

/-- The theorem stating the smallest positive b is 120 -/
theorem smallest_shift (g : ℝ → ℝ) (h : periodic_function g) :
  ∃ b : ℝ, b > 0 ∧ shift_property g b ∧ ∀ b' : ℝ, b' > 0 → shift_property g b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l954_95427


namespace NUMINAMATH_CALUDE_interview_problem_l954_95423

theorem interview_problem (n : ℕ) 
  (h1 : n ≥ 2) 
  (h2 : (Nat.choose 2 2 * Nat.choose (n - 2) 1) / Nat.choose n 3 = 1 / 70) : 
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_interview_problem_l954_95423


namespace NUMINAMATH_CALUDE_problem_statement_l954_95400

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem problem_statement : 8^(2/3) + lg 25 - lg (1/4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l954_95400


namespace NUMINAMATH_CALUDE_Q_subset_P_l954_95452

def P : Set ℝ := {x | x ≥ -1}
def Q : Set ℝ := {y | y ≥ 0}

theorem Q_subset_P : Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_Q_subset_P_l954_95452


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_80_l954_95419

theorem thirty_percent_less_than_80 : 
  80 * (1 - 0.3) = (224 / 5) * (1 + 1 / 4) := by sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_80_l954_95419


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l954_95408

theorem cricket_team_captain_age 
  (team_size : ℕ) 
  (captain_age wicket_keeper_age : ℕ) 
  (team_average_age : ℚ) 
  (remaining_players_average_age : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 7 →
  team_average_age = 23 →
  remaining_players_average_age = team_average_age - 1 →
  (team_size : ℚ) * team_average_age = 
    ((team_size - 2) : ℚ) * remaining_players_average_age + captain_age + wicket_keeper_age →
  captain_age = 24 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l954_95408


namespace NUMINAMATH_CALUDE_grocery_store_bottles_l954_95469

theorem grocery_store_bottles : 
  157 + 126 + 87 + 52 + 64 = 486 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_bottles_l954_95469


namespace NUMINAMATH_CALUDE_base3_addition_l954_95476

-- Define a type for base-3 numbers
def Base3 := ℕ

-- Function to convert a base-3 number to its decimal representation
def to_decimal (n : Base3) : ℕ := sorry

-- Function to convert a decimal number to its base-3 representation
def to_base3 (n : ℕ) : Base3 := sorry

-- Define the given numbers in base 3
def a : Base3 := to_base3 1
def b : Base3 := to_base3 22
def c : Base3 := to_base3 212
def d : Base3 := to_base3 1001

-- Define the result in base 3
def result : Base3 := to_base3 210

-- Theorem statement
theorem base3_addition :
  to_decimal a - to_decimal b + to_decimal c - to_decimal d = to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_base3_addition_l954_95476


namespace NUMINAMATH_CALUDE_age_difference_l954_95459

theorem age_difference (rona_age rachel_age collete_age : ℕ) : 
  rona_age = 8 →
  rachel_age = 2 * rona_age →
  collete_age = rona_age / 2 →
  rachel_age - collete_age = 12 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l954_95459


namespace NUMINAMATH_CALUDE_cone_base_circumference_l954_95488

/-- The circumference of the base of a right circular cone formed from a 180° sector of a circle --/
theorem cone_base_circumference (r : ℝ) (h : r = 5) :
  let full_circle_circumference := 2 * π * r
  let sector_angle := π  -- 180° in radians
  let full_angle := 2 * π  -- 360° in radians
  let base_circumference := (sector_angle / full_angle) * full_circle_circumference
  base_circumference = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l954_95488


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l954_95422

theorem geometric_sum_n1 (x : ℝ) (h : x ≠ 1) :
  1 + x + x^2 = (1 - x^3) / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l954_95422


namespace NUMINAMATH_CALUDE_jade_transactions_l954_95416

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 16 →
  jade = 82 := by
sorry

end NUMINAMATH_CALUDE_jade_transactions_l954_95416


namespace NUMINAMATH_CALUDE_correct_factorization_l954_95475

theorem correct_factorization (x : ℝ) : 10 * x^2 - 5 * x = 5 * x * (2 * x - 1) := by
  sorry

#check correct_factorization

end NUMINAMATH_CALUDE_correct_factorization_l954_95475


namespace NUMINAMATH_CALUDE_mortgage_payment_proof_l954_95464

/-- Calculates the monthly mortgage payment -/
def calculate_monthly_payment (house_price : ℕ) (deposit : ℕ) (years : ℕ) : ℚ :=
  let mortgage := house_price - deposit
  let annual_payment := mortgage / years
  annual_payment / 12

/-- Proves that the monthly payment for the given mortgage scenario is 2 thousand dollars -/
theorem mortgage_payment_proof (house_price deposit years : ℕ) 
  (h1 : house_price = 280000)
  (h2 : deposit = 40000)
  (h3 : years = 10) :
  calculate_monthly_payment house_price deposit years = 2000 := by
  sorry

#eval calculate_monthly_payment 280000 40000 10

end NUMINAMATH_CALUDE_mortgage_payment_proof_l954_95464


namespace NUMINAMATH_CALUDE_min_value_abc_l954_95445

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∃ (min : ℝ), min = 1/2916 ∧ ∀ x, x = a^3 * b^2 * c → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l954_95445


namespace NUMINAMATH_CALUDE_dubblefud_yellow_count_l954_95487

/-- The game of dubblefud with yellow, blue, and green chips -/
def dubblefud (yellow blue green : ℕ) : Prop :=
  2^yellow * 4^blue * 5^green = 16000 ∧ blue = green

theorem dubblefud_yellow_count :
  ∀ y b g : ℕ, dubblefud y b g → y = 1 :=
by sorry

end NUMINAMATH_CALUDE_dubblefud_yellow_count_l954_95487


namespace NUMINAMATH_CALUDE_second_smallest_three_digit_in_pascal_l954_95425

/-- Pascal's Triangle coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Checks if a number is three digits -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- The smallest three-digit number in Pascal's Triangle -/
def smallestThreeDigit : ℕ := 100

/-- The row where the smallest three-digit number first appears -/
def smallestThreeDigitRow : ℕ := 100

theorem second_smallest_three_digit_in_pascal :
  ∃ (n : ℕ), isThreeDigit n ∧
    (∀ (m : ℕ), isThreeDigit m → m < n → m = smallestThreeDigit) ∧
    (∃ (row : ℕ), binomial row 1 = n ∧
      ∀ (r : ℕ), r < row → ¬(∃ (k : ℕ), isThreeDigit (binomial r k) ∧ binomial r k = n)) ∧
    n = 101 ∧ row = 101 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_three_digit_in_pascal_l954_95425


namespace NUMINAMATH_CALUDE_snack_machine_quarters_l954_95482

def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50
def quarter_value : ℕ := 25

def total_cost (candy_bars chocolate juice : ℕ) : ℕ :=
  candy_bars * candy_bar_cost + chocolate * chocolate_cost + juice * juice_cost

def quarters_needed (total : ℕ) : ℕ :=
  (total + quarter_value - 1) / quarter_value

theorem snack_machine_quarters : quarters_needed (total_cost 3 2 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_snack_machine_quarters_l954_95482


namespace NUMINAMATH_CALUDE_petra_age_l954_95414

theorem petra_age (petra_age mother_age : ℕ) : 
  petra_age + mother_age = 47 →
  mother_age = 2 * petra_age + 14 →
  mother_age = 36 →
  petra_age = 11 :=
by sorry

end NUMINAMATH_CALUDE_petra_age_l954_95414


namespace NUMINAMATH_CALUDE_power_of_two_sum_l954_95404

theorem power_of_two_sum (x : ℕ) : 2^x + 2^x + 2^x + 2^x + 2^x = 2048 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l954_95404


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l954_95492

theorem set_equality_implies_sum_of_powers (a b : ℝ) :
  let A : Set ℝ := {a, a^2, a*b}
  let B : Set ℝ := {1, a, b}
  A = B → a^2004 + b^2004 = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l954_95492


namespace NUMINAMATH_CALUDE_star_equation_solution_l954_95451

/-- Custom binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

/-- Theorem stating that if 7 ⋆ y = 85, then y = 92/9 -/
theorem star_equation_solution (y : ℝ) (h : star 7 y = 85) : y = 92 / 9 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l954_95451


namespace NUMINAMATH_CALUDE_last_digit_of_2011_powers_l954_95432

theorem last_digit_of_2011_powers : ∃ n : ℕ, (2^2011 + 3^2011) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2011_powers_l954_95432


namespace NUMINAMATH_CALUDE_duplicate_page_number_l954_95471

/-- The largest positive integer n such that n(n+1)/2 < 2550 -/
def n : ℕ := 70

/-- The theorem stating the existence and uniqueness of the duplicated page number -/
theorem duplicate_page_number :
  ∃! x : ℕ, x ≤ n ∧ (n * (n + 1)) / 2 + x = 2550 := by sorry

end NUMINAMATH_CALUDE_duplicate_page_number_l954_95471


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_theorem_l954_95457

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

theorem line_plane_perpendicular_theorem 
  (a b : Line) (α : Plane) :
  perpendicular_lines a b → 
  perpendicular_line_plane a α → 
  parallel_line_plane b α ∨ subset_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_theorem_l954_95457


namespace NUMINAMATH_CALUDE_beach_probability_l954_95477

/-- Given a beach scenario with people wearing sunglasses and caps -/
structure BeachScenario where
  sunglasses : ℕ  -- Number of people wearing sunglasses
  caps : ℕ        -- Number of people wearing caps
  prob_cap_and_sunglasses : ℚ  -- Probability that a person wearing a cap is also wearing sunglasses

/-- The probability that a person wearing sunglasses is also wearing a cap -/
def prob_sunglasses_and_cap (scenario : BeachScenario) : ℚ :=
  (scenario.prob_cap_and_sunglasses * scenario.caps) / scenario.sunglasses

theorem beach_probability (scenario : BeachScenario) 
  (h1 : scenario.sunglasses = 75)
  (h2 : scenario.caps = 60)
  (h3 : scenario.prob_cap_and_sunglasses = 1/3) :
  prob_sunglasses_and_cap scenario = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_beach_probability_l954_95477


namespace NUMINAMATH_CALUDE_probability_of_sum_seven_l954_95439

-- Define the two dice
def die1 : Finset ℕ := {1, 2, 3, 4, 5, 6}
def die2 : Finset ℕ := {2, 3, 4, 5, 6, 7}

-- Define the total outcomes
def total_outcomes : ℕ := die1.card * die2.card

-- Define the favorable outcomes (pairs that sum to 7)
def favorable_outcomes : Finset (ℕ × ℕ) := 
  {(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)}

-- Theorem statement
theorem probability_of_sum_seven :
  (favorable_outcomes.card : ℚ) / total_outcomes = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_sum_seven_l954_95439


namespace NUMINAMATH_CALUDE_inequality_proof_l954_95401

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l954_95401


namespace NUMINAMATH_CALUDE_sparrow_percentage_among_non_owls_l954_95470

theorem sparrow_percentage_among_non_owls (total : ℝ) (total_pos : 0 < total) :
  let sparrows := 0.4 * total
  let owls := 0.2 * total
  let pigeons := 0.1 * total
  let finches := 0.2 * total
  let robins := total - (sparrows + owls + pigeons + finches)
  let non_owls := total - owls
  (sparrows / non_owls) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sparrow_percentage_among_non_owls_l954_95470


namespace NUMINAMATH_CALUDE_water_canteen_count_l954_95421

def water_problem (flow_rate : ℚ) (duration : ℚ) (additional_water : ℚ) (small_canteen_capacity : ℚ) : ℕ :=
  let total_water := flow_rate * duration + additional_water
  (total_water / small_canteen_capacity).ceil.toNat

theorem water_canteen_count :
  water_problem 9 8 7 6 = 14 := by sorry

end NUMINAMATH_CALUDE_water_canteen_count_l954_95421


namespace NUMINAMATH_CALUDE_triple_composition_even_l954_95443

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Given an even function f, prove that f(f(f(x))) is also even -/
theorem triple_composition_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (fun x ↦ f (f (f x))) := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l954_95443


namespace NUMINAMATH_CALUDE_final_price_approx_l954_95437

/-- The final price after applying two successive discounts to a list price. -/
def final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  list_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price after discounts is approximately 57.33 -/
theorem final_price_approx :
  let list_price : ℝ := 65
  let discount1 : ℝ := 0.1  -- 10%
  let discount2 : ℝ := 0.020000000000000027  -- 2.0000000000000027%
  abs (final_price list_price discount1 discount2 - 57.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_final_price_approx_l954_95437


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l954_95444

theorem gcd_power_minus_one (m n : ℕ+) :
  Nat.gcd ((2 ^ m.val) - 1) ((2 ^ n.val) - 1) = (2 ^ Nat.gcd m.val n.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l954_95444


namespace NUMINAMATH_CALUDE_parabola_sum_l954_95494

/-- Represents a parabola of the form x = dy^2 + ey + f -/
structure Parabola where
  d : ℚ
  e : ℚ
  f : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.xCoord (p : Parabola) (y : ℚ) : ℚ :=
  p.d * y^2 + p.e * y + p.f

theorem parabola_sum (p : Parabola) :
  p.xCoord (-6) = 7 →  -- vertex condition
  p.xCoord (-3) = 2 →  -- point condition
  p.d + p.e + p.f = -182/9 := by
  sorry

#eval (-5/9 : ℚ) + (-20/3 : ℚ) + (-13 : ℚ)  -- Should evaluate to -182/9

end NUMINAMATH_CALUDE_parabola_sum_l954_95494


namespace NUMINAMATH_CALUDE_total_weekly_prayers_l954_95440

/-- The number of prayers Pastor Paul makes on a regular day -/
def paul_regular_prayers : ℕ := 20

/-- The number of prayers Pastor Caroline makes on a regular day -/
def caroline_regular_prayers : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weekdays (non-Sunday days) in a week -/
def weekdays : ℕ := 6

/-- Calculate Pastor Paul's total prayers for a week -/
def paul_weekly_prayers : ℕ :=
  paul_regular_prayers * weekdays + 2 * paul_regular_prayers

/-- Calculate Pastor Bruce's total prayers for a week -/
def bruce_weekly_prayers : ℕ :=
  (paul_regular_prayers / 2) * weekdays + 2 * (2 * paul_regular_prayers)

/-- Calculate Pastor Caroline's total prayers for a week -/
def caroline_weekly_prayers : ℕ :=
  caroline_regular_prayers * weekdays + 3 * caroline_regular_prayers

/-- The main theorem: total prayers of all pastors in a week -/
theorem total_weekly_prayers :
  paul_weekly_prayers + bruce_weekly_prayers + caroline_weekly_prayers = 390 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_prayers_l954_95440


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l954_95467

theorem complex_number_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (Complex.I * 3) / (1 + Complex.I * 2) = ⟨x, y⟩ := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l954_95467


namespace NUMINAMATH_CALUDE_alex_sandwiches_l954_95433

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) : ℕ :=
  num_meats * (num_cheeses.choose 3)

/-- Theorem: Alex can make 1760 different sandwiches -/
theorem alex_sandwiches :
  num_sandwiches 8 12 = 1760 := by
  sorry

end NUMINAMATH_CALUDE_alex_sandwiches_l954_95433


namespace NUMINAMATH_CALUDE_gcd_324_243_135_l954_95468

theorem gcd_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_324_243_135_l954_95468


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l954_95473

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l954_95473


namespace NUMINAMATH_CALUDE_range_of_m_for_p_or_q_l954_95458

-- Define the propositions p and q as functions of m
def proposition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m+2)*x + 1 ≠ 0

-- State the theorem
theorem range_of_m_for_p_or_q :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) ↔ m < -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_or_q_l954_95458


namespace NUMINAMATH_CALUDE_order_of_abc_l954_95438

theorem order_of_abc : ∀ (a b c : ℝ),
  a = 0.1 * Real.exp 0.1 →
  b = 1 / 9 →
  c = -Real.log 0.9 →
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_order_of_abc_l954_95438


namespace NUMINAMATH_CALUDE_calculate_withdrawal_l954_95489

/-- Calculates the withdrawal amount given initial balance and transactions --/
theorem calculate_withdrawal 
  (initial_balance : ℕ) 
  (deposit_last_month : ℕ) 
  (deposit_this_month : ℕ) 
  (balance_increase : ℕ) 
  (h1 : initial_balance = 150)
  (h2 : deposit_last_month = 17)
  (h3 : deposit_this_month = 21)
  (h4 : balance_increase = 16) :
  ∃ (withdrawal : ℕ), 
    initial_balance + deposit_last_month - withdrawal + deposit_this_month 
    = initial_balance + balance_increase ∧ 
    withdrawal = 22 := by
sorry

end NUMINAMATH_CALUDE_calculate_withdrawal_l954_95489


namespace NUMINAMATH_CALUDE_y_intercepts_equal_negative_two_l954_95493

-- Define the equations
def equation1 (x y : ℝ) : Prop := 2 * x - 3 * y = 6
def equation2 (x y : ℝ) : Prop := x + 4 * y = -8

-- Define y-intercept
def is_y_intercept (y : ℝ) (eq : ℝ → ℝ → Prop) : Prop := eq 0 y

-- Theorem statement
theorem y_intercepts_equal_negative_two :
  (is_y_intercept (-2) equation1) ∧ (is_y_intercept (-2) equation2) :=
sorry

end NUMINAMATH_CALUDE_y_intercepts_equal_negative_two_l954_95493


namespace NUMINAMATH_CALUDE_four_row_grid_has_sixteen_triangles_l954_95441

/-- Represents a triangular grid with a given number of rows at the base -/
structure TriangularGrid where
  baseRows : Nat

/-- Calculates the number of small triangles in a triangular grid -/
def smallTriangles (grid : TriangularGrid) : Nat :=
  (grid.baseRows * (grid.baseRows + 1)) / 2

/-- Calculates the number of medium triangles in a triangular grid -/
def mediumTriangles (grid : TriangularGrid) : Nat :=
  ((grid.baseRows - 1) * grid.baseRows) / 2

/-- Calculates the number of large triangles in a triangular grid -/
def largeTriangles (grid : TriangularGrid) : Nat :=
  if grid.baseRows ≥ 3 then 1 else 0

/-- Calculates the total number of triangles in a triangular grid -/
def totalTriangles (grid : TriangularGrid) : Nat :=
  smallTriangles grid + mediumTriangles grid + largeTriangles grid

/-- Theorem: A triangular grid with 4 rows at the base has 16 total triangles -/
theorem four_row_grid_has_sixteen_triangles :
  totalTriangles { baseRows := 4 } = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_row_grid_has_sixteen_triangles_l954_95441


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l954_95455

theorem sum_of_fractions_inequality (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (2 * a^2 + b^2 + 3)) + (1 / (2 * b^2 + c^2 + 3)) + (1 / (2 * c^2 + a^2 + 3)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l954_95455


namespace NUMINAMATH_CALUDE_specific_triangle_intercepted_segments_l954_95478

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  (right_triangle : side1^2 + side2^2 = hypotenuse^2)

/-- Calculates the lengths of segments intercepted by lines drawn through the center of the inscribed circle parallel to the sides of the triangle -/
def intercepted_segments (triangle : RightTriangleWithInscribedCircle) : (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem statement for the specific right triangle with sides 6, 8, and 10 -/
theorem specific_triangle_intercepted_segments :
  let triangle : RightTriangleWithInscribedCircle := {
    side1 := 6,
    side2 := 8,
    hypotenuse := 10,
    right_triangle := by norm_num
  }
  intercepted_segments triangle = (3/2, 8/3, 25/6) := by sorry

end NUMINAMATH_CALUDE_specific_triangle_intercepted_segments_l954_95478


namespace NUMINAMATH_CALUDE_egyptian_fraction_solutions_l954_95442

def EgyptianFractionSolutions : Set (ℕ × ℕ × ℕ × ℕ) := {
  (2, 3, 7, 42), (2, 3, 8, 24), (2, 3, 9, 18), (2, 3, 10, 15), (2, 3, 12, 12),
  (2, 4, 5, 20), (2, 4, 6, 12), (2, 4, 8, 8), (2, 5, 5, 10), (2, 6, 6, 6),
  (3, 3, 4, 12), (3, 3, 6, 6), (3, 4, 4, 6), (4, 4, 4, 4)
}

theorem egyptian_fraction_solutions :
  {(x, y, z, t) : ℕ × ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
    (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z + (1 : ℚ) / t = 1 ∧
    x ≤ y ∧ y ≤ z ∧ z ≤ t} = EgyptianFractionSolutions := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_solutions_l954_95442


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l954_95481

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₁₃ : ℚ) (h₁ : a₁ = 10) (h₂ : a₁₃ = 50) :
  arithmetic_sequence a₁ ((a₁₃ - a₁) / 12) 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l954_95481


namespace NUMINAMATH_CALUDE_cosine_of_45_degree_angle_in_triangle_l954_95466

theorem cosine_of_45_degree_angle_in_triangle (A B C : ℝ) :
  A = 120 ∧ B = 45 ∧ C = 15 ∧ A + B + C = 180 →
  Real.cos (B * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_45_degree_angle_in_triangle_l954_95466


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l954_95462

-- Define the cones and marbles
def narrow_cone_radius : ℝ := 5
def wide_cone_radius : ℝ := 10
def narrow_marble_radius : ℝ := 2
def wide_marble_radius : ℝ := 3

-- Define the volume ratio
def volume_ratio : ℝ := 4

-- Theorem statement
theorem liquid_rise_ratio :
  let narrow_cone_volume := (1/3) * Real.pi * narrow_cone_radius^2
  let wide_cone_volume := (1/3) * Real.pi * wide_cone_radius^2
  let narrow_marble_volume := (4/3) * Real.pi * narrow_marble_radius^3
  let wide_marble_volume := (4/3) * Real.pi * wide_marble_radius^3
  let narrow_cone_rise := narrow_marble_volume / (Real.pi * narrow_cone_radius^2)
  let wide_cone_rise := wide_marble_volume / (Real.pi * wide_cone_radius^2)
  wide_cone_volume = volume_ratio * narrow_cone_volume →
  narrow_cone_rise / wide_cone_rise = 8 := by
  sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l954_95462


namespace NUMINAMATH_CALUDE_max_abs_f_implies_sum_l954_95430

def f (a b x : ℝ) := x^2 + a*x + b

theorem max_abs_f_implies_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| ≤ (1/2 : ℝ)) →
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| = (1/2 : ℝ)) →
  4*a + 3*b = -(3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_max_abs_f_implies_sum_l954_95430


namespace NUMINAMATH_CALUDE_expression_evaluation_l954_95490

theorem expression_evaluation :
  let x : ℚ := 4 / 7
  let y : ℚ := 8 / 5
  (7 * x + 5 * y + 4) / (60 * x * y + 5) = 560 / 559 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l954_95490


namespace NUMINAMATH_CALUDE_midpoint_path_difference_l954_95447

/-- Given a rectangle with sides a and b, and a segment AB of length 4 inside it,
    the path traced by the midpoint C of AB as A completes one revolution around 
    the perimeter is shorter than the perimeter by 16 - 4π. -/
theorem midpoint_path_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 4 < min a b) :
  2 * (a + b) - (2 * (a + b) - 16 + 4 * Real.pi) = 16 - 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_midpoint_path_difference_l954_95447


namespace NUMINAMATH_CALUDE_proportional_function_graph_l954_95434

/-- A proportional function with coefficient 2 -/
def f (x : ℝ) : ℝ := 2 * x

theorem proportional_function_graph (x y : ℝ) :
  y = f x → (∃ k : ℝ, k > 0 ∧ y = k * x) ∧ f 0 = 0 := by
  sorry

#check proportional_function_graph

end NUMINAMATH_CALUDE_proportional_function_graph_l954_95434


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l954_95403

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter
  (t1 : Triangle)
  (h1 : t1.isIsosceles)
  (h2 : t1.a = 12 ∧ t1.b = 12 ∧ t1.c = 18)
  (t2 : Triangle)
  (h3 : areSimilar t1 t2)
  (h4 : min t2.a (min t2.b t2.c) = 30) :
  t2.perimeter = 120 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l954_95403


namespace NUMINAMATH_CALUDE_henry_initial_amount_l954_95413

/-- Henry's initial amount of money -/
def henry_initial : ℕ := sorry

/-- Amount Henry earned from chores -/
def chores_earnings : ℕ := 2

/-- Amount of money Henry's friend had -/
def friend_money : ℕ := 13

/-- Total amount when they put their money together -/
def total_money : ℕ := 20

theorem henry_initial_amount :
  henry_initial + chores_earnings + friend_money = total_money ∧
  henry_initial = 5 := by sorry

end NUMINAMATH_CALUDE_henry_initial_amount_l954_95413


namespace NUMINAMATH_CALUDE_roots_sum_powers_l954_95485

theorem roots_sum_powers (p q : ℝ) : 
  p^2 - 6*p + 10 = 0 → q^2 - 6*q + 10 = 0 → p^3 + p^4*q^2 + p^2*q^4 + p*q^3 + p^5*q^3 = 38676 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l954_95485


namespace NUMINAMATH_CALUDE_cubic_root_solutions_l954_95496

/-- A rational triple (a, b, c) is a solution if a, b, and c are the roots of the polynomial x^3 + ax^2 + bx + c = 0 -/
def IsSolution (a b c : ℚ) : Prop :=
  ∀ x : ℚ, x^3 + a*x^2 + b*x + c = 0 ↔ (x = a ∨ x = b ∨ x = c)

/-- The only rational triples (a, b, c) that are solutions are (0, 0, 0), (1, -1, -1), and (1, -2, 0) -/
theorem cubic_root_solutions :
  ∀ a b c : ℚ, IsSolution a b c ↔ ((a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, -1, -1) ∨ (a, b, c) = (1, -2, 0)) :=
sorry

end NUMINAMATH_CALUDE_cubic_root_solutions_l954_95496


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l954_95454

theorem quadratic_roots_nature (k : ℂ) (h : k.re = 0 ∧ k.im ≠ 0) :
  ∃ (z₁ z₂ : ℂ), 10 * z₁^2 - 5 * z₁ - k = 0 ∧
                 10 * z₂^2 - 5 * z₂ - k = 0 ∧
                 z₁.im = 0 ∧
                 z₂.re = 0 ∧ z₂.im ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l954_95454


namespace NUMINAMATH_CALUDE_otimes_inequality_l954_95456

/-- Custom binary operation ⊗ on ℝ -/
def otimes (x y : ℝ) : ℝ := (1 - x) * (1 + y)

/-- Theorem: If (x-a) ⊗ (x+a) < 1 holds for any real x, then -2 < a < 0 -/
theorem otimes_inequality (a : ℝ) : 
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -2 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_l954_95456


namespace NUMINAMATH_CALUDE_min_nSn_l954_95426

/-- Represents an arithmetic sequence and its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  m : ℕ      -- Given index
  h_m : m ≥ 2
  h_sum_pred : S (m - 1) = -2
  h_sum : S m = 0
  h_sum_succ : S (m + 1) = 3

/-- The minimum value of nS_n for the given arithmetic sequence -/
theorem min_nSn (seq : ArithmeticSequence) : 
  ∃ (k : ℝ), k = -9 ∧ ∀ (n : ℕ), n * seq.S n ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l954_95426


namespace NUMINAMATH_CALUDE_no_odd_solution_l954_95435

theorem no_odd_solution :
  ¬∃ (a b c d e f : ℕ), 
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧
    (1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_odd_solution_l954_95435


namespace NUMINAMATH_CALUDE_perfect_square_count_l954_95491

theorem perfect_square_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n ≤ 1500 ∧ ∃ k : ℕ, 21 * n = k^2) ∧ 
  S.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_count_l954_95491


namespace NUMINAMATH_CALUDE_abc_def_ratio_l954_95420

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 10) :
  a * b * c / (d * e * f) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l954_95420


namespace NUMINAMATH_CALUDE_pen_purchase_cost_l954_95418

/-- The cost of a single brand X pen -/
def brand_x_cost : ℚ := 4

/-- The cost of a single brand Y pen -/
def brand_y_cost : ℚ := 14/5

/-- The number of brand X pens purchased -/
def num_brand_x : ℕ := 8

/-- The total number of pens purchased -/
def total_pens : ℕ := 12

/-- The number of brand Y pens purchased -/
def num_brand_y : ℕ := total_pens - num_brand_x

/-- The total cost of all pens purchased -/
def total_cost : ℚ := num_brand_x * brand_x_cost + num_brand_y * brand_y_cost

theorem pen_purchase_cost : total_cost = 216/5 := by sorry

end NUMINAMATH_CALUDE_pen_purchase_cost_l954_95418


namespace NUMINAMATH_CALUDE_max_product_of_functions_l954_95436

/-- Given functions f and g on ℝ with specified ranges, prove that the maximum value of their product is 10 -/
theorem max_product_of_functions (f g : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.Icc (-5) 3) 
  (hg : ∀ x, g x ∈ Set.Icc (-2) 1) : 
  (∃ x, f x * g x = 10) ∧ (∀ x, f x * g x ≤ 10) := by
  sorry

#check max_product_of_functions

end NUMINAMATH_CALUDE_max_product_of_functions_l954_95436


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l954_95431

/-- A sequence of 10 natural numbers where each number from the third onwards
    is the sum of the two preceding numbers. -/
def FibonacciLikeSequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, i.val ≥ 2 → a i = a (i - 1) + a (i - 2)

theorem fourth_number_in_sequence
  (a : Fin 10 → ℕ)
  (h_seq : FibonacciLikeSequence a)
  (h_seventh : a 6 = 42)
  (h_ninth : a 8 = 110) :
  a 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l954_95431


namespace NUMINAMATH_CALUDE_six_digit_square_numbers_l954_95499

theorem six_digit_square_numbers : 
  ∀ n : ℕ, 
    (100000 ≤ n ∧ n < 1000000) → 
    (∃ m : ℕ, m < 1000 ∧ n = m^2) → 
    (n = 390625 ∨ n = 141376) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_square_numbers_l954_95499


namespace NUMINAMATH_CALUDE_largest_room_width_l954_95448

theorem largest_room_width (width smallest_width smallest_length largest_length area_difference : ℝ) :
  smallest_width = 15 →
  smallest_length = 8 →
  largest_length = 30 →
  area_difference = 1230 →
  width * largest_length - smallest_width * smallest_length = area_difference →
  width = 45 := by
sorry

end NUMINAMATH_CALUDE_largest_room_width_l954_95448


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l954_95463

theorem line_segment_endpoint (x y : ℝ) :
  let start : ℝ × ℝ := (2, 2)
  let length : ℝ := 8
  let slope : ℝ := 3/4
  y > 0 ∧
  (y - start.2) / (x - start.1) = slope ∧
  Real.sqrt ((x - start.1)^2 + (y - start.2)^2) = length →
  ((x = 2 + 4 * Real.sqrt 5475 / 25 ∧ y = 3/4 * (2 + 4 * Real.sqrt 5475 / 25) + 1/2) ∨
   (x = 2 - 4 * Real.sqrt 5475 / 25 ∧ y = 3/4 * (2 - 4 * Real.sqrt 5475 / 25) + 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l954_95463


namespace NUMINAMATH_CALUDE_calculator_game_result_l954_95474

/-- The number of participants in the game -/
def num_participants : ℕ := 60

/-- The operation applied to the first calculator -/
def op1 (n : ℕ) (x : ℤ) : ℤ := x ^ 3 ^ n

/-- The operation applied to the second calculator -/
def op2 (n : ℕ) (x : ℤ) : ℤ := x ^ (2 ^ n)

/-- The operation applied to the third calculator -/
def op3 (n : ℕ) (x : ℤ) : ℤ := (-1) ^ n * x

/-- The final sum of the numbers on the calculators after one complete round -/
def final_sum : ℤ := op1 num_participants 2 + op2 num_participants 0 + op3 num_participants (-1)

theorem calculator_game_result : final_sum = 2 ^ (3 ^ 60) + 1 := by
  sorry

end NUMINAMATH_CALUDE_calculator_game_result_l954_95474


namespace NUMINAMATH_CALUDE_circle_area_not_quadrupled_l954_95415

theorem circle_area_not_quadrupled (r : ℝ) (h : r > 0) : 
  ∃ k : ℝ, k ≠ 4 ∧ π * (r^2)^2 = k * (π * r^2) :=
sorry

end NUMINAMATH_CALUDE_circle_area_not_quadrupled_l954_95415


namespace NUMINAMATH_CALUDE_phone_number_theorem_l954_95424

def phone_number_count (n : ℕ) (k : ℕ) : ℕ := 2^n

theorem phone_number_theorem : phone_number_count 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_theorem_l954_95424


namespace NUMINAMATH_CALUDE_AE_length_l954_95446

-- Define the points A, B, C, D, E, and M on a line
variable (A B C D E M : ℝ)

-- Define the conditions
axiom divide_four_equal : B - A = C - B ∧ C - B = D - C ∧ D - C = E - D
axiom M_midpoint : M - A = E - M
axiom MC_length : M - C = 12

-- Theorem to prove
theorem AE_length : E - A = 48 := by
  sorry

end NUMINAMATH_CALUDE_AE_length_l954_95446


namespace NUMINAMATH_CALUDE_data_transmission_time_l954_95411

/-- Proves that the time to send 80 blocks of 400 chunks each at 160 chunks per second is 3 minutes -/
theorem data_transmission_time :
  let num_blocks : ℕ := 80
  let chunks_per_block : ℕ := 400
  let transmission_rate : ℕ := 160
  let total_chunks : ℕ := num_blocks * chunks_per_block
  let transmission_time_seconds : ℕ := total_chunks / transmission_rate
  let transmission_time_minutes : ℚ := transmission_time_seconds / 60
  transmission_time_minutes = 3 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l954_95411


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l954_95497

theorem sum_of_squares_zero (x y z : ℝ) 
  (h : x / (y + z) + y / (z + x) + z / (x + y) = 1) :
  x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l954_95497


namespace NUMINAMATH_CALUDE_library_experience_l954_95405

/-- Given two employees' years of experience satisfying certain conditions,
    prove that one employee has 10 years of experience. -/
theorem library_experience (b j : ℝ) 
  (h1 : j - 5 = 3 * (b - 5))
  (h2 : j = 2 * b) : 
  b = 10 := by sorry

end NUMINAMATH_CALUDE_library_experience_l954_95405


namespace NUMINAMATH_CALUDE_sqrt_11_parts_sum_l954_95406

theorem sqrt_11_parts_sum (x y : ℝ) : 
  (x = ⌊Real.sqrt 11⌋) → 
  (y = Real.sqrt 11 - x) → 
  (2 * x * y + y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_11_parts_sum_l954_95406


namespace NUMINAMATH_CALUDE_tan_equation_solutions_l954_95484

theorem tan_equation_solutions (x : ℝ) :
  -π < x ∧ x ≤ π ∧ 2 * Real.tan x - Real.sqrt 3 = 0 ↔ 
  x = Real.arctan (Real.sqrt 3 / 2) ∨ x = Real.arctan (Real.sqrt 3 / 2) - π :=
by sorry

end NUMINAMATH_CALUDE_tan_equation_solutions_l954_95484


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l954_95480

def num_flavors : ℕ := 4
def num_scoops : ℕ := 4

def ice_cream_combinations (n m : ℕ) : ℕ :=
  Nat.choose (n + m - 1) (n - 1)

theorem ice_cream_theorem :
  ice_cream_combinations num_flavors num_scoops = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l954_95480


namespace NUMINAMATH_CALUDE_positive_solution_equation_l954_95472

theorem positive_solution_equation : ∃ x : ℝ, 
  x > 0 ∧ 
  x = 21 + Real.sqrt 449 ∧ 
  (1 / 2) * (4 * x^2 - 2) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_equation_l954_95472


namespace NUMINAMATH_CALUDE_number_of_people_entered_l954_95429

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The total number of placards the basket can hold -/
def basket_capacity : ℕ := 823

/-- The number of people who entered the stadium -/
def people_entered : ℕ := basket_capacity / placards_per_person

/-- Theorem stating the number of people who entered the stadium -/
theorem number_of_people_entered : people_entered = 411 := by sorry

end NUMINAMATH_CALUDE_number_of_people_entered_l954_95429


namespace NUMINAMATH_CALUDE_inequality_proof_l954_95402

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 5*y)) + (y / (y + 5*x)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l954_95402


namespace NUMINAMATH_CALUDE_min_value_of_power_difference_l954_95409

theorem min_value_of_power_difference (m n : ℕ) : 12^m - 5^n ≥ 7 ∧ ∃ m n : ℕ, 12^m - 5^n = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_power_difference_l954_95409


namespace NUMINAMATH_CALUDE_max_words_is_16056_l954_95461

/-- Represents a language with two letters and words of maximum length 13 -/
structure TwoLetterLanguage where
  max_word_length : ℕ
  max_word_length_eq : max_word_length = 13

/-- Calculates the maximum number of words in the language -/
def max_words (L : TwoLetterLanguage) : ℕ :=
  2^14 - 2^7

/-- States that no concatenation of two words forms another word -/
axiom no_concat_word (L : TwoLetterLanguage) :
  ∀ (w1 w2 : String), (w1.length ≤ L.max_word_length ∧ w2.length ≤ L.max_word_length) →
    (w1 ++ w2).length > L.max_word_length

/-- Theorem: The maximum number of words in the language is 16056 -/
theorem max_words_is_16056 (L : TwoLetterLanguage) :
  max_words L = 16056 := by
  sorry

end NUMINAMATH_CALUDE_max_words_is_16056_l954_95461


namespace NUMINAMATH_CALUDE_travel_speed_l954_95479

/-- Given a distance of 195 km and a travel time of 3 hours, prove that the speed is 65 km/h -/
theorem travel_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 195 ∧ time = 3 ∧ speed = distance / time → speed = 65 := by
  sorry

end NUMINAMATH_CALUDE_travel_speed_l954_95479


namespace NUMINAMATH_CALUDE_total_students_in_line_l954_95498

theorem total_students_in_line 
  (students_in_front : ℕ) 
  (students_behind : ℕ) 
  (h1 : students_in_front = 15)
  (h2 : students_behind = 12) :
  students_in_front + 1 + students_behind = 28 :=
by sorry

end NUMINAMATH_CALUDE_total_students_in_line_l954_95498


namespace NUMINAMATH_CALUDE_phone_price_calculation_l954_95407

/-- Proves that given specific conditions on phone accessories and contract,
    the phone price that results in a total yearly cost of $3700 is $1000. -/
theorem phone_price_calculation (phone_price : ℝ) : 
  (∀ (monthly_contract case_cost headphones_cost : ℝ),
    monthly_contract = 200 ∧
    case_cost = 0.2 * phone_price ∧
    headphones_cost = 0.5 * case_cost ∧
    phone_price + 12 * monthly_contract + case_cost + headphones_cost = 3700) →
  phone_price = 1000 := by
  sorry

end NUMINAMATH_CALUDE_phone_price_calculation_l954_95407


namespace NUMINAMATH_CALUDE_find_b_l954_95495

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - 2 = 0 ∧ p.1 - 2*p.2 + 4 = 0}
def set2 (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3*p.1 + b}

-- State the theorem
theorem find_b : ∃ b : ℝ, set1 ⊂ set2 b → b = 2 := by sorry

end NUMINAMATH_CALUDE_find_b_l954_95495


namespace NUMINAMATH_CALUDE_orphanage_children_count_l954_95460

/-- Represents the number of cupcakes in a package -/
inductive PackageSize
| small : PackageSize
| large : PackageSize

/-- Returns the number of cupcakes in a package -/
def packageCupcakes (size : PackageSize) : ℕ :=
  match size with
  | PackageSize.small => 10
  | PackageSize.large => 15

/-- Calculates the total number of cupcakes from a given number of packages -/
def totalCupcakes (size : PackageSize) (numPackages : ℕ) : ℕ :=
  numPackages * packageCupcakes size

/-- Represents Jean's cupcake purchase and distribution plan -/
structure CupcakePlan where
  largePacks : ℕ
  smallPacks : ℕ
  childrenCount : ℕ

/-- Theorem: The number of children in the orphanage equals the total number of cupcakes -/
theorem orphanage_children_count (plan : CupcakePlan)
  (h1 : plan.largePacks = 4)
  (h2 : plan.smallPacks = 4)
  (h3 : plan.childrenCount = totalCupcakes PackageSize.large plan.largePacks + totalCupcakes PackageSize.small plan.smallPacks) :
  plan.childrenCount = 100 := by
  sorry

end NUMINAMATH_CALUDE_orphanage_children_count_l954_95460


namespace NUMINAMATH_CALUDE_max_value_x_4_minus_3x_l954_95449

theorem max_value_x_4_minus_3x :
  ∃ (max : ℝ), max = 4/3 ∧
  (∀ x : ℝ, 0 < x → x < 4/3 → x * (4 - 3 * x) ≤ max) ∧
  (∃ x : ℝ, 0 < x ∧ x < 4/3 ∧ x * (4 - 3 * x) = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_4_minus_3x_l954_95449


namespace NUMINAMATH_CALUDE_no_snow_probability_l954_95483

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l954_95483


namespace NUMINAMATH_CALUDE_article_pages_count_l954_95412

-- Define the constants
def total_word_limit : ℕ := 48000
def large_font_words_per_page : ℕ := 1800
def small_font_words_per_page : ℕ := 2400
def large_font_pages : ℕ := 4

-- Define the theorem
theorem article_pages_count :
  let words_in_large_font := large_font_pages * large_font_words_per_page
  let remaining_words := total_word_limit - words_in_large_font
  let small_font_pages := remaining_words / small_font_words_per_page
  large_font_pages + small_font_pages = 21 := by
sorry

end NUMINAMATH_CALUDE_article_pages_count_l954_95412


namespace NUMINAMATH_CALUDE_book_cost_price_l954_95410

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 270 → 
  profit_percentage = 20 → 
  selling_price = cost_price * (1 + profit_percentage / 100) → 
  cost_price = 225 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l954_95410


namespace NUMINAMATH_CALUDE_max_floors_is_fourteen_fourteen_floors_is_feasible_l954_95450

/-- Represents a building with elevators -/
structure Building where
  num_elevators : ℕ
  num_floors : ℕ
  stops_per_elevator : ℕ
  every_two_floors_connected : Bool

/-- The conditions of our specific building -/
def our_building : Building := {
  num_elevators := 7,
  num_floors := 14,  -- We'll prove this is the maximum
  stops_per_elevator := 6,
  every_two_floors_connected := true
}

/-- The theorem stating that 14 is the maximum number of floors -/
theorem max_floors_is_fourteen (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.every_two_floors_connected = true) :
  b.num_floors ≤ 14 := by
  sorry

/-- The theorem stating that 14 floors is feasible -/
theorem fourteen_floors_is_feasible (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.every_two_floors_connected = true) :
  ∃ (b' : Building), b'.num_floors = 14 ∧ 
    b'.num_elevators = b.num_elevators ∧ 
    b'.stops_per_elevator = b.stops_per_elevator ∧ 
    b'.every_two_floors_connected = b.every_two_floors_connected := by
  sorry

end NUMINAMATH_CALUDE_max_floors_is_fourteen_fourteen_floors_is_feasible_l954_95450


namespace NUMINAMATH_CALUDE_justin_jersey_problem_l954_95486

/-- Represents the problem of determining the number of long-sleeved jerseys Justin bought. -/
theorem justin_jersey_problem (long_sleeve_cost stripe_cost total_cost : ℕ) 
                               (stripe_count : ℕ) (total_spent : ℕ) : 
  long_sleeve_cost = 15 →
  stripe_cost = 10 →
  stripe_count = 2 →
  total_spent = 80 →
  ∃ long_sleeve_count : ℕ, 
    long_sleeve_count * long_sleeve_cost + stripe_count * stripe_cost = total_spent ∧
    long_sleeve_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_justin_jersey_problem_l954_95486


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_seven_l954_95428

theorem fraction_zero_implies_x_equals_seven :
  ∀ x : ℝ, (x^2 - 49) / (x + 7) = 0 → x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_seven_l954_95428
