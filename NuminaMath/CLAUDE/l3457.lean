import Mathlib

namespace NUMINAMATH_CALUDE_min_expression_value_l3457_345723

-- Define the type for our permutation
def Permutation := Fin 9 → Fin 9

-- Define our expression as a function of a permutation
def expression (p : Permutation) : ℕ :=
  let x₁ := (p 0).val + 1
  let x₂ := (p 1).val + 1
  let x₃ := (p 2).val + 1
  let y₁ := (p 3).val + 1
  let y₂ := (p 4).val + 1
  let y₃ := (p 5).val + 1
  let z₁ := (p 6).val + 1
  let z₂ := (p 7).val + 1
  let z₃ := (p 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃

-- State the theorem
theorem min_expression_value :
  (∀ p : Permutation, expression p ≥ 214) ∧
  (∃ p : Permutation, expression p = 214) := by
  sorry

end NUMINAMATH_CALUDE_min_expression_value_l3457_345723


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_formula_l3457_345753

/-- A cyclic quadrilateral is a quadrilateral inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  φ : ℝ
  S : ℝ

/-- The area formula for a cyclic quadrilateral -/
theorem cyclic_quadrilateral_area_formula (Q : CyclicQuadrilateral) :
  Q.S = Real.sqrt (Q.a * Q.b * Q.c * Q.d) * Real.sin Q.φ := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_formula_l3457_345753


namespace NUMINAMATH_CALUDE_line_through_points_l3457_345784

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The theorem statement -/
theorem line_through_points :
  let p1 : Point := ⟨2, 0⟩
  let p2 : Point := ⟨0, -3⟩
  let l : Line := ⟨1/2, -1/3, -1⟩
  pointOnLine p1 l ∧ pointOnLine p2 l :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l3457_345784


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3457_345719

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 9 → x = -77 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3457_345719


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l3457_345708

theorem isosceles_triangle_angle_measure (D E F : ℝ) : 
  D + E + F = 180 →  -- sum of angles in a triangle is 180°
  E = F →            -- isosceles triangle condition
  F = 3 * D →        -- angle F is three times angle D
  E = 540 / 7 :=     -- measure of angle E
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l3457_345708


namespace NUMINAMATH_CALUDE_halfway_point_l3457_345791

theorem halfway_point (a b : ℚ) (ha : a = 1/12) (hb : b = 1/10) :
  (a + b) / 2 = 11/120 := by
sorry

end NUMINAMATH_CALUDE_halfway_point_l3457_345791


namespace NUMINAMATH_CALUDE_geralds_toy_cars_l3457_345755

theorem geralds_toy_cars (initial_cars : ℕ) : 
  (initial_cars : ℚ) * (3/4 : ℚ) = 15 → initial_cars = 20 := by
  sorry

end NUMINAMATH_CALUDE_geralds_toy_cars_l3457_345755


namespace NUMINAMATH_CALUDE_add_like_terms_l3457_345741

theorem add_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_add_like_terms_l3457_345741


namespace NUMINAMATH_CALUDE_cab_driver_income_l3457_345760

theorem cab_driver_income (incomes : Fin 5 → ℕ) 
  (h1 : incomes 1 = 50)
  (h2 : incomes 2 = 60)
  (h3 : incomes 3 = 65)
  (h4 : incomes 4 = 70)
  (h_avg : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 58) :
  incomes 0 = 45 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3457_345760


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3457_345738

theorem square_perimeters_sum (x y : ℝ) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3457_345738


namespace NUMINAMATH_CALUDE_history_books_count_l3457_345768

theorem history_books_count (total : ℕ) (reading_fraction : ℚ) (math_fraction : ℚ) :
  total = 10 →
  reading_fraction = 2 / 5 →
  math_fraction = 3 / 10 →
  let reading := (reading_fraction * total).floor
  let math := (math_fraction * total).floor
  let science := math - 1
  let non_history := reading + math + science
  let history := total - non_history
  history = 1 := by sorry

end NUMINAMATH_CALUDE_history_books_count_l3457_345768


namespace NUMINAMATH_CALUDE_cathy_win_probability_l3457_345733

/-- Represents a player in the die-rolling game -/
inductive Player : Type
| Ana : Player
| Bob : Player
| Cathy : Player

/-- The number of sides on the die -/
def dieSides : ℕ := 6

/-- The winning number on the die -/
def winningNumber : ℕ := 6

/-- The probability of rolling the winning number -/
def winProbability : ℚ := 1 / dieSides

/-- The probability of not rolling the winning number -/
def loseProbability : ℚ := 1 - winProbability

/-- The number of players before Cathy -/
def playersBeforeCathy : ℕ := 2

/-- Theorem stating the probability of Cathy winning -/
theorem cathy_win_probability :
  let p : ℚ := winProbability
  let q : ℚ := loseProbability
  (q^playersBeforeCathy * p) / (1 - q^3) = 25 / 91 := by sorry

end NUMINAMATH_CALUDE_cathy_win_probability_l3457_345733


namespace NUMINAMATH_CALUDE_number_satisfies_condition_l3457_345725

theorem number_satisfies_condition : ∃ n : ℕ, n = 250 ∧ (5 * n) / 8 = 156 ∧ (5 * n) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_condition_l3457_345725


namespace NUMINAMATH_CALUDE_log_stack_sum_l3457_345778

/-- Sum of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

theorem log_stack_sum :
  let a₁ : ℕ := 5  -- first term (top row)
  let aₙ : ℕ := 15 -- last term (bottom row)
  let n : ℕ := aₙ - a₁ + 1 -- number of terms
  sum_arithmetic_sequence a₁ aₙ n = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3457_345778


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l3457_345712

/-- An arithmetic-geometric sequence is defined by the recurrence relation u_{n+1} = a * u_n + b -/
def arithmetic_geometric_sequence (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 1) = a * u n + b

/-- The closed form of the nth term of an arithmetic-geometric sequence -/
theorem arithmetic_geometric_sequence_closed_form (a b : ℝ) (u : ℕ → ℝ) (h : arithmetic_geometric_sequence a b u) :
  ∀ n, u n = a^n * u 0 + (a^n - 1) / (a - 1) * b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l3457_345712


namespace NUMINAMATH_CALUDE_final_value_is_15_l3457_345774

def loop_operation (x : ℕ) (s : ℕ) : ℕ := s * x + 1

def iterate_n_times (n : ℕ) (x : ℕ) (initial_s : ℕ) : ℕ :=
  match n with
  | 0 => initial_s
  | m + 1 => loop_operation x (iterate_n_times m x initial_s)

theorem final_value_is_15 :
  let x : ℕ := 2
  let initial_s : ℕ := 0
  let n : ℕ := 4
  iterate_n_times n x initial_s = 15 := by
  sorry

#eval iterate_n_times 4 2 0

end NUMINAMATH_CALUDE_final_value_is_15_l3457_345774


namespace NUMINAMATH_CALUDE_total_animals_bought_l3457_345789

/-- The number of guppies Rick bought -/
def rickGuppies : ℕ := 30

/-- The number of clowns Tim bought -/
def timClowns : ℕ := 2 * rickGuppies

/-- The number of tetras I bought -/
def myTetras : ℕ := 4 * timClowns

/-- The total number of animals bought -/
def totalAnimals : ℕ := myTetras + timClowns + rickGuppies

theorem total_animals_bought : totalAnimals = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_bought_l3457_345789


namespace NUMINAMATH_CALUDE_area_of_region_l3457_345728

/-- The equation of the curve enclosing the region -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + 2*y + 50 = 25 + 7*y - y^2

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 5

/-- The region above the line -/
def region_above_line (x y : ℝ) : Prop :=
  y > x - 5

/-- The theorem stating the area of the region -/
theorem area_of_region : 
  ∃ (A : ℝ), A = 25 * Real.pi / 4 ∧ 
  (∀ x y : ℝ, curve_equation x y ∧ region_above_line x y → 
    (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ A / Real.pi}) :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l3457_345728


namespace NUMINAMATH_CALUDE_email_cleaning_l3457_345748

/-- Represents the email cleaning process and proves the number of emails deleted in the first round -/
theorem email_cleaning (initial_emails : ℕ) : 
  -- After first round, emails remain the same (deleted some, received 15)
  ∃ (first_round_deleted : ℕ), initial_emails = initial_emails - first_round_deleted + 15 →
  -- After second round, 20 deleted, 5 received
  initial_emails - 20 + 5 = 30 →
  -- Final inbox has 30 emails (15 + 5 + 10 new ones)
  30 = 15 + 5 + 10 →
  -- Prove that first_round_deleted is 0
  first_round_deleted = 0 := by
  sorry

end NUMINAMATH_CALUDE_email_cleaning_l3457_345748


namespace NUMINAMATH_CALUDE_bonus_interval_proof_l3457_345796

/-- The number of cards after which Brady gets a bonus -/
def bonus_interval : ℕ := sorry

/-- The pay per card in cents -/
def pay_per_card : ℕ := 70

/-- The bonus amount in cents -/
def bonus_amount : ℕ := 1000

/-- The total number of cards transcribed -/
def total_cards : ℕ := 200

/-- The total earnings in cents including bonuses -/
def total_earnings : ℕ := 16000

theorem bonus_interval_proof : 
  bonus_interval = 100 ∧
  total_earnings = total_cards * pay_per_card + 
    (total_cards / bonus_interval) * bonus_amount :=
by sorry

end NUMINAMATH_CALUDE_bonus_interval_proof_l3457_345796


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3457_345786

theorem triangle_angle_measure (A B C : Real) (BC AC : Real) :
  BC = Real.sqrt 3 →
  AC = Real.sqrt 2 →
  A = π / 3 →
  B = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3457_345786


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3457_345769

theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x - Real.sqrt 3 * y + m = 0 → 
    (x^2 + y^2 - 2*y - 2 = 0 → 
      ∃ x' y' : ℝ, x' - Real.sqrt 3 * y' + m = 0 ∧ 
        x'^2 + y'^2 - 2*y' - 2 = 0 ∧ 
        ∀ x'' y'' : ℝ, x'' - Real.sqrt 3 * y'' + m = 0 → 
          x''^2 + y''^2 - 2*y'' - 2 ≤ 0)) ↔ 
  (m = -Real.sqrt 3 ∨ m = 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3457_345769


namespace NUMINAMATH_CALUDE_min_lines_correct_l3457_345702

/-- A line in a 2D plane represented by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The quadrants a line passes through -/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines required to guarantee at least two lines pass through the same quadrants -/
def min_lines : ℕ := 7

/-- Theorem stating that the minimum number of lines is correct -/
theorem min_lines_correct :
  ∀ (lines : Finset Line),
    lines.card ≥ min_lines →
    ∃ (l1 l2 : Line), l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ quadrants l1 = quadrants l2 :=
  sorry

end NUMINAMATH_CALUDE_min_lines_correct_l3457_345702


namespace NUMINAMATH_CALUDE_real_root_implies_m_value_l3457_345750

theorem real_root_implies_m_value (m : ℂ) :
  (∃ x : ℝ, x^2 + (1 + 2*I)*x - 2*(m + 1) = 0) →
  (∃ b : ℝ, m = b*I) →
  (m = I ∨ m = -2*I) := by
sorry

end NUMINAMATH_CALUDE_real_root_implies_m_value_l3457_345750


namespace NUMINAMATH_CALUDE_polygon_sides_l3457_345763

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  (180 * (n - 2) = 5 * 360) → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3457_345763


namespace NUMINAMATH_CALUDE_sum_of_digits_A_l3457_345798

def A (n : ℕ) : ℕ := 
  match n with
  | 0 => 9
  | m + 1 => A m * (10^(2^(m+1)) - 1)

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem sum_of_digits_A (n : ℕ) : sumOfDigits (A n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_A_l3457_345798


namespace NUMINAMATH_CALUDE_janette_camping_duration_l3457_345735

/-- Calculates the number of days Janette went camping based on her beef jerky consumption --/
def camping_days (
  initial_jerky : ℕ
  ) (daily_consumption : ℕ
  ) (final_jerky : ℕ
  ) : ℕ :=
  (initial_jerky - 2 * final_jerky) / daily_consumption

/-- Proves that Janette went camping for 5 days --/
theorem janette_camping_duration :
  camping_days 40 4 10 = 5 := by
  sorry

#eval camping_days 40 4 10

end NUMINAMATH_CALUDE_janette_camping_duration_l3457_345735


namespace NUMINAMATH_CALUDE_sibling_product_l3457_345709

/-- Represents a family with a specific sibling structure -/
structure Family :=
  (total_sisters : ℕ)
  (total_brothers : ℕ)

/-- Calculates the number of sisters for a given family member -/
def sisters_count (f : Family) : ℕ := f.total_sisters - 1

/-- Calculates the number of brothers for a given family member -/
def brothers_count (f : Family) : ℕ := f.total_brothers

theorem sibling_product (f : Family) 
  (h1 : f.total_sisters = 5) 
  (h2 : f.total_brothers = 7) : 
  sisters_count f * brothers_count f = 24 := by
  sorry

#check sibling_product

end NUMINAMATH_CALUDE_sibling_product_l3457_345709


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3457_345781

theorem bowling_ball_weight :
  let b := (108 / 3) * (4 / 10)  -- weight of one bowling ball
  let c := 108 / 3               -- weight of one canoe
  (10 * b = 4 * c) →             -- 10 bowling balls weigh the same as 4 canoes
  (3 * c = 108) →                -- 3 canoes weigh 108 pounds
  b = 14.4                       -- weight of one bowling ball is 14.4 pounds
:= by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3457_345781


namespace NUMINAMATH_CALUDE_paul_takes_remaining_l3457_345766

def initial_sweets : ℕ := 22

def jack_takes (total : ℕ) : ℕ := total / 2 + 4

theorem paul_takes_remaining (paul_takes : ℕ) : 
  paul_takes = initial_sweets - jack_takes initial_sweets := by
  sorry

end NUMINAMATH_CALUDE_paul_takes_remaining_l3457_345766


namespace NUMINAMATH_CALUDE_fraction_simplification_l3457_345773

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3457_345773


namespace NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l3457_345767

theorem traffic_class_drunk_drivers :
  ∀ (drunk_drivers speeders seatbelt_violators texting_drivers : ℕ),
    speeders = 7 * drunk_drivers - 3 →
    seatbelt_violators = 2 * drunk_drivers →
    texting_drivers = (speeders / 2) + 5 →
    drunk_drivers + speeders + seatbelt_violators + texting_drivers = 180 →
    drunk_drivers = 13 := by
  sorry

end NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l3457_345767


namespace NUMINAMATH_CALUDE_volume_per_gram_l3457_345770

/-- Given a substance with specific properties, calculate its volume per gram -/
theorem volume_per_gram (density : ℝ) (mass_per_cubic_meter : ℝ) (grams_per_kg : ℝ) (cc_per_cubic_meter : ℝ) :
  density = mass_per_cubic_meter ∧ 
  grams_per_kg = 1000 ∧ 
  cc_per_cubic_meter = 1000000 →
  (1 / density) * grams_per_kg * cc_per_cubic_meter = 10 := by
  sorry

#check volume_per_gram

end NUMINAMATH_CALUDE_volume_per_gram_l3457_345770


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3457_345717

theorem absolute_value_simplification : |-4^2 + 5 - 2| = 13 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3457_345717


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3457_345777

theorem degree_to_radian_conversion (π : Real) : 
  (180 : Real) = π → (750 : Real) = (25 / 6 : Real) * π := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3457_345777


namespace NUMINAMATH_CALUDE_fruit_drawing_orders_l3457_345754

def basket : Finset String := {"apple", "peach", "pear", "melon"}

theorem fruit_drawing_orders :
  (basket.card * (basket.card - 1) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drawing_orders_l3457_345754


namespace NUMINAMATH_CALUDE_greatest_integer_value_l3457_345747

theorem greatest_integer_value (x : ℤ) : (∀ y : ℤ, 3 * |y| - 1 ≤ 8 → y ≤ x) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_value_l3457_345747


namespace NUMINAMATH_CALUDE_team_incorrect_answers_17_l3457_345780

/-- Represents a math contest team with two members -/
structure MathTeam where
  total_questions : Nat
  riley_mistakes : Nat
  ofelia_additional_correct : Nat

/-- Calculates the total number of incorrect answers for a math team -/
def total_incorrect_answers (team : MathTeam) : Nat :=
  let riley_correct := team.total_questions - team.riley_mistakes
  let ofelia_correct := riley_correct / 2 + team.ofelia_additional_correct
  let ofelia_incorrect := team.total_questions - ofelia_correct
  team.riley_mistakes + ofelia_incorrect

/-- Theorem stating that for the given conditions, the team's total incorrect answers is 17 -/
theorem team_incorrect_answers_17 :
  ∃ (team : MathTeam),
    team.total_questions = 35 ∧
    team.riley_mistakes = 3 ∧
    team.ofelia_additional_correct = 5 ∧
    total_incorrect_answers team = 17 := by
  sorry

end NUMINAMATH_CALUDE_team_incorrect_answers_17_l3457_345780


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3457_345759

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of side BC -/
  bc : ℝ
  /-- The perpendicular distance from point P on BC to side AB -/
  p_to_ab : ℝ
  /-- The perpendicular distance from point P on BC to side AC -/
  p_to_ac : ℝ
  /-- Assertion that BC = 65 -/
  h_bc : bc = 65
  /-- Assertion that the perpendicular distance from P to AB is 24 -/
  h_p_to_ab : p_to_ab = 24
  /-- Assertion that the perpendicular distance from P to AC is 36 -/
  h_p_to_ac : p_to_ac = 36

/-- The area of the isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ := 2535

/-- Theorem stating that the area of the isosceles triangle is 2535 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : area t = 2535 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3457_345759


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3457_345729

theorem smallest_absolute_value : 
  let numbers : Finset ℚ := {-1/2, -2/3, 4, -5}
  ∀ x ∈ numbers, x ≠ -1/2 → abs (-1/2) < abs x :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3457_345729


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3457_345716

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3457_345716


namespace NUMINAMATH_CALUDE_strawberry_picking_total_l3457_345761

/-- The total number of strawberries picked by a group of friends -/
def total_strawberries (lilibeth_baskets mia_baskets jake_baskets natalie_baskets 
  layla_baskets oliver_baskets ava_baskets : ℕ) 
  (lilibeth_per_basket mia_per_basket jake_per_basket natalie_per_basket 
  layla_per_basket oliver_per_basket ava_per_basket : ℕ) : ℕ :=
  lilibeth_baskets * lilibeth_per_basket +
  mia_baskets * mia_per_basket +
  jake_baskets * jake_per_basket +
  natalie_baskets * natalie_per_basket +
  layla_baskets * layla_per_basket +
  oliver_baskets * oliver_per_basket +
  ava_baskets * ava_per_basket

/-- Theorem stating the total number of strawberries picked by the friends -/
theorem strawberry_picking_total : 
  total_strawberries 6 3 4 5 2 7 6 50 65 45 55 80 40 60 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_total_l3457_345761


namespace NUMINAMATH_CALUDE_container_volume_increase_three_gallon_to_twentyfour_gallon_l3457_345711

theorem container_volume_increase (original_volume : ℝ) (scale_factor : ℝ) :
  original_volume > 0 →
  scale_factor = 2 →
  scale_factor * scale_factor * scale_factor * original_volume = 8 * original_volume :=
by sorry

theorem three_gallon_to_twentyfour_gallon :
  let original_volume : ℝ := 3
  let scale_factor : ℝ := 2
  let new_volume : ℝ := scale_factor * scale_factor * scale_factor * original_volume
  new_volume = 24 :=
by sorry

end NUMINAMATH_CALUDE_container_volume_increase_three_gallon_to_twentyfour_gallon_l3457_345711


namespace NUMINAMATH_CALUDE_area_circle_outside_square_l3457_345799

/-- The area inside a circle but outside a square, when both share the same center -/
theorem area_circle_outside_square (r : ℝ) (s : ℝ) (h : r = Real.sqrt 3 / 3) (hs : s = 2) :
  π * r^2 = π / 3 :=
sorry

end NUMINAMATH_CALUDE_area_circle_outside_square_l3457_345799


namespace NUMINAMATH_CALUDE_root_difference_ratio_l3457_345787

noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 3
noncomputable def f₂ (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - b
noncomputable def f₃ (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + (2 - 2*a)*x - 6 - b
noncomputable def f₄ (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + (4 - a)*x - 3 - 2*b

noncomputable def A (a : ℝ) : ℝ := Real.sqrt (a^2 + 12)
noncomputable def B (b : ℝ) : ℝ := Real.sqrt (4 + 4*b)
noncomputable def C (a b : ℝ) : ℝ := Real.sqrt ((2 - 2*a)^2 + 12*(6 + b)) / 3
noncomputable def D (a b : ℝ) : ℝ := Real.sqrt ((4 - a)^2 + 12*(3 + 2*b)) / 3

theorem root_difference_ratio (a b : ℝ) (h : C a b ≠ D a b) :
  (A a ^ 2 - B b ^ 2) / (C a b ^ 2 - D a b ^ 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l3457_345787


namespace NUMINAMATH_CALUDE_k_range_l3457_345707

theorem k_range (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x - 1 < 0) → -4 < k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l3457_345707


namespace NUMINAMATH_CALUDE_outside_circle_inequality_l3457_345785

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop :=
  (p.x + c.D/2)^2 + (p.y + c.E/2)^2 > (c.D^2 + c.E^2 - 4*c.F)/4

/-- The main theorem -/
theorem outside_circle_inequality (p : Point) (c : Circle) :
  is_outside p c → p.x^2 + p.y^2 + c.D*p.x + c.E*p.y + c.F > 0 := by
  sorry

end NUMINAMATH_CALUDE_outside_circle_inequality_l3457_345785


namespace NUMINAMATH_CALUDE_sum_zero_implies_product_nonpositive_l3457_345751

theorem sum_zero_implies_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) :
  a * b + a * c + b * c ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sum_zero_implies_product_nonpositive_l3457_345751


namespace NUMINAMATH_CALUDE_existence_and_not_forall_l3457_345794

theorem existence_and_not_forall : 
  (∃ x : ℝ, x > 2) ∧ ¬(∀ x : ℝ, x^3 > x^2) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_and_not_forall_l3457_345794


namespace NUMINAMATH_CALUDE_bottle_production_l3457_345714

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 16 such machines will produce 2880 bottles in 4 minutes. -/
theorem bottle_production 
  (machines : ℕ → ℕ) -- number of machines
  (bottles_per_minute : ℕ → ℕ) -- bottles produced per minute
  (h1 : machines 1 = 6)
  (h2 : bottles_per_minute 1 = 270)
  (h3 : ∀ n : ℕ, bottles_per_minute n = n * (bottles_per_minute 1 / machines 1)) :
  bottles_per_minute 16 * 4 = 2880 :=
by sorry

end NUMINAMATH_CALUDE_bottle_production_l3457_345714


namespace NUMINAMATH_CALUDE_position_of_2022_l3457_345737

-- Define the sequence
def sequence_element (n : ℕ) : ℕ :=
  if n % 3 = 0 then 4 * ((n - 1) / 3) + 3
  else if n % 3 = 1 then 4 * (n / 3) + 1
  else 4 * (n / 3) + 2

-- Define the group number for a given element
def group_number (x : ℕ) : ℕ :=
  (x - 1) / 3 + 1

-- Define the position within a group for a given element
def position_in_group (x : ℕ) : ℕ :=
  (x - 1) % 3 + 1

-- Theorem statement
theorem position_of_2022 :
  group_number 2022 = 506 ∧ position_in_group 2022 = 2 :=
sorry

end NUMINAMATH_CALUDE_position_of_2022_l3457_345737


namespace NUMINAMATH_CALUDE_uncertain_relationship_l3457_345726

-- Define a type for planes in 3D space
variable (Plane : Type)

-- Define the perpendicular relation between planes
variable (perp : Plane → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersecting (neither perpendicular nor parallel) relation between planes
variable (intersecting : Plane → Plane → Prop)

-- State the theorem
theorem uncertain_relationship 
  (a₁ a₂ a₃ a₄ : Plane) 
  (distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) 
  (h₁ : perp a₁ a₂) 
  (h₂ : perp a₂ a₃) 
  (h₃ : perp a₃ a₄) : 
  ¬(∀ a₁ a₄, perp a₁ a₄ ∨ parallel a₁ a₄ ∨ intersecting a₁ a₄) :=
by sorry

end NUMINAMATH_CALUDE_uncertain_relationship_l3457_345726


namespace NUMINAMATH_CALUDE_tina_postcards_per_day_l3457_345797

/-- The number of postcards Tina can make in a day -/
def postcards_per_day : ℕ := 30

/-- The price of each postcard in dollars -/
def price_per_postcard : ℕ := 5

/-- The number of days Tina sold postcards -/
def days_sold : ℕ := 6

/-- The total amount earned in dollars -/
def total_earned : ℕ := 900

theorem tina_postcards_per_day :
  postcards_per_day * price_per_postcard * days_sold = total_earned :=
by sorry

end NUMINAMATH_CALUDE_tina_postcards_per_day_l3457_345797


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_with_pair_l3457_345782

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes,
    where two specific objects must always be together -/
def distributeWithPair (n k : ℕ) : ℕ := k * (distribute (n - 1) k)

theorem five_balls_three_boxes_with_pair :
  distributeWithPair 5 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_with_pair_l3457_345782


namespace NUMINAMATH_CALUDE_negative_two_cubed_l3457_345706

theorem negative_two_cubed : (-2 : ℤ)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_l3457_345706


namespace NUMINAMATH_CALUDE_unique_solution_system_l3457_345731

theorem unique_solution_system (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! (x y z : ℝ), 
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧
    y = (c + a) / 2 ∧
    z = (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3457_345731


namespace NUMINAMATH_CALUDE_tile_arrangement_theorem_l3457_345746

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Checks if a new configuration is valid given an original configuration -/
def is_valid_new_configuration (original new : TileConfiguration) : Prop :=
  new.tiles = original.tiles + 3 ∧ 
  new.perimeter < original.perimeter + 6

theorem tile_arrangement_theorem : ∃ (original new : TileConfiguration), 
  original.tiles = 10 ∧ 
  original.perimeter = 18 ∧
  new.tiles = 13 ∧
  new.perimeter = 17 ∧
  is_valid_new_configuration original new :=
sorry

end NUMINAMATH_CALUDE_tile_arrangement_theorem_l3457_345746


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l3457_345775

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
    (h1 : seq.S (m - 1) = -2)
    (h2 : seq.S m = 0)
    (h3 : seq.S (m + 1) = 3) :
  m = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l3457_345775


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3457_345701

theorem greatest_integer_radius (r : ℕ) : 
  (∀ k : ℕ, k > r → ¬(Real.pi * k^2 < 30 * Real.pi ∧ 2 * Real.pi * k > 10 * Real.pi)) ∧
  (Real.pi * r^2 < 30 * Real.pi ∧ 2 * Real.pi * r > 10 * Real.pi) →
  r = 5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3457_345701


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3457_345740

theorem simplify_polynomial (z : ℝ) : (4 - 5*z) - (2 + 7*z - z^2) = z^2 - 12*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3457_345740


namespace NUMINAMATH_CALUDE_intersection_M_N_l3457_345790

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3457_345790


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3457_345744

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 12y -/
def Parabola := {p : Point | p.x^2 = 12 * p.y}

/-- Represents a line y = kx + m -/
def Line (k m : ℝ) := {p : Point | p.y = k * p.x + m}

/-- The focus of the parabola -/
def focus : Point := ⟨0, 3⟩

theorem parabola_line_intersection (k m : ℝ) (h_k : k > 0) :
  ∃ A B : Point,
    A ∈ Parabola ∧
    B ∈ Parabola ∧
    A ∈ Line k m ∧
    B ∈ Line k m ∧
    focus ∈ Line k m ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = 36^2 →
    k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3457_345744


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3457_345721

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →
  (S 1 + 2 * S 5 = 3 * S 3) →
  q = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3457_345721


namespace NUMINAMATH_CALUDE_cone_base_radius_l3457_345720

/-- Represents a cone with given surface area and net shape -/
structure Cone where
  surfaceArea : ℝ
  netIsSemicircle : Prop

/-- Theorem: Given a cone with surface area 12π cm² and semicircular net, its base radius is 2 cm -/
theorem cone_base_radius (c : Cone) 
  (h1 : c.surfaceArea = 12 * Real.pi) 
  (h2 : c.netIsSemicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r * r * Real.pi + 2 * r * r * Real.pi = c.surfaceArea := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3457_345720


namespace NUMINAMATH_CALUDE_lauras_change_l3457_345772

def pants_quantity : ℕ := 2
def pants_price : ℕ := 54
def shirts_quantity : ℕ := 4
def shirts_price : ℕ := 33
def amount_given : ℕ := 250

def total_cost : ℕ := pants_quantity * pants_price + shirts_quantity * shirts_price

theorem lauras_change :
  amount_given - total_cost = 10 := by sorry

end NUMINAMATH_CALUDE_lauras_change_l3457_345772


namespace NUMINAMATH_CALUDE_bobs_speed_limit_l3457_345739

/-- Proves that Bob's speed must be less than 40 mph for Alice to arrive before him --/
theorem bobs_speed_limit (distance : ℝ) (alice_delay : ℝ) (alice_min_speed : ℝ) 
  (h_distance : distance = 180)
  (h_delay : alice_delay = 1/2)
  (h_min_speed : alice_min_speed = 45) :
  ∀ v : ℝ, (∃ (alice_speed : ℝ), 
    alice_speed > alice_min_speed ∧ 
    distance / alice_speed < distance / v - alice_delay) → 
  v < 40 := by
  sorry

end NUMINAMATH_CALUDE_bobs_speed_limit_l3457_345739


namespace NUMINAMATH_CALUDE_expression_1_expression_2_expression_3_expression_4_expression_5_l3457_345718

-- Expression 1
theorem expression_1 : 0.11 * 1.8 + 8.2 * 0.11 = 1.1 := by sorry

-- Expression 2
theorem expression_2 : 0.8 * (3.2 - 2.99 / 2.3) = 1.52 := by sorry

-- Expression 3
theorem expression_3 : 3.5 - 3.5 * 0.98 = 0.07 := by sorry

-- Expression 4
theorem expression_4 : 12.5 * 2.5 * 3.2 = 100 := by sorry

-- Expression 5
theorem expression_5 : (8.1 - 5.4) / 3.6 + 85.7 = 86.45 := by sorry

end NUMINAMATH_CALUDE_expression_1_expression_2_expression_3_expression_4_expression_5_l3457_345718


namespace NUMINAMATH_CALUDE_even_number_representation_l3457_345736

theorem even_number_representation (x y : ℕ) : 
  ∃! n : ℕ, 2 * n = (x + y)^2 + 3 * x + y := by sorry

end NUMINAMATH_CALUDE_even_number_representation_l3457_345736


namespace NUMINAMATH_CALUDE_san_antonio_austin_bus_passes_l3457_345732

/-- Represents the time interval between bus departures in hours -/
def departure_interval : ℕ := 2

/-- Represents the duration of the journey between cities in hours -/
def journey_duration : ℕ := 7

/-- Represents the offset between San Antonio and Austin departures in hours -/
def departure_offset : ℕ := 1

/-- Calculates the number of buses passed during the journey -/
def buses_passed : ℕ :=
  journey_duration / departure_interval + 1

theorem san_antonio_austin_bus_passes :
  buses_passed = 4 :=
sorry

end NUMINAMATH_CALUDE_san_antonio_austin_bus_passes_l3457_345732


namespace NUMINAMATH_CALUDE_combined_teaching_experience_l3457_345757

/-- Represents the teaching experience of the four teachers --/
structure TeachingExperience where
  james : ℕ
  sarah : ℕ
  robert : ℕ
  emily : ℕ

/-- Calculates the combined teaching experience --/
def combined_experience (te : TeachingExperience) : ℕ :=
  te.james + te.sarah + te.robert + te.emily

/-- Theorem stating the combined teaching experience --/
theorem combined_teaching_experience :
  ∃ (te : TeachingExperience),
    te.james = 40 ∧
    te.sarah = te.james - 10 ∧
    te.robert = 2 * te.sarah ∧
    te.emily = 3 * te.sarah - 5 ∧
    combined_experience te = 215 := by
  sorry

end NUMINAMATH_CALUDE_combined_teaching_experience_l3457_345757


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3457_345771

theorem inequality_equivalence (x : ℝ) : 
  (2*x + 1) / 3 ≤ (5*x - 1) / 2 - 1 ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3457_345771


namespace NUMINAMATH_CALUDE_injective_impl_neq_injective_impl_unique_preimage_l3457_345764

variable {A B : Type*} (f : A → B)

/-- Definition of injective function -/
def Injective (f : A → B) : Prop :=
  ∀ x₁ x₂ : A, f x₁ = f x₂ → x₁ = x₂

theorem injective_impl_neq (hf : Injective f) :
    ∀ x₁ x₂ : A, x₁ ≠ x₂ → f x₁ ≠ f x₂ := by sorry

theorem injective_impl_unique_preimage (hf : Injective f) :
    ∀ b : B, ∃! a : A, f a = b := by sorry

end NUMINAMATH_CALUDE_injective_impl_neq_injective_impl_unique_preimage_l3457_345764


namespace NUMINAMATH_CALUDE_no_constant_term_in_expansion_l3457_345752

theorem no_constant_term_in_expansion :
  let expression := (fun x => (5 * x^2 + 2 / x)^8)
  ∀ c : ℝ, (∀ x : ℝ, x ≠ 0 → expression x = c) → c = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_constant_term_in_expansion_l3457_345752


namespace NUMINAMATH_CALUDE_no_real_solutions_l3457_345795

theorem no_real_solutions :
  ∀ x y : ℝ, 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3457_345795


namespace NUMINAMATH_CALUDE_find_t_l3457_345783

theorem find_t (x y z t : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x + y + z + t) / 4 = 12 →
  t = 18 := by
sorry

end NUMINAMATH_CALUDE_find_t_l3457_345783


namespace NUMINAMATH_CALUDE_power_function_special_case_l3457_345762

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x^a

-- State the theorem
theorem power_function_special_case (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 4 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_function_special_case_l3457_345762


namespace NUMINAMATH_CALUDE_largest_perimeter_right_triangle_l3457_345779

theorem largest_perimeter_right_triangle (x : ℕ) : 
  -- Right triangle condition (using Pythagorean theorem)
  x * x ≤ 8 * 8 + 9 * 9 →
  -- Triangle inequality conditions
  8 + 9 > x →
  8 + x > 9 →
  9 + x > 8 →
  -- Perimeter definition
  let perimeter := 8 + 9 + x
  -- Theorem statement
  perimeter ≤ 29 ∧ ∃ y : ℕ, y * y ≤ 8 * 8 + 9 * 9 ∧ 8 + 9 > y ∧ 8 + y > 9 ∧ 9 + y > 8 ∧ 8 + 9 + y = 29 :=
by sorry

end NUMINAMATH_CALUDE_largest_perimeter_right_triangle_l3457_345779


namespace NUMINAMATH_CALUDE_snowfall_total_l3457_345710

theorem snowfall_total (morning_snowfall afternoon_snowfall : ℝ) 
  (h1 : morning_snowfall = 0.12)
  (h2 : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 := by
sorry

end NUMINAMATH_CALUDE_snowfall_total_l3457_345710


namespace NUMINAMATH_CALUDE_cow_manure_plant_height_cow_manure_plant_height_is_90_l3457_345724

theorem cow_manure_plant_height : ℝ → Prop :=
  fun height_cow_manure =>
    let height_control : ℝ := 36
    let height_bone_meal : ℝ := 1.25 * height_control
    height_cow_manure = 2 * height_bone_meal

-- The proof
theorem cow_manure_plant_height_is_90 : cow_manure_plant_height 90 := by
  sorry

end NUMINAMATH_CALUDE_cow_manure_plant_height_cow_manure_plant_height_is_90_l3457_345724


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l3457_345758

/-- Calculates the number of books Robert can read given his reading speed, book size, and available time. -/
def books_read (reading_speed : ℕ) (pages_per_book : ℕ) (available_time : ℕ) : ℕ :=
  (reading_speed * available_time) / pages_per_book

theorem robert_reading_capacity : books_read 200 400 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l3457_345758


namespace NUMINAMATH_CALUDE_largest_x_abs_equation_l3457_345730

theorem largest_x_abs_equation : ∃ (x_max : ℝ), (∀ (x : ℝ), |x - 5| = 12 → x ≤ x_max) ∧ |x_max - 5| = 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_abs_equation_l3457_345730


namespace NUMINAMATH_CALUDE_salary_restoration_l3457_345705

theorem salary_restoration (initial_salary : ℝ) (h : initial_salary > 0) :
  let reduced_salary := 0.7 * initial_salary
  (initial_salary / reduced_salary - 1) * 100 = 42.86 := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l3457_345705


namespace NUMINAMATH_CALUDE_max_e_is_one_l3457_345745

/-- The sequence b_n defined as (8^n - 1) / 7 -/
def b (n : ℕ) : ℤ := (8^n - 1) / 7

/-- The greatest common divisor of b_n and b_(n+1) -/
def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

/-- Theorem: The maximum value of e_n is always 1 -/
theorem max_e_is_one : ∀ n : ℕ, e n = 1 := by sorry

end NUMINAMATH_CALUDE_max_e_is_one_l3457_345745


namespace NUMINAMATH_CALUDE_system_implies_sum_l3457_345788

theorem system_implies_sum (x y m : ℝ) : x + m = 4 → y - 5 = m → x + y = 9 := by sorry

end NUMINAMATH_CALUDE_system_implies_sum_l3457_345788


namespace NUMINAMATH_CALUDE_kureishi_ratio_l3457_345793

/-- Represents the number of workers in Palabras bookstore who have read certain books -/
structure BookReadership where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  both : ℕ
  neither : ℕ

/-- The conditions of the Palabras bookstore problem -/
def palabras_conditions (r : BookReadership) : Prop :=
  r.total = 150 ∧
  r.saramago = r.total / 2 ∧
  r.both = 12 ∧
  r.neither = (r.saramago - r.both) - 1 ∧
  r.total = r.saramago + r.kureishi - r.both + r.neither

/-- The theorem stating the ratio of Kureishi readers to total workers -/
theorem kureishi_ratio (r : BookReadership) 
  (h : palabras_conditions r) : r.kureishi * 6 = r.total := by
  sorry

end NUMINAMATH_CALUDE_kureishi_ratio_l3457_345793


namespace NUMINAMATH_CALUDE_courses_last_year_is_six_l3457_345715

/-- Represents the number of courses taken last year -/
def courses_last_year : ℕ := 6

/-- Represents the average grade for the entire two-year period -/
def two_year_average : ℚ := 81

/-- Represents the number of courses taken the year before last -/
def courses_year_before : ℕ := 5

/-- Represents the average grade for the year before last -/
def average_year_before : ℚ := 60

/-- Represents the average grade for last year -/
def average_last_year : ℚ := 100

theorem courses_last_year_is_six :
  (courses_year_before * average_year_before + courses_last_year * average_last_year) / 
  (courses_year_before + courses_last_year : ℚ) = two_year_average :=
sorry

end NUMINAMATH_CALUDE_courses_last_year_is_six_l3457_345715


namespace NUMINAMATH_CALUDE_smallest_x_sqrt_3x_eq_5x_l3457_345704

theorem smallest_x_sqrt_3x_eq_5x (x : ℝ) :
  x ≥ 0 ∧ Real.sqrt (3 * x) = 5 * x → x = 0 := by sorry

end NUMINAMATH_CALUDE_smallest_x_sqrt_3x_eq_5x_l3457_345704


namespace NUMINAMATH_CALUDE_total_candies_l3457_345722

theorem total_candies (linda_candies chloe_candies michael_candies : ℕ) 
  (h1 : linda_candies = 340)
  (h2 : chloe_candies = 280)
  (h3 : michael_candies = 450) :
  linda_candies + chloe_candies + michael_candies = 1070 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3457_345722


namespace NUMINAMATH_CALUDE_original_bikes_count_l3457_345776

/-- Represents the number of bikes added per week -/
def bikes_added_per_week : ℕ := 3

/-- Represents the number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Represents the number of bikes sold in a month -/
def bikes_sold : ℕ := 18

/-- Represents the number of bikes in stock after a month -/
def bikes_in_stock : ℕ := 45

/-- Theorem stating that the original number of bikes is 51 -/
theorem original_bikes_count : 
  ∃ (original : ℕ), 
    original + (bikes_added_per_week * weeks_in_month) - bikes_sold = bikes_in_stock ∧ 
    original = 51 := by
  sorry

end NUMINAMATH_CALUDE_original_bikes_count_l3457_345776


namespace NUMINAMATH_CALUDE_fraction_equality_l3457_345713

theorem fraction_equality (a b : ℝ) (h : ((1/a) + (1/b)) / ((1/a) - (1/b)) = 2020) : 
  (a + b) / (a - b) = 2020 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3457_345713


namespace NUMINAMATH_CALUDE_prob_8_or_9_is_half_l3457_345756

/-- The probability of hitting the 10 ring in one shot -/
def prob_10 : ℝ := 0.3

/-- The probability of hitting the 9 ring in one shot -/
def prob_9 : ℝ := 0.3

/-- The probability of hitting the 8 ring in one shot -/
def prob_8 : ℝ := 0.2

/-- The probability of hitting the 8 or 9 rings in one shot -/
def prob_8_or_9 : ℝ := prob_9 + prob_8

theorem prob_8_or_9_is_half : prob_8_or_9 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_prob_8_or_9_is_half_l3457_345756


namespace NUMINAMATH_CALUDE_milk_water_ratio_in_mixed_vessel_l3457_345700

/-- Given three vessels with volumes in ratio 3:5:7 and milk-to-water ratios,
    prove the final milk-to-water ratio when mixed -/
theorem milk_water_ratio_in_mixed_vessel
  (v1 v2 v3 : ℚ)  -- Volumes of the three vessels
  (m1 w1 m2 w2 m3 w3 : ℚ)  -- Milk and water ratios in each vessel
  (hv : v1 / v2 = 3 / 5 ∧ v2 / v3 = 5 / 7)  -- Volume ratio condition
  (hr1 : m1 / w1 = 1 / 2)  -- Milk-to-water ratio in first vessel
  (hr2 : m2 / w2 = 3 / 2)  -- Milk-to-water ratio in second vessel
  (hr3 : m3 / w3 = 2 / 3)  -- Milk-to-water ratio in third vessel
  (hm1 : m1 + w1 = 1)  -- Normalization of ratios
  (hm2 : m2 + w2 = 1)
  (hm3 : m3 + w3 = 1) :
  (v1 * m1 + v2 * m2 + v3 * m3) / (v1 * w1 + v2 * w2 + v3 * w3) = 34 / 41 :=
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_in_mixed_vessel_l3457_345700


namespace NUMINAMATH_CALUDE_undefined_value_of_fraction_l3457_345743

theorem undefined_value_of_fraction (a : ℝ) : a^3 - 8 = 0 ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_undefined_value_of_fraction_l3457_345743


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3457_345765

/-- The polynomial function f(x) = x^8 + 6x^7 + 12x^6 + 2027x^5 - 1586x^4 -/
def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 12*x^6 + 2027*x^5 - 1586*x^4

/-- Theorem: The equation f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3457_345765


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l3457_345734

theorem angle_measure_in_triangle (A B C : Real) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A + B = 80 →       -- Given condition
  C = 100            -- Conclusion to prove
  := by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l3457_345734


namespace NUMINAMATH_CALUDE_S_independent_of_position_l3457_345749

/-- The sum of distances from vertices of an n-gon to the nearest vertices of an (n-1)-gon -/
def S (n : ℕ) (θ : ℝ) : ℝ := sorry

/-- The theorem stating that S depends only on n and not on the relative position θ -/
theorem S_independent_of_position (n : ℕ) (h1 : n ≥ 4) (h2 : Even n) (θ₁ θ₂ : ℝ) :
  S n θ₁ = S n θ₂ := by sorry

end NUMINAMATH_CALUDE_S_independent_of_position_l3457_345749


namespace NUMINAMATH_CALUDE_square_divisibility_l3457_345742

theorem square_divisibility (m n : ℕ) (h1 : m > n) (h2 : m % 2 = n % 2) 
  (h3 : (m^2 - n^2 + 1) ∣ (n^2 - 1)) : 
  ∃ k : ℕ, m^2 - n^2 + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l3457_345742


namespace NUMINAMATH_CALUDE_cost_of_pencils_and_pens_l3457_345727

/-- Given the cost of pencils and pens, prove the cost of 4 pencils and 4 pens -/
theorem cost_of_pencils_and_pens 
  (pencil_cost pen_cost : ℝ)
  (h1 : 6 * pencil_cost + 3 * pen_cost = 5.40)
  (h2 : 3 * pencil_cost + 5 * pen_cost = 4.80) :
  4 * pencil_cost + 4 * pen_cost = 4.80 := by
sorry


end NUMINAMATH_CALUDE_cost_of_pencils_and_pens_l3457_345727


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3457_345792

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    if a_1, a_3, and a_9 form a geometric sequence,
    then (a_1 + a_3 + a_9) / (a_2 + a_4 + a_10) = 13/16 -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) 
  (d : ℚ)
  (h1 : d ≠ 0)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1) * (a 3 - a 1)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3457_345792


namespace NUMINAMATH_CALUDE_z_squared_in_second_quadrant_l3457_345703

theorem z_squared_in_second_quadrant (z : ℂ) :
  z = Complex.exp (75 * Real.pi / 180 * Complex.I) →
  Complex.arg (z^2) > Real.pi / 2 ∧ Complex.arg (z^2) < Real.pi :=
sorry

end NUMINAMATH_CALUDE_z_squared_in_second_quadrant_l3457_345703
