import Mathlib

namespace smallest_three_digit_twice_in_pascal_l2987_298743

/-- Represents a position in Pascal's triangle by row and column -/
structure PascalPosition where
  row : Nat
  col : Nat
  h : col ≤ row

/-- Returns the value at a given position in Pascal's triangle -/
def pascal_value (pos : PascalPosition) : Nat :=
  sorry

/-- Predicate to check if a number appears at least twice in Pascal's triangle -/
def appears_twice (n : Nat) : Prop :=
  ∃ (pos1 pos2 : PascalPosition), pos1 ≠ pos2 ∧ pascal_value pos1 = n ∧ pascal_value pos2 = n

/-- The smallest three-digit number is 100 -/
def smallest_three_digit : Nat := 100

theorem smallest_three_digit_twice_in_pascal :
  (appears_twice smallest_three_digit) ∧
  (∀ n : Nat, n < smallest_three_digit → ¬(appears_twice n ∧ n ≥ 100)) :=
sorry

end smallest_three_digit_twice_in_pascal_l2987_298743


namespace booster_club_tickets_l2987_298719

/-- Represents the ticket information for the Booster Club trip --/
structure TicketInfo where
  num_nine_dollar : Nat
  total_cost : Nat
  cost_seven : Nat
  cost_nine : Nat

/-- Calculates the total number of tickets bought given the ticket information --/
def total_tickets (info : TicketInfo) : Nat :=
  info.num_nine_dollar + (info.total_cost - info.num_nine_dollar * info.cost_nine) / info.cost_seven

/-- Theorem stating that given the specific ticket information, the total number of tickets is 29 --/
theorem booster_club_tickets :
  let info : TicketInfo := {
    num_nine_dollar := 11,
    total_cost := 225,
    cost_seven := 7,
    cost_nine := 9
  }
  total_tickets info = 29 := by sorry

end booster_club_tickets_l2987_298719


namespace bottles_recycled_l2987_298705

def bottle_deposit : ℚ := 10 / 100
def can_deposit : ℚ := 5 / 100
def cans_recycled : ℕ := 140
def total_earned : ℚ := 15

theorem bottles_recycled : 
  ∃ (bottles : ℕ), (bottles : ℚ) * bottle_deposit + (cans_recycled : ℚ) * can_deposit = total_earned ∧ bottles = 80 := by
  sorry

end bottles_recycled_l2987_298705


namespace inequality_proof_l2987_298730

theorem inequality_proof (a : ℝ) : 3 * (1 + a^2 + a^4) - (1 + a + a^2)^2 ≥ 0 := by
  sorry

end inequality_proof_l2987_298730


namespace zinc_copper_ratio_in_mixture_l2987_298715

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the composition of a metal mixture -/
structure MetalMixture where
  totalWeight : ℝ
  zincWeight : ℝ

/-- Calculates the ratio of zinc to copper in a metal mixture -/
def zincCopperRatio (mixture : MetalMixture) : Ratio :=
  sorry

theorem zinc_copper_ratio_in_mixture :
  let mixture : MetalMixture := { totalWeight := 70, zincWeight := 31.5 }
  (zincCopperRatio mixture).numerator = 9 ∧
  (zincCopperRatio mixture).denominator = 11 :=
by sorry

end zinc_copper_ratio_in_mixture_l2987_298715


namespace two_sin_plus_three_cos_l2987_298713

theorem two_sin_plus_three_cos (x : ℝ) : 
  2 * Real.cos x - 3 * Real.sin x = 4 → 
  (2 * Real.sin x + 3 * Real.cos x = 3) ∨ (2 * Real.sin x + 3 * Real.cos x = 1) := by
sorry

end two_sin_plus_three_cos_l2987_298713


namespace direct_proportion_l2987_298785

theorem direct_proportion (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, y = k * x) ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by sorry

end direct_proportion_l2987_298785


namespace only_B_and_C_participate_l2987_298761

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a type for the activity participation
def Activity := Person → Prop

-- Define the conditions
def condition1 (act : Activity) : Prop := act Person.A → act Person.B
def condition2 (act : Activity) : Prop := ¬act Person.C → ¬act Person.B
def condition3 (act : Activity) : Prop := act Person.C → ¬act Person.D

-- Define the property of exactly two people participating
def exactlyTwo (act : Activity) : Prop :=
  ∃ (p1 p2 : Person), p1 ≠ p2 ∧ act p1 ∧ act p2 ∧ ∀ (p : Person), act p → (p = p1 ∨ p = p2)

-- The main theorem
theorem only_B_and_C_participate :
  ∀ (act : Activity),
    condition1 act →
    condition2 act →
    condition3 act →
    exactlyTwo act →
    act Person.B ∧ act Person.C ∧ ¬act Person.A ∧ ¬act Person.D :=
by sorry

end only_B_and_C_participate_l2987_298761


namespace max_value_of_function_l2987_298767

/-- The function f(x) = -x - 9/x + 18 for x > 0 has a maximum value of 12 -/
theorem max_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ (M : ℝ), M = 12 ∧ ∀ y, y > 0 → -y - 9/y + 18 ≤ M :=
by sorry

end max_value_of_function_l2987_298767


namespace speed_with_400_people_l2987_298768

/-- Represents the speed of a spaceship given the number of people on board. -/
def spaceshipSpeed (people : ℕ) : ℝ :=
  sorry

/-- The speed halves for every 100 additional people. -/
axiom speed_halves (n : ℕ) : spaceshipSpeed (n + 100) = (spaceshipSpeed n) / 2

/-- The speed of the spaceship with 200 people on board is 500 km/hr. -/
axiom initial_speed : spaceshipSpeed 200 = 500

/-- The speed of the spaceship with 400 people on board is 125 km/hr. -/
theorem speed_with_400_people : spaceshipSpeed 400 = 125 := by
  sorry

end speed_with_400_people_l2987_298768


namespace count_sequences_eq_fib_21_l2987_298749

/-- The number of increasing sequences satisfying the given conditions -/
def count_sequences : ℕ := sorry

/-- The 21st Fibonacci number -/
def fib_21 : ℕ := sorry

/-- Predicate for valid sequences -/
def valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i, 1 ≤ a i ∧ a i ≤ 20) ∧
  (∀ i j, i < j → a i < a j) ∧
  (∀ i, a i % 2 = i % 2)

theorem count_sequences_eq_fib_21 : count_sequences = fib_21 := by
  sorry

end count_sequences_eq_fib_21_l2987_298749


namespace special_parallelogram_existence_l2987_298726

/-- The existence of a special parallelogram for any point on an ellipse -/
theorem special_parallelogram_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 →
    ∃ (p q r s : ℝ × ℝ),
      -- P is on the ellipse
      (x, y) = p ∧
      -- PQRS forms a parallelogram
      (p.1 - q.1 = r.1 - s.1 ∧ p.2 - q.2 = r.2 - s.2) ∧
      (p.1 - s.1 = q.1 - r.1 ∧ p.2 - s.2 = q.2 - r.2) ∧
      -- Parallelogram is tangent to the ellipse
      (∃ (t : ℝ × ℝ), t.1^2/a^2 + t.2^2/b^2 = 1 ∧
        ((t.1 - p.1) * (q.1 - p.1) + (t.2 - p.2) * (q.2 - p.2) = 0 ∨
         (t.1 - q.1) * (r.1 - q.1) + (t.2 - q.2) * (r.2 - q.2) = 0 ∨
         (t.1 - r.1) * (s.1 - r.1) + (t.2 - r.2) * (s.2 - r.2) = 0 ∨
         (t.1 - s.1) * (p.1 - s.1) + (t.2 - s.2) * (p.2 - s.2) = 0)) ∧
      -- Parallelogram is externally tangent to the unit circle
      (∃ (u : ℝ × ℝ), u.1^2 + u.2^2 = 1 ∧
        ((u.1 - p.1) * (q.1 - p.1) + (u.2 - p.2) * (q.2 - p.2) = 0 ∨
         (u.1 - q.1) * (r.1 - q.1) + (u.2 - q.2) * (r.2 - q.2) = 0 ∨
         (u.1 - r.1) * (s.1 - r.1) + (u.2 - r.2) * (s.2 - r.2) = 0 ∨
         (u.1 - s.1) * (p.1 - s.1) + (u.2 - s.2) * (p.2 - s.2) = 0))) ↔
  1/a^2 + 1/b^2 = 1 :=
by sorry

end special_parallelogram_existence_l2987_298726


namespace smallest_coprime_to_210_l2987_298789

theorem smallest_coprime_to_210 : 
  ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ (y : ℕ), y > 1 ∧ y < x → Nat.gcd y 210 ≠ 1 :=
by sorry

end smallest_coprime_to_210_l2987_298789


namespace chef_pies_total_l2987_298748

theorem chef_pies_total (apple : ℕ) (pecan : ℕ) (pumpkin : ℕ) 
  (h1 : apple = 2) (h2 : pecan = 4) (h3 : pumpkin = 7) : 
  apple + pecan + pumpkin = 13 := by
  sorry

end chef_pies_total_l2987_298748


namespace equation_solutions_l2987_298721

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2) ∧
  (∀ x : ℝ, 4 * x^2 - 2 * Real.sqrt 3 * x - 1 = 0 ↔ 
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4) := by
  sorry

end equation_solutions_l2987_298721


namespace locus_of_center_P_l2987_298795

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Define that B is inside circle A
def B_inside_A : Prop := circle_A (point_B.1) (point_B.2)

-- Define circle P
def circle_P (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  -- P passes through B
  (center.1 - point_B.1)^2 + (center.2 - point_B.2)^2 = radius^2 ∧
  -- P is tangent to A internally
  ((center.1 + 3)^2 + center.2^2)^(1/2) + radius = 10

-- Theorem statement
theorem locus_of_center_P :
  ∀ (x y : ℝ), (∃ (r : ℝ), circle_P (x, y) r) ↔ x^2/25 + y^2/16 = 1 :=
sorry

end locus_of_center_P_l2987_298795


namespace certain_number_problem_l2987_298736

theorem certain_number_problem (N : ℚ) : 
  (5 / 6 : ℚ) * N = (5 / 16 : ℚ) * N + 200 → N = 384 := by
sorry

end certain_number_problem_l2987_298736


namespace regular_decagon_angles_l2987_298753

/-- Properties of a regular decagon -/
theorem regular_decagon_angles :
  let n : ℕ := 10  -- number of sides in a decagon
  let exterior_angle : ℝ := 360 / n
  let interior_angle : ℝ := (n - 2) * 180 / n
  exterior_angle = 36 ∧ interior_angle = 144 := by
  sorry

end regular_decagon_angles_l2987_298753


namespace intersection_complement_equality_l2987_298702

open Set

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,3}
def B : Set Nat := {3,5}

theorem intersection_complement_equality : A ∩ (U \ B) = {2} := by
  sorry

end intersection_complement_equality_l2987_298702


namespace ms_hatcher_students_l2987_298790

def total_students (third_graders : ℕ) : ℕ :=
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  let sixth_graders := (third_graders + fourth_graders) * 3 / 4
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem ms_hatcher_students :
  total_students 20 = 115 := by
  sorry

end ms_hatcher_students_l2987_298790


namespace total_earnings_is_228_l2987_298712

/-- Calculates Zainab's total earnings for 4 weeks of passing out flyers -/
def total_earnings : ℝ :=
  let monday_hours : ℝ := 3
  let monday_rate : ℝ := 2.5
  let wednesday_hours : ℝ := 4
  let wednesday_rate : ℝ := 3
  let saturday_hours : ℝ := 5
  let saturday_rate : ℝ := 3.5
  let saturday_flyers : ℝ := 200
  let flyer_commission : ℝ := 0.1
  let weeks : ℝ := 4

  let monday_earnings := monday_hours * monday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let saturday_hourly_earnings := saturday_hours * saturday_rate
  let saturday_commission := saturday_flyers * flyer_commission
  let saturday_total_earnings := saturday_hourly_earnings + saturday_commission
  let weekly_earnings := monday_earnings + wednesday_earnings + saturday_total_earnings

  weeks * weekly_earnings

theorem total_earnings_is_228 : total_earnings = 228 := by
  sorry

end total_earnings_is_228_l2987_298712


namespace unique_sequence_l2987_298747

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, a (n * m) = a n * a m) ∧
  (∀ k : ℕ, ∃ n > k, Finset.range n = Finset.image a (Finset.range n))

theorem unique_sequence (a : ℕ → ℕ) (h : is_valid_sequence a) : ∀ n : ℕ, a n = n := by
  sorry

end unique_sequence_l2987_298747


namespace modular_inverse_13_mod_1200_l2987_298764

theorem modular_inverse_13_mod_1200 : ∃ x : ℕ, x < 1200 ∧ (13 * x) % 1200 = 1 := by
  sorry

end modular_inverse_13_mod_1200_l2987_298764


namespace exist_consecutive_lucky_tickets_l2987_298722

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: There exist two consecutive natural numbers whose sums of digits are both divisible by 7 -/
theorem exist_consecutive_lucky_tickets : ∃ n : ℕ, 7 ∣ sum_of_digits n ∧ 7 ∣ sum_of_digits (n + 1) :=
sorry

end exist_consecutive_lucky_tickets_l2987_298722


namespace faster_train_length_l2987_298740

/-- Proves that the length of a faster train is 340 meters given the specified conditions -/
theorem faster_train_length (faster_speed slower_speed : ℝ) (crossing_time : ℝ) : 
  faster_speed = 108 →
  slower_speed = 36 →
  crossing_time = 17 →
  (faster_speed - slower_speed) * crossing_time * (5/18) = 340 :=
by sorry

end faster_train_length_l2987_298740


namespace tangent_slope_determines_a_l2987_298755

/-- Given a function f(x) = (x^2 + a) / (x + 1), prove that if the slope of the tangent line
    at x = 1 is 1, then a = -1 -/
theorem tangent_slope_determines_a (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (x^2 + a) / (x + 1)
  (deriv f 1 = 1) → a = -1 := by
  sorry

end tangent_slope_determines_a_l2987_298755


namespace change_calculation_l2987_298784

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def num_sandals : ℕ := 3
def payment : ℕ := 100

def total_cost : ℕ := shirt_price * num_shirts + sandal_price * num_sandals

theorem change_calculation : payment - total_cost = 41 := by
  sorry

end change_calculation_l2987_298784


namespace sandwich_count_l2987_298745

theorem sandwich_count (billy_sandwiches : ℕ) (katelyn_extra : ℕ) :
  billy_sandwiches = 49 →
  katelyn_extra = 47 →
  (billy_sandwiches + katelyn_extra + billy_sandwiches + (billy_sandwiches + katelyn_extra) / 4 = 169) :=
by
  sorry

end sandwich_count_l2987_298745


namespace gina_charity_fraction_l2987_298731

def initial_amount : ℚ := 400
def mom_fraction : ℚ := 1/4
def clothes_fraction : ℚ := 1/8
def kept_amount : ℚ := 170
def charity_fraction : ℚ := 1/5

theorem gina_charity_fraction :
  charity_fraction = (initial_amount - mom_fraction * initial_amount - clothes_fraction * initial_amount - kept_amount) / initial_amount := by
  sorry

end gina_charity_fraction_l2987_298731


namespace line_relationships_l2987_298733

/-- Definition of parallel lines based on slopes -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- Definition of perpendicular lines based on slopes -/
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The main theorem -/
theorem line_relationships :
  let slopes : List ℚ := [2, -3, 3, 4, -3/2]
  ∃! (pair : (ℚ × ℚ)), pair ∈ (slopes.product slopes) ∧
    (parallel pair.1 pair.2 ∨ perpendicular pair.1 pair.2) ∧
    pair.1 ≠ pair.2 :=
by sorry

end line_relationships_l2987_298733


namespace parabola_translation_l2987_298763

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically -/
def translateVertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- Translates a parabola horizontally -/
def translateHorizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := 2 * p.a * h + p.b, c := p.a * h^2 + p.b * h + p.c }

/-- The main theorem stating that translating y = 3x² upwards by 3 and left by 2 results in y = 3(x+2)² + 3 -/
theorem parabola_translation (original : Parabola) 
  (h : original = { a := 3, b := 0, c := 0 }) : 
  translateHorizontal (translateVertical original 3) 2 = { a := 3, b := 12, c := 15 } := by
  sorry

end parabola_translation_l2987_298763


namespace total_distance_traveled_l2987_298773

theorem total_distance_traveled (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 2) (h2 : v2 = 6) (h3 : v3 = 6) (h4 : t = 11 / 60) :
  let d := t * (v1⁻¹ + v2⁻¹ + v3⁻¹)⁻¹
  3 * d = 33 / 50 := by
  sorry

end total_distance_traveled_l2987_298773


namespace curve_perimeter_ge_twice_diagonal_curve_perimeter_eq_twice_diagonal_l2987_298704

/-- A closed curve in 2D space -/
structure ClosedCurve where
  -- Add necessary fields/axioms for a closed curve

/-- A rectangle in 2D space -/
structure Rectangle where
  -- Add necessary fields for a rectangle (e.g., width, height, position)

/-- The perimeter of a closed curve -/
noncomputable def perimeter (c : ClosedCurve) : ℝ :=
  sorry

/-- The diagonal length of a rectangle -/
def diagonal (r : Rectangle) : ℝ :=
  sorry

/-- Predicate to check if a curve intersects all sides of a rectangle -/
def intersectsAllSides (c : ClosedCurve) (r : Rectangle) : Prop :=
  sorry

theorem curve_perimeter_ge_twice_diagonal 
  (c : ClosedCurve) (r : Rectangle) 
  (h : intersectsAllSides c r) : 
  perimeter c ≥ 2 * diagonal r :=
sorry

/-- Condition for equality -/
def equalityCondition (c : ClosedCurve) (r : Rectangle) : Prop :=
  sorry

theorem curve_perimeter_eq_twice_diagonal 
  (c : ClosedCurve) (r : Rectangle) 
  (h1 : intersectsAllSides c r)
  (h2 : equalityCondition c r) : 
  perimeter c = 2 * diagonal r :=
sorry

end curve_perimeter_ge_twice_diagonal_curve_perimeter_eq_twice_diagonal_l2987_298704


namespace triangle_perimeter_l2987_298708

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 7 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →
  a + b + c = 5 + Real.sqrt 7 := by
  sorry

end triangle_perimeter_l2987_298708


namespace gcd_228_1995_l2987_298791

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l2987_298791


namespace reaction_compound_is_chloramine_l2987_298798

/-- Represents a chemical compound --/
structure Compound where
  formula : String

/-- Represents a chemical reaction --/
structure Reaction where
  reactant : Compound
  water_amount : ℝ
  hcl_product : ℝ
  nh4oh_product : ℝ

/-- The molecular weight of water in g/mol --/
def water_molecular_weight : ℝ := 18

/-- Checks if a compound is chloramine --/
def is_chloramine (c : Compound) : Prop :=
  c.formula = "NH2Cl"

/-- Theorem stating that the compound in the reaction is chloramine --/
theorem reaction_compound_is_chloramine (r : Reaction) : 
  r.water_amount = water_molecular_weight ∧ 
  r.hcl_product = 1 ∧ 
  r.nh4oh_product = 1 → 
  is_chloramine r.reactant :=
by
  sorry


end reaction_compound_is_chloramine_l2987_298798


namespace triangle_angle_inequalities_l2987_298796

theorem triangle_angle_inequalities (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  (Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) ≤ 1/8) ∧ 
  (Real.cos α * Real.cos β * Real.cos γ ≤ 1/8) := by
  sorry

end triangle_angle_inequalities_l2987_298796


namespace binary_10111_equals_43_base_5_l2987_298771

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-5 representation -/
def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The binary representation of 10111 -/
def binary_10111 : List Bool := [true, true, true, false, true]

theorem binary_10111_equals_43_base_5 :
  to_base_5 (binary_to_decimal binary_10111) = [4, 3] :=
sorry

end binary_10111_equals_43_base_5_l2987_298771


namespace complement_of_A_in_U_l2987_298775

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem stating that the complement of A in U is {2, 4}
theorem complement_of_A_in_U :
  (U \ A) = {2, 4} := by
  sorry

end complement_of_A_in_U_l2987_298775


namespace company_workshops_l2987_298751

/-- Given a total number of employees and a maximum workshop capacity,
    calculate the minimum number of workshops required. -/
def min_workshops (total_employees : ℕ) (max_capacity : ℕ) : ℕ :=
  (total_employees + max_capacity - 1) / max_capacity

/-- Theorem stating the minimum number of workshops required for the given problem -/
theorem company_workshops :
  let total_employees := 56
  let max_capacity := 15
  min_workshops total_employees max_capacity = 4 := by
  sorry

end company_workshops_l2987_298751


namespace two_preserving_transformations_l2987_298728

/-- Represents the regular, infinite pattern of squares and line segments along a line ℓ -/
structure RegularPattern :=
  (ℓ : Line)
  (square_size : ℝ)
  (diagonal_length : ℝ)

/-- Enumeration of the four types of rigid motion transformations -/
inductive RigidMotion
  | Rotation
  | Translation
  | ReflectionAcross
  | ReflectionPerpendicular

/-- Predicate to check if a rigid motion maps the pattern onto itself -/
def preserves_pattern (r : RegularPattern) (m : RigidMotion) : Prop :=
  sorry

/-- The main theorem stating that exactly two rigid motions preserve the pattern -/
theorem two_preserving_transformations (r : RegularPattern) :
  ∃! (s : Finset RigidMotion), s.card = 2 ∧ ∀ m ∈ s, preserves_pattern r m :=
sorry

end two_preserving_transformations_l2987_298728


namespace survivor_same_tribe_probability_l2987_298700

/-- The probability that both quitters are from the same tribe in a Survivor-like game. -/
theorem survivor_same_tribe_probability :
  let total_contestants : ℕ := 18
  let tribe_size : ℕ := 9
  let immune_contestants : ℕ := 1
  let quitters : ℕ := 2
  let contestants_at_risk : ℕ := total_contestants - immune_contestants
  let same_tribe_quitters : ℕ := 2 * (tribe_size.choose quitters)
  let total_quitter_combinations : ℕ := contestants_at_risk.choose quitters
  (same_tribe_quitters : ℚ) / total_quitter_combinations = 9 / 17 :=
by sorry

end survivor_same_tribe_probability_l2987_298700


namespace coin_denominations_exist_l2987_298741

theorem coin_denominations_exist : ∃ (S : Finset ℕ), 
  (Finset.card S = 12) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 → 
    ∃ (T : Finset ℕ), 
      (∀ m ∈ T, m ∈ S) ∧ 
      (Finset.card T ≤ 8) ∧ 
      (Finset.sum T id = n)) :=
sorry

end coin_denominations_exist_l2987_298741


namespace intersection_with_complement_l2987_298781

def U : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}

theorem intersection_with_complement : P ∩ (U \ Q) = {1} := by
  sorry

end intersection_with_complement_l2987_298781


namespace synesthesia_demonstrates_mutual_influence_and_restriction_l2987_298711

/-- Represents a sensory perception -/
inductive Sense
  | Sight
  | Hearing
  | Taste
  | Smell
  | Touch

/-- Represents the phenomenon of synesthesia -/
def Synesthesia := Set (Sense × Sense)

/-- Represents the property of mutual influence and restriction -/
def MutualInfluenceAndRestriction (s : Synesthesia) : Prop := sorry

/-- Represents a thing and its internal elements -/
structure Thing where
  elements : Set Sense

theorem synesthesia_demonstrates_mutual_influence_and_restriction 
  (s : Synesthesia) 
  (h : s.Nonempty) : 
  MutualInfluenceAndRestriction s := by
  sorry

#check synesthesia_demonstrates_mutual_influence_and_restriction

end synesthesia_demonstrates_mutual_influence_and_restriction_l2987_298711


namespace matrix_crossout_theorem_l2987_298758

theorem matrix_crossout_theorem (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ (rows : Finset (Fin 1000)), rows.card = 10 ∧
    ∀ j, ∃ i ∈ rows, M i j = true) ∨
  (∃ (cols : Finset (Fin 1000)), cols.card = 10 ∧
    ∀ i, ∃ j ∈ cols, M i j = false) :=
sorry

end matrix_crossout_theorem_l2987_298758


namespace prob_same_color_l2987_298725

def box_prob (white : ℕ) (black : ℕ) : ℚ :=
  let total := white + black
  let same_color := (white.choose 3) + (black.choose 3)
  let total_combinations := total.choose 3
  same_color / total_combinations

theorem prob_same_color : box_prob 7 9 = 119 / 560 := by
  sorry

end prob_same_color_l2987_298725


namespace cos_sixty_degrees_l2987_298762

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l2987_298762


namespace cafeteria_apples_l2987_298777

/-- The number of apples in the school cafeteria after using some for lunch and buying more. -/
def final_apples (initial : ℕ) (used : ℕ) (bought : ℕ) : ℕ :=
  initial - used + bought

/-- Theorem stating that given the specific numbers in the problem, the final number of apples is 9. -/
theorem cafeteria_apples : final_apples 23 20 6 = 9 := by
  sorry

end cafeteria_apples_l2987_298777


namespace lcm_48_147_l2987_298776

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := by
  sorry

end lcm_48_147_l2987_298776


namespace sum_of_reciprocals_l2987_298765

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) :
  1 / x + 1 / y = -3 := by
  sorry

end sum_of_reciprocals_l2987_298765


namespace smallest_number_with_condition_condition_satisfied_by_725_l2987_298746

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n - 1)).filter (fun d => d ≠ 1 ∧ n % d = 0)

def divisors_condition (n : ℕ) : Prop :=
  let divs := proper_divisors n
  let largest_sum := (Finset.max' divs (by sorry) + Finset.max' (divs.erase (Finset.max' divs (by sorry))) (by sorry))
  let smallest_sum := (Finset.min' divs (by sorry) + Finset.min' (divs.erase (Finset.min' divs (by sorry))) (by sorry))
  ¬(largest_sum % smallest_sum = 0)

theorem smallest_number_with_condition :
  ∀ n : ℕ, n < 725 → ¬(ends_with_five n ∧ divisors_condition n) :=
by sorry

theorem condition_satisfied_by_725 :
  ends_with_five 725 ∧ divisors_condition 725 :=
by sorry

end smallest_number_with_condition_condition_satisfied_by_725_l2987_298746


namespace simplification_to_5x_squared_l2987_298794

theorem simplification_to_5x_squared (k : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, (x - k*x) * (2*x - k*x) - 3*x * (2*x - k*x) = 5*x^2) ∧ 
  (∀ k : ℝ, (∀ x : ℝ, (x - k*x) * (2*x - k*x) - 3*x * (2*x - k*x) = 5*x^2) → (k = 3 ∨ k = -3)) :=
by sorry

end simplification_to_5x_squared_l2987_298794


namespace events_mutually_exclusive_and_complementary_l2987_298778

-- Define the sample space
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define event A
def A : Set ℕ := {n ∈ S | n % 2 = 0}

-- Define event B
def B : Set ℕ := {n ∈ S | n % 2 ≠ 0}

-- Theorem statement
theorem events_mutually_exclusive_and_complementary : 
  (A ∩ B = ∅) ∧ (A ∪ B = S) := by
  sorry

end events_mutually_exclusive_and_complementary_l2987_298778


namespace n_fourth_plus_four_prime_iff_n_eq_one_l2987_298738

theorem n_fourth_plus_four_prime_iff_n_eq_one (n : ℕ+) :
  Nat.Prime (n^4 + 4) ↔ n = 1 := by
  sorry

end n_fourth_plus_four_prime_iff_n_eq_one_l2987_298738


namespace animal_sightings_l2987_298703

theorem animal_sightings (january : ℕ) (february : ℕ) (march : ℕ) 
  (h1 : february = 3 * january)
  (h2 : march = february / 2)
  (h3 : january + february + march = 143) :
  january = 26 := by
sorry

end animal_sightings_l2987_298703


namespace data_average_l2987_298760

theorem data_average (a : ℝ) : 
  (1 + 3 + 2 + 5 + a) / 5 = 3 → a = 4 := by
  sorry

end data_average_l2987_298760


namespace magnitude_of_sum_equals_five_l2987_298742

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![2, 2]

theorem magnitude_of_sum_equals_five :
  ‖vector_a + vector_b‖ = 5 := by
  sorry

end magnitude_of_sum_equals_five_l2987_298742


namespace pumpkin_contest_result_l2987_298724

/-- The weight of Brad's pumpkin in pounds -/
def brads_pumpkin : ℕ := 54

/-- The weight of Jessica's pumpkin in pounds -/
def jessicas_pumpkin : ℕ := brads_pumpkin / 2

/-- The weight of Betty's pumpkin in pounds -/
def bettys_pumpkin : ℕ := jessicas_pumpkin * 4

/-- The difference between the heaviest and lightest pumpkin in pounds -/
def pumpkin_weight_difference : ℕ := max brads_pumpkin (max jessicas_pumpkin bettys_pumpkin) - 
                                     min brads_pumpkin (min jessicas_pumpkin bettys_pumpkin)

theorem pumpkin_contest_result : pumpkin_weight_difference = 81 := by
  sorry

end pumpkin_contest_result_l2987_298724


namespace largest_even_digit_multiple_of_5_proof_l2987_298754

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_5 : ℕ := 8800

theorem largest_even_digit_multiple_of_5_proof :
  (has_only_even_digits largest_even_digit_multiple_of_5) ∧
  (largest_even_digit_multiple_of_5 < 10000) ∧
  (largest_even_digit_multiple_of_5 % 5 = 0) ∧
  (∀ n : ℕ, n > largest_even_digit_multiple_of_5 →
    ¬(has_only_even_digits n ∧ n < 10000 ∧ n % 5 = 0)) :=
by sorry

#check largest_even_digit_multiple_of_5_proof

end largest_even_digit_multiple_of_5_proof_l2987_298754


namespace division_remainder_proof_l2987_298732

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 1375 →
  divisor = 66 →
  quotient = 20 →
  dividend = divisor * quotient + remainder →
  remainder = 55 := by
sorry

end division_remainder_proof_l2987_298732


namespace probabilities_in_mathematics_l2987_298707

def word : String := "mathematics"

def is_vowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u']

def count_char (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

theorem probabilities_in_mathematics :
  (count_char word 't' : ℚ) / word.length = 2 / 11 ∧
  (count_vowels word : ℚ) / word.length = 4 / 11 := by
  sorry

end probabilities_in_mathematics_l2987_298707


namespace internal_diagonal_cubes_l2987_298750

theorem internal_diagonal_cubes (a b c : ℕ) (ha : a = 200) (hb : b = 300) (hc : c = 350) :
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c) = 700 := by
  sorry

end internal_diagonal_cubes_l2987_298750


namespace unique_solution_l2987_298752

/-- Function that calculates the product of digits of a positive integer -/
def product_of_digits (x : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 12 is the only positive integer solution -/
theorem unique_solution :
  ∃! (x : ℕ+), product_of_digits x = x^2 - 10*x - 22 :=
sorry

end unique_solution_l2987_298752


namespace no_snow_no_fog_probability_l2987_298779

theorem no_snow_no_fog_probability
  (p_snow : ℝ)
  (p_fog_given_no_snow : ℝ)
  (h_p_snow : p_snow = 1/4)
  (h_p_fog_given_no_snow : p_fog_given_no_snow = 1/3) :
  (1 - p_snow) * (1 - p_fog_given_no_snow) = 1/2 := by
  sorry

end no_snow_no_fog_probability_l2987_298779


namespace rhombus_prism_volume_l2987_298734

/-- A right prism with a rhombus base -/
structure RhombusPrism where
  /-- The acute angle of the rhombus base -/
  α : ℝ
  /-- The length of the larger diagonal of the rhombus base -/
  l : ℝ
  /-- The angle between the larger diagonal and the base plane -/
  β : ℝ
  /-- The acute angle condition -/
  h_α_acute : 0 < α ∧ α < π / 2
  /-- The positive length condition -/
  h_l_pos : l > 0
  /-- The angle β condition -/
  h_β_acute : 0 < β ∧ β < π / 2

/-- The volume of a rhombus-based right prism -/
noncomputable def volume (p : RhombusPrism) : ℝ :=
  1/2 * p.l^3 * Real.sin p.β * Real.cos p.β^2 * Real.tan (p.α/2)

theorem rhombus_prism_volume (p : RhombusPrism) :
  volume p = 1/2 * p.l^3 * Real.sin p.β * Real.cos p.β^2 * Real.tan (p.α/2) := by
  sorry

end rhombus_prism_volume_l2987_298734


namespace sphere_cylinder_ratio_l2987_298727

theorem sphere_cylinder_ratio (R : ℝ) (h : R > 0) : 
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_volume := 2 * Real.pi * R^3
  let empty_space := cylinder_volume - sphere_volume
  let total_empty_space := 5 * empty_space
  let total_occupied_space := 5 * sphere_volume
  (total_empty_space / total_occupied_space) = 1 / 2 := by
sorry

end sphere_cylinder_ratio_l2987_298727


namespace place_three_after_correct_l2987_298788

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.tens + n.units

/-- The result of placing 3 after a two-digit number -/
def place_three_after (n : TwoDigitNumber) : ℕ :=
  100 * n.tens + 10 * n.units + 3

theorem place_three_after_correct (n : TwoDigitNumber) :
  place_three_after n = 100 * n.tens + 10 * n.units + 3 := by
  sorry

end place_three_after_correct_l2987_298788


namespace sinusoidal_period_l2987_298720

theorem sinusoidal_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.sin (b * x + c) + d) →
  (∃ n : ℕ, n = 4 ∧ (2 * π) / b = (2 * π) / n) →
  b = 4 :=
by sorry

end sinusoidal_period_l2987_298720


namespace angle_value_proof_l2987_298787

theorem angle_value_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β)) : α = π/4 := by
  sorry

end angle_value_proof_l2987_298787


namespace twelve_sticks_need_two_breaks_fifteen_sticks_no_breaks_l2987_298774

/-- Given n sticks of lengths 1, 2, ..., n, this function returns the minimum number
    of sticks that need to be broken in half to form a square. If it's possible to form
    a square without breaking any sticks, it returns 0. -/
def minSticksToBreak (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 12 sticks, we need to break 2 sticks to form a square -/
theorem twelve_sticks_need_two_breaks : minSticksToBreak 12 = 2 :=
  sorry

/-- Theorem stating that for 15 sticks, we can form a square without breaking any sticks -/
theorem fifteen_sticks_no_breaks : minSticksToBreak 15 = 0 :=
  sorry

end twelve_sticks_need_two_breaks_fifteen_sticks_no_breaks_l2987_298774


namespace halfway_point_l2987_298782

theorem halfway_point (a b : ℚ) (ha : a = 1/8) (hb : b = 1/10) :
  (a + b) / 2 = 9/80 := by
  sorry

end halfway_point_l2987_298782


namespace zero_product_property_l2987_298710

theorem zero_product_property (x : ℤ) : (∀ y : ℤ, x * y = 0) → x = 0 := by
  sorry

end zero_product_property_l2987_298710


namespace subtract_negatives_l2987_298701

theorem subtract_negatives : -2 - 1 = -3 := by
  sorry

end subtract_negatives_l2987_298701


namespace expansion_properties_l2987_298772

/-- Represents the coefficient of x^(k/3) in the expansion of (∛x - 3/∛x)^n -/
def coeff (n : ℕ) (k : ℤ) : ℚ :=
  sorry

/-- The sixth term in the expansion -/
def sixth_term (n : ℕ) : ℚ := coeff n (n - 10)

/-- The coefficient of x² in the expansion -/
def x_squared_coeff (n : ℕ) : ℚ := coeff n 6

theorem expansion_properties (n : ℕ) :
  sixth_term n = 0 →
  n = 10 ∧ x_squared_coeff 10 = 405 := by
  sorry

end expansion_properties_l2987_298772


namespace unique_prime_factorization_l2987_298792

theorem unique_prime_factorization : 
  ∃! (d e f : ℕ), 
    d.Prime ∧ e.Prime ∧ f.Prime ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    d * e * f = 7902 ∧
    d + e + f = 1322 :=
by sorry

end unique_prime_factorization_l2987_298792


namespace sin_cos_relation_l2987_298770

theorem sin_cos_relation (θ : Real) (h1 : Real.sin θ + Real.cos θ = 1/2) 
  (h2 : π/2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = -Real.sqrt 7 / 2 := by
  sorry

end sin_cos_relation_l2987_298770


namespace cos_pi_plus_alpha_l2987_298757

theorem cos_pi_plus_alpha (α : Real) (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π + α) = -3 / 5 := by
  sorry

end cos_pi_plus_alpha_l2987_298757


namespace product_of_roots_plus_one_l2987_298799

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (x^3 - 15*x^2 + 22*x - 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 46 := by
sorry

end product_of_roots_plus_one_l2987_298799


namespace excellent_credit_prob_expectation_X_l2987_298717

/-- Credit score distribution --/
def credit_distribution : Finset (ℕ × ℕ) := {(150, 25), (120, 60), (100, 65), (80, 35), (0, 15)}

/-- Total population --/
def total_population : ℕ := 200

/-- Voucher allocation function --/
def voucher (score : ℕ) : ℕ :=
  if score > 150 then 100
  else if score > 100 then 50
  else 0

/-- Probability of selecting 2 people with excellent credit --/
theorem excellent_credit_prob : 
  (Nat.choose 25 2 : ℚ) / (Nat.choose total_population 2) = 3 / 199 := by sorry

/-- Distribution of total vouchers X for 2 randomly selected people --/
def voucher_distribution : Finset (ℕ × ℚ) := {(0, 1/16), (50, 5/16), (100, 29/64), (150, 5/32), (200, 1/64)}

/-- Expectation of X --/
theorem expectation_X : 
  (voucher_distribution.sum (λ (x, p) => x * p)) = 175 / 2 := by sorry

end excellent_credit_prob_expectation_X_l2987_298717


namespace surface_area_ratio_of_cubes_l2987_298718

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a > 0) (k : b > 0) (ratio : a = 4 * b) :
  (6 * a^2) / (6 * b^2) = 16 := by sorry

end surface_area_ratio_of_cubes_l2987_298718


namespace unique_element_in_A_l2987_298756

/-- The set A defined by the quadratic equation ax^2 - x + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - x + 1 = 0}

/-- The theorem stating that if A contains only one element, then a = 0 or a = 1/4 -/
theorem unique_element_in_A (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1/4 := by
  sorry

end unique_element_in_A_l2987_298756


namespace greatest_k_for_100_power_dividing_50_factorial_l2987_298714

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def highest_power_of_2 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n / 2) + highest_power_of_2 (n / 2)

def highest_power_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n / 5) + highest_power_of_5 (n / 5)

theorem greatest_k_for_100_power_dividing_50_factorial :
  (∃ k : ℕ, k = 6 ∧
    ∀ m : ℕ, (100 ^ m : ℕ) ∣ factorial 50 → m ≤ k) :=
by sorry

end greatest_k_for_100_power_dividing_50_factorial_l2987_298714


namespace part_one_part_two_l2987_298780

-- Define the function f
def f (x m : ℝ) : ℝ := 3 * x^2 + m * (m - 6) * x + 5

-- Theorem for part 1
theorem part_one (m : ℝ) : f 1 m > 0 ↔ m > 4 ∨ m < 2 := by sorry

-- Theorem for part 2
theorem part_two (m n : ℝ) : 
  (∀ x, f x m < n ↔ -1 < x ∧ x < 4) → m = 3 ∧ n = 17 := by sorry

end part_one_part_two_l2987_298780


namespace min_value_ab_l2987_298706

theorem min_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) :
  9 ≤ a * b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ - a₀ * b₀ + 3 = 0 ∧ a₀ * b₀ = 9 :=
by sorry

end min_value_ab_l2987_298706


namespace total_spokes_in_garage_l2987_298793

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 4

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- Theorem: The total number of spokes in the garage is 80 -/
theorem total_spokes_in_garage : 
  num_bicycles * wheels_per_bicycle * spokes_per_wheel = 80 := by
  sorry

end total_spokes_in_garage_l2987_298793


namespace car_sales_third_day_l2987_298783

theorem car_sales_third_day 
  (total_sales : ℕ) 
  (first_day : ℕ) 
  (second_day : ℕ) 
  (h1 : total_sales = 57) 
  (h2 : first_day = 14) 
  (h3 : second_day = 16) : 
  total_sales - (first_day + second_day) = 27 := by
sorry

end car_sales_third_day_l2987_298783


namespace cos_equality_theorem_l2987_298729

theorem cos_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (812 * π / 180)) ↔ (n = 92 ∨ n = 268) := by
  sorry

end cos_equality_theorem_l2987_298729


namespace outfit_cost_theorem_l2987_298744

/-- The cost of an outfit given the prices of individual items -/
def outfit_cost (pant_price t_shirt_price jacket_price : ℚ) : ℚ :=
  pant_price + 4 * t_shirt_price + jacket_price

/-- The theorem stating the cost of the outfit given the constraints -/
theorem outfit_cost_theorem (pant_price t_shirt_price jacket_price : ℚ) :
  (4 * pant_price + 8 * t_shirt_price + 2 * jacket_price = 2400) →
  (2 * pant_price + 14 * t_shirt_price + 3 * jacket_price = 2400) →
  (3 * pant_price + 6 * t_shirt_price = 1500) →
  outfit_cost pant_price t_shirt_price jacket_price = 860 := by
  sorry

#eval outfit_cost 340 80 200

end outfit_cost_theorem_l2987_298744


namespace coltons_stickers_coltons_initial_stickers_l2987_298759

theorem coltons_stickers (friends_count : ℕ) (stickers_per_friend : ℕ) 
  (extra_for_mandy : ℕ) (less_for_justin : ℕ) (stickers_left : ℕ) : ℕ :=
  let friends_total := friends_count * stickers_per_friend
  let mandy_stickers := friends_total + extra_for_mandy
  let justin_stickers := mandy_stickers - less_for_justin
  let given_away := friends_total + mandy_stickers + justin_stickers
  given_away + stickers_left

theorem coltons_initial_stickers : 
  coltons_stickers 3 4 2 10 42 = 72 := by sorry

end coltons_stickers_coltons_initial_stickers_l2987_298759


namespace die_roll_probability_l2987_298723

theorem die_roll_probability (p_greater_than_four : ℚ) 
  (h : p_greater_than_four = 1/3) : 
  1 - p_greater_than_four = 2/3 := by
  sorry

end die_roll_probability_l2987_298723


namespace sum_of_even_factors_720_l2987_298797

def sum_of_even_factors (n : ℕ) : ℕ := sorry

theorem sum_of_even_factors_720 : sum_of_even_factors 720 = 2340 := by sorry

end sum_of_even_factors_720_l2987_298797


namespace buddy_met_66_boys_l2987_298769

/-- The number of girl students in the third grade -/
def num_girls : ℕ := 57

/-- The total number of third graders Buddy met -/
def total_students : ℕ := 123

/-- The number of boy students Buddy met -/
def num_boys : ℕ := total_students - num_girls

theorem buddy_met_66_boys : num_boys = 66 := by
  sorry

end buddy_met_66_boys_l2987_298769


namespace z_in_fourth_quadrant_l2987_298735

-- Define the complex number z
def z : ℂ := (2 - Complex.I) ^ 2

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l2987_298735


namespace simplify_and_evaluate_l2987_298739

theorem simplify_and_evaluate (a : ℝ) (h : a = 19) :
  (1 + 2 / (a - 1)) / ((a^2 + 2*a + 1) / (a - 1)) = 1 / 20 := by
  sorry

end simplify_and_evaluate_l2987_298739


namespace y_intercept_of_line_y_intercept_specific_line_l2987_298786

/-- The y-intercept of a line with equation ax + by + c = 0 is -c/b when b ≠ 0 -/
theorem y_intercept_of_line (a b c : ℝ) (hb : b ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let y_intercept := {y : ℝ | (0, y) ∈ line}
  y_intercept = {-c/b} :=
by sorry

/-- The y-intercept of the line x + 2y + 1 = 0 is -1/2 -/
theorem y_intercept_specific_line :
  let line := {p : ℝ × ℝ | p.1 + 2 * p.2 + 1 = 0}
  let y_intercept := {y : ℝ | (0, y) ∈ line}
  y_intercept = {-1/2} :=
by sorry

end y_intercept_of_line_y_intercept_specific_line_l2987_298786


namespace sequence_general_formula_l2987_298737

/-- Given a sequence {a_n} where a₁ = 6 and aₙ₊₁/aₙ = (n+3)/n for n ≥ 1,
    this theorem states that aₙ = n(n+1)(n+2) for all n ≥ 1 -/
theorem sequence_general_formula (a : ℕ → ℝ) 
    (h1 : a 1 = 6)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = (n + 3) / n) :
  ∀ n : ℕ, n ≥ 1 → a n = n * (n + 1) * (n + 2) := by
  sorry

end sequence_general_formula_l2987_298737


namespace water_depth_at_points_l2987_298709

/-- The depth function that calculates water depth based on Ron's height -/
def depth (x : ℝ) : ℝ := 16 * x

/-- Ron's height at point A -/
def ronHeightA : ℝ := 13

/-- Ron's height at point B -/
def ronHeightB : ℝ := ronHeightA + 4

/-- Theorem: The depth of water at points A and B -/
theorem water_depth_at_points : 
  depth ronHeightA = 208 ∧ depth ronHeightB = 272 := by
  sorry

/-- Dean's height relative to Ron -/
def deanHeight (ronHeight : ℝ) : ℝ := ronHeight + 9

/-- Alex's height relative to Dean -/
def alexHeight (deanHeight : ℝ) : ℝ := deanHeight - 5

end water_depth_at_points_l2987_298709


namespace max_spheres_in_specific_cylinder_l2987_298716

/-- Represents a cylindrical container -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  diameter : ℝ

/-- Calculates the maximum number of spheres that can fit in a cylinder -/
def maxSpheresInCylinder (c : Cylinder) (s : Sphere) : ℕ :=
  sorry

theorem max_spheres_in_specific_cylinder :
  let c := Cylinder.mk 82 225
  let s := Sphere.mk 38
  maxSpheresInCylinder c s = 21 := by
  sorry

end max_spheres_in_specific_cylinder_l2987_298716


namespace trigonometric_equation_solution_l2987_298766

theorem trigonometric_equation_solution (t : ℝ) : 
  2 * (Real.sin t)^4 * (Real.sin (2 * t) - 3) - 2 * (Real.sin t)^2 * (Real.sin (2 * t) - 3) - 1 = 0 ↔ 
  (∃ k : ℤ, t = π/4 * (4 * k + 1)) ∨ 
  (∃ n : ℤ, t = (-1)^n * (1/2 * Real.arcsin (1 - Real.sqrt 3)) + π/2 * n) :=
sorry

end trigonometric_equation_solution_l2987_298766
