import Mathlib

namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3689_368903

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a / (b * c) + c / (a * b) - b / (a * c) = 1 / (a * Real.cos C + c * Real.cos A)) →
  (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2) →
  (b / Real.sin B = 2 * Real.sqrt 3) →
  (c > a) →
  (B = π / 3 ∧ c = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3689_368903


namespace NUMINAMATH_CALUDE_decimal_23_to_binary_l3689_368984

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

-- Theorem statement
theorem decimal_23_to_binary :
  decimalToBinary 23 = [true, true, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_decimal_23_to_binary_l3689_368984


namespace NUMINAMATH_CALUDE_sixth_term_is_negative_four_l3689_368973

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- Sum of first 3 terms is 12
  sum_first_three : a + (a + d) + (a + 2*d) = 12
  -- Fourth term is 0
  fourth_term_zero : a + 3*d = 0

/-- The sixth term of the arithmetic sequence is -4 -/
theorem sixth_term_is_negative_four (seq : ArithmeticSequence) : 
  seq.a + 5*seq.d = -4 := by
  sorry

#check sixth_term_is_negative_four

end NUMINAMATH_CALUDE_sixth_term_is_negative_four_l3689_368973


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l3689_368967

theorem tan_half_product_squared (a b : ℝ) :
  6 * (Real.cos a + Real.cos b) + 3 * (Real.sin a + Real.sin b) + 5 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2)) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l3689_368967


namespace NUMINAMATH_CALUDE_john_quilt_cost_l3689_368942

def quilt_cost (length width cost_per_sqft discount_rate tax_rate : ℝ) : ℝ :=
  let area := length * width
  let initial_cost := area * cost_per_sqft
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_cost := discounted_cost * (1 + tax_rate)
  total_cost

theorem john_quilt_cost :
  quilt_cost 12 15 70 0.1 0.05 = 11907 := by
  sorry

end NUMINAMATH_CALUDE_john_quilt_cost_l3689_368942


namespace NUMINAMATH_CALUDE_angela_action_figures_l3689_368955

theorem angela_action_figures (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 →
  sold_fraction = 1 / 4 →
  given_fraction = 1 / 3 →
  initial - (initial * sold_fraction).floor - ((initial - (initial * sold_fraction).floor) * given_fraction).floor = 12 := by
sorry

end NUMINAMATH_CALUDE_angela_action_figures_l3689_368955


namespace NUMINAMATH_CALUDE_initial_brownies_count_l3689_368948

/-- The number of brownies initially made by Mother -/
def initial_brownies : ℕ := sorry

/-- The number of brownies eaten by Father -/
def father_eaten : ℕ := 8

/-- The number of brownies eaten by Mooney -/
def mooney_eaten : ℕ := 4

/-- The number of new brownies added the next morning -/
def new_brownies : ℕ := 24

/-- The total number of brownies after adding the new ones -/
def total_brownies : ℕ := 36

theorem initial_brownies_count : initial_brownies = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_brownies_count_l3689_368948


namespace NUMINAMATH_CALUDE_range_of_x_l3689_368952

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) :
  x ≤ -2 ∨ x ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l3689_368952


namespace NUMINAMATH_CALUDE_insurance_problem_l3689_368945

/-- Number of policyholders -/
def n : ℕ := 10000

/-- Claim payment amount in yuan -/
def claim_payment : ℕ := 10000

/-- Operational cost in yuan -/
def operational_cost : ℕ := 50000

/-- Probability of the company paying at least one claim -/
def prob_at_least_one_claim : ℝ := 1 - 0.999^n

/-- Probability of a single policyholder making a claim -/
def p : ℝ := 0.001

/-- Minimum premium that ensures non-negative expected profit -/
def min_premium : ℝ := 15

theorem insurance_problem (a : ℝ) :
  (1 - (1 - p)^n = prob_at_least_one_claim) ∧
  (a ≥ min_premium ↔ n * a - n * p * claim_payment - operational_cost ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_insurance_problem_l3689_368945


namespace NUMINAMATH_CALUDE_max_notebooks_purchasable_l3689_368901

def total_money : ℚ := 30
def notebook_cost : ℚ := 2.4

theorem max_notebooks_purchasable :
  ⌊total_money / notebook_cost⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchasable_l3689_368901


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l3689_368918

theorem beta_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (beta_day1_score beta_day1_total : ℕ)
  (beta_day2_score beta_day2_total : ℕ) :
  alpha_day1_score = 210 →
  alpha_day1_total = 400 →
  alpha_day2_score = 210 →
  alpha_day2_total = 300 →
  beta_day1_total + beta_day2_total = 700 →
  beta_day1_total < 400 →
  beta_day2_total < 400 →
  beta_day1_score > 0 →
  beta_day2_score > 0 →
  (beta_day1_score : ℚ) / beta_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total →
  (beta_day2_score : ℚ) / beta_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total →
  (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 3/5 →
  (beta_day1_score + beta_day2_score : ℚ) / 700 ≤ 139/700 :=
by sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l3689_368918


namespace NUMINAMATH_CALUDE_function_composition_equality_l3689_368966

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem function_composition_equality (a : ℝ) :
  f a (f a 0) = 4*a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3689_368966


namespace NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l3689_368927

/-- Calculates the final number of fish in Jonah's aquarium after a series of events. -/
def final_fish_count (initial : ℕ) (added : ℕ) (eaten : ℕ) (returned : ℕ) (exchanged : ℕ) : ℕ :=
  initial + added - eaten - returned + exchanged

/-- Theorem stating that given the initial conditions and series of events, 
    the final number of fish in Jonah's aquarium is 11. -/
theorem jonah_aquarium_fish_count : 
  final_fish_count 14 2 6 2 3 = 11 := by sorry

end NUMINAMATH_CALUDE_jonah_aquarium_fish_count_l3689_368927


namespace NUMINAMATH_CALUDE_remainder_problem_l3689_368923

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 38 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3689_368923


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3689_368919

/-- Given non-zero vectors a and b in a real inner product space, 
    if |a| = √2|b| and (a - b) ⊥ (2a + 3b), 
    then the angle between a and b is 3π/4. -/
theorem angle_between_vectors 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a‖ = Real.sqrt 2 * ‖b‖) 
  (h2 : @inner ℝ V _ (a - b) (2 • a + 3 • b) = 0) : 
  Real.arccos ((@inner ℝ V _ a b) / (‖a‖ * ‖b‖)) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3689_368919


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3689_368982

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set S
def S : Set Nat := {1, 3}

-- Define set T
def T : Set Nat := {4}

-- Theorem statement
theorem complement_union_theorem :
  (Sᶜ ∪ T) = {2, 4} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3689_368982


namespace NUMINAMATH_CALUDE_zongzi_price_calculation_l3689_368902

theorem zongzi_price_calculation (pork_total red_bean_total : ℕ) 
  (h1 : pork_total = 8000)
  (h2 : red_bean_total = 6000)
  (h3 : ∃ (n : ℕ), n ≠ 0 ∧ pork_total = n * 40 ∧ red_bean_total = n * 30) :
  ∃ (pork_price red_bean_price : ℕ),
    pork_price = 40 ∧
    red_bean_price = 30 ∧
    pork_price = red_bean_price + 10 ∧
    pork_total = red_bean_total + 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_zongzi_price_calculation_l3689_368902


namespace NUMINAMATH_CALUDE_stating_clock_hands_overlap_at_316_l3689_368951

/-- Represents the number of degrees the hour hand moves in one minute -/
def hourHandDegPerMin : ℝ := 0.5

/-- Represents the number of degrees the minute hand moves in one minute -/
def minuteHandDegPerMin : ℝ := 6

/-- Represents the number of degrees between the hour and minute hands at 3:00 -/
def initialAngle : ℝ := 90

/-- 
Theorem stating that the hour and minute hands of a clock overlap 16 minutes after 3:00
-/
theorem clock_hands_overlap_at_316 :
  ∃ (x : ℝ), x > 0 ∧ x < 60 ∧ 
  minuteHandDegPerMin * x - hourHandDegPerMin * x = initialAngle ∧
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_stating_clock_hands_overlap_at_316_l3689_368951


namespace NUMINAMATH_CALUDE_fraction_order_l3689_368938

theorem fraction_order : 
  let f1 := 18 / 14
  let f2 := 16 / 12
  let f3 := 20 / 16
  5 / 4 < f1 ∧ f1 < f2 ∧ f3 < f1 := by sorry

end NUMINAMATH_CALUDE_fraction_order_l3689_368938


namespace NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3689_368994

theorem equilateral_triangle_hexagon_area (s t : ℝ) : 
  s > 0 → t > 0 → -- Ensure positive side lengths
  3 * s = 6 * t → -- Equal perimeters
  (s^2 * Real.sqrt 3) / 4 = 9 → -- Triangle area is 9
  (3 * t^2 * Real.sqrt 3) / 2 = 13.5 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3689_368994


namespace NUMINAMATH_CALUDE_unique_special_function_l3689_368930

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f ((x + 2) + 2) = f (x + 2)) ∧  -- g(x) = f(x+2) is even
  (∀ x, x ∈ Set.Icc 0 2 → f x = x)  -- f(x) = x for x ∈ [0, 2]

/-- There exists a unique function satisfying the special_function conditions -/
theorem unique_special_function : ∃! f : ℝ → ℝ, special_function f :=
sorry

end NUMINAMATH_CALUDE_unique_special_function_l3689_368930


namespace NUMINAMATH_CALUDE_bus_problem_l3689_368961

/-- The number of people on a bus after a stop, given the original number and the difference between those who left and those who got on. -/
def peopleOnBusAfterStop (originalCount : ℕ) (exitEnterDifference : ℕ) : ℕ :=
  originalCount - exitEnterDifference

/-- Theorem stating that given the initial conditions, the number of people on the bus after the stop is 29. -/
theorem bus_problem :
  peopleOnBusAfterStop 38 9 = 29 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l3689_368961


namespace NUMINAMATH_CALUDE_vector_problem_l3689_368936

/-- Given vectors in 2D space -/
def OA : Fin 2 → ℝ := ![1, -2]
def OB : Fin 2 → ℝ := ![4, -1]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, m + 1]

/-- Vector AB -/
def AB : Fin 2 → ℝ := ![3, 1]

/-- Vector AC -/
def AC (m : ℝ) : Fin 2 → ℝ := ![m - 1, m + 3]

/-- Vector BC -/
def BC (m : ℝ) : Fin 2 → ℝ := ![m - 4, m + 2]

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- Two vectors are perpendicular if their dot product is zero -/
def are_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 = 0

/-- Triangle ABC is right-angled if any two of its sides are perpendicular -/
def is_right_angled (m : ℝ) : Prop :=
  are_perpendicular AB (AC m) ∨ are_perpendicular AB (BC m) ∨ are_perpendicular (AC m) (BC m)

theorem vector_problem (m : ℝ) :
  (are_parallel AB (OC m) → m = -3/2) ∧
  (is_right_angled m → m = 0 ∨ m = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3689_368936


namespace NUMINAMATH_CALUDE_circle_equation_l3689_368915

/-- The circle passing through points A(-1, 1) and B(-2, -2), with center C lying on the line x+y-1=0, has the standard equation (x - 3)² + (y + 2)² = 25 -/
theorem circle_equation : 
  ∀ (C : ℝ × ℝ) (r : ℝ),
  (C.1 + C.2 - 1 = 0) →  -- Center C lies on the line x+y-1=0
  ((-1 - C.1)^2 + (1 - C.2)^2 = r^2) →  -- Circle passes through A(-1, 1)
  ((-2 - C.1)^2 + (-2 - C.2)^2 = r^2) →  -- Circle passes through B(-2, -2)
  ∀ (x y : ℝ), 
  ((x - 3)^2 + (y + 2)^2 = 25) ↔ ((x - C.1)^2 + (y - C.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3689_368915


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l3689_368985

theorem midpoint_of_complex_line_segment : 
  let z₁ : ℂ := 2 + 4 * Complex.I
  let z₂ : ℂ := -6 + 10 * Complex.I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -2 + 7 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l3689_368985


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3689_368913

theorem inserted_numbers_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ r : ℝ, r > 0 ∧ a = 4 * r ∧ b = 4 * r^2) →  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ 16 = b + d) →           -- Arithmetic progression condition
  b = a + 4 →                                   -- Difference condition
  a + b = 8 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3689_368913


namespace NUMINAMATH_CALUDE_horner_method_v4_l3689_368939

def horner_polynomial (x : ℝ) : ℝ := 1 + 8*x + 7*x^2 + 5*x^4 + 4*x^5 + 3*x^6

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 4
  let v2 := v1 * x + 5
  let v3 := v2 * x + 0
  v3 * x + 7

theorem horner_method_v4 :
  horner_v4 5 = 2507 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l3689_368939


namespace NUMINAMATH_CALUDE_john_needs_two_planks_l3689_368983

/-- The number of planks needed for a house wall, given the total number of nails and nails per plank. -/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) : ℕ :=
  total_nails / nails_per_plank

/-- Theorem stating that John needs 2 planks for the house wall. -/
theorem john_needs_two_planks :
  let total_nails : ℕ := 4
  let nails_per_plank : ℕ := 2
  planks_needed total_nails nails_per_plank = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_two_planks_l3689_368983


namespace NUMINAMATH_CALUDE_nine_integer_chord_lengths_l3689_368907

/-- Represents a circle with a given radius and a point inside it -/
structure CircleWithPoint where
  radius : ℝ
  pointDistance : ℝ

/-- Counts the number of different integer chord lengths containing the given point -/
def countIntegerChordLengths (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle of radius 25 and a point 13 units from the center,
    there are exactly 9 different integer chord lengths -/
theorem nine_integer_chord_lengths :
  let c := CircleWithPoint.mk 25 13
  countIntegerChordLengths c = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_integer_chord_lengths_l3689_368907


namespace NUMINAMATH_CALUDE_stork_comparison_l3689_368990

def initial_sparrows : ℕ := 12
def initial_pigeons : ℕ := 5
def initial_crows : ℕ := 9
def initial_storks : ℕ := 8
def additional_storks : ℕ := 15
def additional_pigeons : ℕ := 4

def final_storks : ℕ := initial_storks + additional_storks
def final_pigeons : ℕ := initial_pigeons + additional_pigeons
def final_other_birds : ℕ := initial_sparrows + final_pigeons + initial_crows

theorem stork_comparison : 
  (final_storks : ℤ) - (final_other_birds : ℤ) = -7 := by sorry

end NUMINAMATH_CALUDE_stork_comparison_l3689_368990


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l3689_368934

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (total_subjects : ℕ) 
  (h1 : english = 86)
  (h2 : mathematics = 89)
  (h3 : physics = 82)
  (h4 : biology = 81)
  (h5 : average = 85)
  (h6 : total_subjects = 5) :
  let total_marks := average * total_subjects
  let known_subjects_marks := english + mathematics + physics + biology
  let chemistry := total_marks - known_subjects_marks
  chemistry = 87 := by
    sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l3689_368934


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l3689_368970

/-- Given that α^2005 + β^2005 can be expressed as a polynomial in α+β and αβ,
    this function represents that polynomial. -/
def polynomial_expression (x y : ℝ) : ℝ := sorry

/-- The sum of the coefficients of the polynomial expression -/
def sum_of_coefficients : ℝ := sorry

/-- Theorem stating that the sum of the coefficients is 1 -/
theorem sum_of_coefficients_is_one : sum_of_coefficients = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l3689_368970


namespace NUMINAMATH_CALUDE_total_pears_picked_l3689_368957

theorem total_pears_picked (sara tim emily max : ℕ) 
  (h_sara : sara = 6)
  (h_tim : tim = 5)
  (h_emily : emily = 9)
  (h_max : max = 12) :
  sara + tim + emily + max = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3689_368957


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3689_368954

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 20*x^2 + 400) * (x^2 - 20) = x^6 - 8000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3689_368954


namespace NUMINAMATH_CALUDE_min_colors_17gon_l3689_368958

/-- A coloring of the vertices of a regular 17-gon -/
def Coloring := Fin 17 → ℕ

/-- The distance between two vertices in a 17-gon -/
def distance (i j : Fin 17) : Fin 17 := 
  Fin.ofNat ((i.val - j.val + 17) % 17)

/-- Whether two vertices should have different colors -/
def should_differ (i j : Fin 17) : Prop :=
  let d := distance i j
  d = 2 ∨ d = 4 ∨ d = 8 ∨ d = 15 ∨ d = 13 ∨ d = 9

/-- A valid coloring of the 17-gon -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ i j : Fin 17, should_differ i j → c i ≠ c j

/-- The main theorem -/
theorem min_colors_17gon : 
  (∃ c : Coloring, is_valid_coloring c ∧ Finset.card (Finset.image c Finset.univ) = 4) ∧
  (∀ c : Coloring, is_valid_coloring c → Finset.card (Finset.image c Finset.univ) ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_colors_17gon_l3689_368958


namespace NUMINAMATH_CALUDE_cubic_function_property_l3689_368991

/-- A cubic function passing through the point (-3, -2) -/
structure CubicFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  passes_through : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = -2

/-- Theorem: For a cubic function g(x) = px^3 + qx^2 + rx + s passing through (-3, -2),
    the expression 12p - 6q + 3r - s equals 2 -/
theorem cubic_function_property (g : CubicFunction) : 
  12 * g.p - 6 * g.q + 3 * g.r - g.s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3689_368991


namespace NUMINAMATH_CALUDE_product_of_large_integers_l3689_368988

theorem product_of_large_integers : ∃ (A B : ℕ), 
  (A > 2009^182) ∧ 
  (B > 2009^182) ∧ 
  (3^2008 + 4^2009 = A * B) := by
sorry

end NUMINAMATH_CALUDE_product_of_large_integers_l3689_368988


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3689_368956

theorem wire_cutting_problem (wire_length : ℕ) (num_pieces : ℕ) (piece_length : ℕ) :
  wire_length = 1040 ∧ 
  num_pieces = 15 ∧ 
  wire_length = num_pieces * piece_length ∧
  piece_length > 0 →
  piece_length = 66 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3689_368956


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3689_368953

/-- A geometric sequence with real terms -/
def GeometricSequence := ℕ → ℝ

/-- Sum of the first n terms of a geometric sequence -/
def SumGeometric (a : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_sum 
  (a : GeometricSequence) 
  (h1 : SumGeometric a 10 = 10) 
  (h2 : SumGeometric a 30 = 70) : 
  SumGeometric a 40 = 150 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3689_368953


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l3689_368910

theorem square_plus_inverse_square (a : ℝ) (h : a - (1 / a) = 5) : a^2 + (1 / a^2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l3689_368910


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3689_368941

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) : 
  base = 12 →
  (1/2) * base * height = 30 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3689_368941


namespace NUMINAMATH_CALUDE_teacher_age_l3689_368998

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 30 →
  student_avg_age = 14 →
  total_avg_age = 15 →
  (num_students : ℝ) * student_avg_age + 45 = (num_students + 1 : ℝ) * total_avg_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l3689_368998


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3689_368937

theorem product_of_sum_of_squares (a b n k : ℝ) :
  let K := a^2 + b^2
  let P := n^2 + k^2
  K * P = (a*n + b*k)^2 + (a*k - b*n)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3689_368937


namespace NUMINAMATH_CALUDE_M_equals_N_l3689_368977

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3689_368977


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3689_368965

theorem emily_egg_collection (total_baskets : ℕ) (first_group_baskets : ℕ) (second_group_baskets : ℕ)
  (eggs_per_first_basket : ℕ) (eggs_per_second_basket : ℕ) :
  total_baskets = first_group_baskets + second_group_baskets →
  first_group_baskets = 450 →
  second_group_baskets = 405 →
  eggs_per_first_basket = 36 →
  eggs_per_second_basket = 42 →
  first_group_baskets * eggs_per_first_basket + second_group_baskets * eggs_per_second_basket = 33210 := by
sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3689_368965


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3689_368929

theorem quadratic_equation_root (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x => p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f (-1) = 0) →
  (f (-r * (p - q) / (p * (q - r))) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3689_368929


namespace NUMINAMATH_CALUDE_wire_length_is_250_meters_l3689_368931

-- Define the density of copper
def copper_density : Real := 8900

-- Define the volume of wire bought by Chek
def wire_volume : Real := 0.5e-3

-- Define the diagonal of the wire's square cross-section
def wire_diagonal : Real := 2e-3

-- Theorem to prove
theorem wire_length_is_250_meters :
  let cross_section_area := (wire_diagonal ^ 2) / 2
  let wire_length := wire_volume / cross_section_area
  wire_length = 250 := by sorry

end NUMINAMATH_CALUDE_wire_length_is_250_meters_l3689_368931


namespace NUMINAMATH_CALUDE_common_altitude_of_triangles_l3689_368989

theorem common_altitude_of_triangles (area1 area2 base1 base2 : ℝ) 
  (h_area1 : area1 = 800)
  (h_area2 : area2 = 1200)
  (h_base1 : base1 = 40)
  (h_base2 : base2 = 60)
  (h_positive1 : area1 > 0)
  (h_positive2 : area2 > 0)
  (h_positive3 : base1 > 0)
  (h_positive4 : base2 > 0) :
  ∃ h : ℝ, h > 0 ∧ area1 = (1/2) * base1 * h ∧ area2 = (1/2) * base2 * h ∧ h = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_common_altitude_of_triangles_l3689_368989


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l3689_368935

/-- Represents the duration of a workday in hours -/
def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℝ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℝ := 3 * first_meeting_minutes

/-- Calculates the total minutes in a workday -/
def workday_minutes : ℝ := workday_hours * 60

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes

/-- Theorem stating that the percentage of workday spent in meetings is 40% -/
theorem workday_meeting_percentage :
  (total_meeting_minutes / workday_minutes) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l3689_368935


namespace NUMINAMATH_CALUDE_at_least_one_acute_angle_not_greater_than_45_l3689_368964

-- Define a right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  right_angle : C = 90
  angle_sum : A + B + C = 180

-- Theorem statement
theorem at_least_one_acute_angle_not_greater_than_45 (t : RightTriangle) :
  t.A ≤ 45 ∨ t.B ≤ 45 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_acute_angle_not_greater_than_45_l3689_368964


namespace NUMINAMATH_CALUDE_t_shirts_per_package_l3689_368996

theorem t_shirts_per_package (total_shirts : ℕ) (num_packages : ℕ) 
  (h1 : total_shirts = 51) (h2 : num_packages = 17) : 
  total_shirts / num_packages = 3 := by
  sorry

end NUMINAMATH_CALUDE_t_shirts_per_package_l3689_368996


namespace NUMINAMATH_CALUDE_amy_candy_problem_l3689_368959

theorem amy_candy_problem (initial_candy : ℕ) : ∃ (given : ℕ), 
  given + 5 ≤ initial_candy ∧ given - 5 = 1 → given = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_candy_problem_l3689_368959


namespace NUMINAMATH_CALUDE_polynomial_identity_l3689_368963

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3689_368963


namespace NUMINAMATH_CALUDE_postal_stamp_problem_l3689_368949

theorem postal_stamp_problem :
  ∀ (x : ℕ),
  (75 : ℕ) = 40 + (75 - 40) →
  (480 : ℕ) = 40 * 5 + (75 - 40) * x →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_postal_stamp_problem_l3689_368949


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3689_368971

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x^2 + 1 = 6 * x) →
  (∀ x, a * x^2 + b * x + c = 0) →
  b = 6 →
  a = -3 ∧ c = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3689_368971


namespace NUMINAMATH_CALUDE_school_boys_count_l3689_368917

theorem school_boys_count :
  ∀ (total_students : ℕ) (boys : ℕ),
    total_students = 400 →
    boys + (boys * total_students / 100) = total_students →
    boys = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3689_368917


namespace NUMINAMATH_CALUDE_box_side_length_l3689_368950

/-- Proves that the length of one side of a cubic box can be calculated given the total volume,
    total cost, and cost per box. -/
theorem box_side_length 
  (cost_per_box : ℝ) 
  (total_volume : ℝ) 
  (total_cost : ℝ) 
  (cost_per_box_positive : cost_per_box > 0)
  (total_volume_positive : total_volume > 0)
  (total_cost_positive : total_cost > 0) :
  ∃ (side_length : ℝ), 
    side_length = (total_volume / (total_cost / cost_per_box)) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_box_side_length_l3689_368950


namespace NUMINAMATH_CALUDE_least_3digit_base8_divisible_by_7_l3689_368946

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem least_3digit_base8_divisible_by_7 :
  let n := 106
  isThreeDigitBase8 n ∧ 
  base8ToDecimal n % 7 = 0 ∧
  ∀ m : ℕ, isThreeDigitBase8 m ∧ base8ToDecimal m % 7 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_3digit_base8_divisible_by_7_l3689_368946


namespace NUMINAMATH_CALUDE_largest_common_number_l3689_368904

def first_sequence (n : ℕ) : ℤ := 5 + 8 * (n - 1)

def second_sequence (n : ℕ) : ℤ := 3 + 9 * (n - 1)

def is_common (x : ℤ) : Prop :=
  ∃ (n m : ℕ), first_sequence n = x ∧ second_sequence m = x

theorem largest_common_number :
  ∃ (x : ℤ), is_common x ∧ x ≤ 150 ∧
  ∀ (y : ℤ), is_common y ∧ y ≤ 150 → y ≤ x ∧
  x = 93 :=
sorry

end NUMINAMATH_CALUDE_largest_common_number_l3689_368904


namespace NUMINAMATH_CALUDE_max_distance_with_swap_20000_30000_l3689_368981

/-- Represents the maximum distance a car can travel with one tire swap -/
def maxDistanceWithSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + min frontTireLife (rearTireLife - frontTireLife)

/-- Theorem stating the maximum distance for the given problem -/
theorem max_distance_with_swap_20000_30000 :
  maxDistanceWithSwap 20000 30000 = 30000 := by
  sorry

#eval maxDistanceWithSwap 20000 30000

end NUMINAMATH_CALUDE_max_distance_with_swap_20000_30000_l3689_368981


namespace NUMINAMATH_CALUDE_percentage_calculation_l3689_368925

theorem percentage_calculation (p : ℝ) : 
  0.25 * 900 = p / 100 * 1600 - 15 → p = 1500 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3689_368925


namespace NUMINAMATH_CALUDE_robins_walk_distance_l3689_368908

/-- The total distance Robin walks given his journey to the city center -/
theorem robins_walk_distance (house_to_center : ℕ) (initial_walk : ℕ) : 
  house_to_center = 500 →
  initial_walk = 200 →
  initial_walk + initial_walk + house_to_center = 900 := by
  sorry

end NUMINAMATH_CALUDE_robins_walk_distance_l3689_368908


namespace NUMINAMATH_CALUDE_twins_age_problem_l3689_368905

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 17 → age = 8 :=
by sorry

end NUMINAMATH_CALUDE_twins_age_problem_l3689_368905


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l3689_368962

/-- For any acute triangle ABC with side lengths a, b, c and angles A, B, C,
    the inequality 4abc < (a^2 + b^2 + c^2)(a cos A + b cos B + c cos C) ≤ 9/2 abc holds. -/
theorem acute_triangle_inequality (a b c : ℝ) (A B C : Real) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_acute : A > 0 ∧ B > 0 ∧ C > 0)
    (h_angles : A + B + C = π)
    (h_cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
                    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
                    c^2 = a^2 + b^2 - 2*a*b*Real.cos C) :
  4*a*b*c < (a^2 + b^2 + c^2)*(a*Real.cos A + b*Real.cos B + c*Real.cos C) ∧
  (a^2 + b^2 + c^2)*(a*Real.cos A + b*Real.cos B + c*Real.cos C) ≤ 9/2*a*b*c :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l3689_368962


namespace NUMINAMATH_CALUDE_bobs_assorted_candies_l3689_368928

/-- The problem of calculating Bob's assorted candies -/
theorem bobs_assorted_candies 
  (total_candies : ℕ) 
  (chewing_gums : ℕ) 
  (chocolate_bars : ℕ) 
  (h1 : total_candies = 50)
  (h2 : chewing_gums = 15)
  (h3 : chocolate_bars = 20) :
  total_candies - (chewing_gums + chocolate_bars) = 15 :=
by sorry

end NUMINAMATH_CALUDE_bobs_assorted_candies_l3689_368928


namespace NUMINAMATH_CALUDE_min_weights_to_balance_three_grams_l3689_368987

/-- Represents a combination of weights -/
structure WeightCombination :=
  (nine_gram : ℤ)
  (thirteen_gram : ℤ)

/-- Calculates the total weight of a combination -/
def total_weight (w : WeightCombination) : ℤ :=
  9 * w.nine_gram + 13 * w.thirteen_gram

/-- Calculates the total number of weights used -/
def num_weights (w : WeightCombination) : ℕ :=
  w.nine_gram.natAbs + w.thirteen_gram.natAbs

/-- Checks if a combination balances 3 grams -/
def balances_three_grams (w : WeightCombination) : Prop :=
  total_weight w = 3

/-- The set of all weight combinations that balance 3 grams -/
def balancing_combinations : Set WeightCombination :=
  {w | balances_three_grams w}

theorem min_weights_to_balance_three_grams :
  ∃ (w : WeightCombination),
    w ∈ balancing_combinations ∧
    num_weights w = 7 ∧
    ∀ (w' : WeightCombination),
      w' ∈ balancing_combinations →
      num_weights w' ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_min_weights_to_balance_three_grams_l3689_368987


namespace NUMINAMATH_CALUDE_area_FYG_value_l3689_368944

/-- Represents a trapezoid EFGH with point Y at the intersection of diagonals -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FYG in the given trapezoid -/
def area_FYG (t : Trapezoid) : ℝ := sorry

theorem area_FYG_value (t : Trapezoid) (h1 : t.EF = 15) (h2 : t.GH = 25) (h3 : t.area = 200) :
  area_FYG t = 46.875 := by sorry

end NUMINAMATH_CALUDE_area_FYG_value_l3689_368944


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3689_368920

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 9 + a 27 = 12) : 
  a 13 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3689_368920


namespace NUMINAMATH_CALUDE_unique_rectangle_arrangement_l3689_368926

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.height)

/-- Checks if two rectangles have equal perimeters -/
def equalPerimeters (r1 r2 : Rectangle) : Prop := r1.perimeter = r2.perimeter

/-- Checks if the total area of two rectangles is 81 -/
def totalAreaIs81 (r1 r2 : Rectangle) : Prop := r1.area + r2.area = 81

/-- The main theorem stating that the only way to arrange 81 unit squares into two rectangles
    with equal perimeters is to form rectangles with dimensions 3 × 11 and 6 × 8 -/
theorem unique_rectangle_arrangement :
  ∀ r1 r2 : Rectangle,
    equalPerimeters r1 r2 → totalAreaIs81 r1 r2 →
    ((r1.width = 3 ∧ r1.height = 11) ∧ (r2.width = 6 ∧ r2.height = 8)) ∨
    ((r1.width = 6 ∧ r1.height = 8) ∧ (r2.width = 3 ∧ r2.height = 11)) := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_arrangement_l3689_368926


namespace NUMINAMATH_CALUDE_exam_failure_count_l3689_368974

theorem exam_failure_count (total : ℕ) (pass_percent : ℚ) (fail_count : ℕ) : 
  total = 800 → 
  pass_percent = 35 / 100 → 
  fail_count = total - (pass_percent * total).floor → 
  fail_count = 520 := by
sorry

end NUMINAMATH_CALUDE_exam_failure_count_l3689_368974


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l3689_368980

theorem quadratic_equation_rational_solutions : 
  ∃ (c₁ c₂ : ℕ+), 
    (∀ (c : ℕ+), (∃ (x : ℚ), 3 * x^2 + 7 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
    (c₁.val * c₂.val = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l3689_368980


namespace NUMINAMATH_CALUDE_sum_of_roots_equal_l3689_368979

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  10 = (x^3 - 3*x^2 - 4*x) / (x + 3)

-- Define the derived polynomial
def derived_polynomial (x : ℝ) : ℝ :=
  x^3 - 3*x^2 - 14*x - 30

-- Theorem statement
theorem sum_of_roots_equal :
  ∃ (r₁ r₂ r₃ : ℝ),
    (∀ x, derived_polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
    (∀ x, original_equation x ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
    r₁ + r₂ + r₃ = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equal_l3689_368979


namespace NUMINAMATH_CALUDE_sqrt_320_simplification_l3689_368916

theorem sqrt_320_simplification : Real.sqrt 320 = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_320_simplification_l3689_368916


namespace NUMINAMATH_CALUDE_power_product_equals_four_l3689_368978

theorem power_product_equals_four (x y : ℝ) (h : x + 2 * y = 2) :
  (2 : ℝ) ^ x * (4 : ℝ) ^ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_four_l3689_368978


namespace NUMINAMATH_CALUDE_inequality_proof_l3689_368976

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (3 * x^2 + x * y) + Real.sqrt (3 * y^2 + y * z) + Real.sqrt (3 * z^2 + z * x) ≤ 2 * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3689_368976


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l3689_368914

theorem pigeonhole_on_permutation_sums (n : ℕ) :
  ∀ (p : Fin (2 * n) → Fin (2 * n)),
  Function.Bijective p →
  ∃ i j : Fin (2 * n), i ≠ j ∧ 
    (p i + i.val + 1) % (2 * n) = (p j + j.val + 1) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l3689_368914


namespace NUMINAMATH_CALUDE_larger_number_proof_l3689_368960

theorem larger_number_proof (x y : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 29) 
  (h3 : x * y > 200) : 
  max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3689_368960


namespace NUMINAMATH_CALUDE_inequality_proof_l3689_368932

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3689_368932


namespace NUMINAMATH_CALUDE_no_k_exists_product_odd_primes_minus_one_is_power_l3689_368924

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k odd prime numbers -/
def productFirstKOddPrimes (k : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number k such that the product of the first k odd prime numbers minus 1 is an exact power of a natural number greater than one -/
theorem no_k_exists_product_odd_primes_minus_one_is_power :
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstKOddPrimes k = a^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_k_exists_product_odd_primes_minus_one_is_power_l3689_368924


namespace NUMINAMATH_CALUDE_binary_digit_difference_l3689_368969

/-- The number of digits in the binary representation of a positive integer -/
def binaryDigits (n : ℕ+) : ℕ := Nat.log2 n + 1

/-- The difference in the number of binary digits between 950 and 150 -/
theorem binary_digit_difference : binaryDigits 950 - binaryDigits 150 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l3689_368969


namespace NUMINAMATH_CALUDE_decimal_sum_l3689_368997

theorem decimal_sum : (0.35 : ℚ) + 0.048 + 0.0072 = 0.4052 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l3689_368997


namespace NUMINAMATH_CALUDE_calculate_number_of_children_l3689_368968

/-- Calculates the number of children in a family based on their savings distribution --/
theorem calculate_number_of_children 
  (husband_contribution : ℝ) 
  (wife_contribution : ℝ) 
  (saving_period_months : ℕ) 
  (weeks_per_month : ℕ) 
  (amount_per_child : ℝ) 
  (h1 : husband_contribution = 335)
  (h2 : wife_contribution = 225)
  (h3 : saving_period_months = 6)
  (h4 : weeks_per_month = 4)
  (h5 : amount_per_child = 1680) :
  ⌊(((husband_contribution + wife_contribution) * (saving_period_months * weeks_per_month)) / 2) / amount_per_child⌋ = 4 := by
  sorry


end NUMINAMATH_CALUDE_calculate_number_of_children_l3689_368968


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3689_368986

theorem polynomial_factorization (x : ℝ) :
  x^4 - 6*x^3 + 11*x^2 - 6*x = x*(x - 1)*(x - 2)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3689_368986


namespace NUMINAMATH_CALUDE_sum_reciprocals_and_range_l3689_368947

theorem sum_reciprocals_and_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧
  ({x : ℝ | ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → |x - 2| + |2*x - 1| ≤ 1 / a + 1 / b} = Set.Icc (-1/3) (7/3)) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_and_range_l3689_368947


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l3689_368943

/-- The area of the shaded region in a figure with two concentric squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h1 : large_side = 10) 
  (h2 : small_side = 4) 
  (h3 : large_side > small_side) : 
  (large_side^2 - small_side^2) / 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l3689_368943


namespace NUMINAMATH_CALUDE_inequality_with_product_condition_l3689_368940

theorem inequality_with_product_condition (x y z : ℝ) (h : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_product_condition_l3689_368940


namespace NUMINAMATH_CALUDE_simplify_expression_l3689_368906

theorem simplify_expression : ((-Real.sqrt 3)^2)^(-1/2 : ℝ) = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3689_368906


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3689_368911

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, (x + 1) * (x - 3) < 0 → x > -1) ∧
  ¬(∀ x : ℝ, x > -1 → (x + 1) * (x - 3) < 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3689_368911


namespace NUMINAMATH_CALUDE_fourth_root_of_33177600_l3689_368912

theorem fourth_root_of_33177600 : (33177600 : ℝ) ^ (1/4 : ℝ) = 576 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_33177600_l3689_368912


namespace NUMINAMATH_CALUDE_fourth_sample_is_75_l3689_368975

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) (nthSample : ℕ) : ℕ :=
  firstSample + (populationSize / sampleSize) * (nthSample - 1)

/-- Theorem: In a systematic sampling scheme with a population of 480, a sample size of 20, 
    and a first sample of 3, the fourth sample will be 75 -/
theorem fourth_sample_is_75 :
  systematicSample 480 20 3 4 = 75 := by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_75_l3689_368975


namespace NUMINAMATH_CALUDE_trace_bag_count_is_five_l3689_368995

/-- The weight of one of Gordon's shopping bags in pounds -/
def gordon_bag1_weight : ℕ := 3

/-- The weight of the other of Gordon's shopping bags in pounds -/
def gordon_bag2_weight : ℕ := 7

/-- The weight of each of Trace's shopping bags in pounds -/
def trace_bag_weight : ℕ := 2

/-- The number of Trace's shopping bags -/
def trace_bag_count : ℕ := (gordon_bag1_weight + gordon_bag2_weight) / trace_bag_weight

theorem trace_bag_count_is_five : trace_bag_count = 5 := by
  sorry

#eval trace_bag_count

end NUMINAMATH_CALUDE_trace_bag_count_is_five_l3689_368995


namespace NUMINAMATH_CALUDE_inscribed_square_product_l3689_368999

theorem inscribed_square_product (θ : Real) : 
  θ = π / 6 →  -- 30° in radians
  ∃ (a b : Real),
    -- Conditions
    16 = (2 * a)^2 ∧  -- Area of smaller square
    18 = (a + b)^2 ∧  -- Area of larger square
    a = 2 * Real.sqrt 6 ∧  -- Length of segment a
    b = 2 * Real.sqrt 2 →  -- Length of segment b
    -- Conclusion
    a * b = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_product_l3689_368999


namespace NUMINAMATH_CALUDE_problem_statement_l3689_368992

theorem problem_statement : (2222 - 2002)^2 / 144 = 3025 / 9 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3689_368992


namespace NUMINAMATH_CALUDE_expand_and_simplify_simplify_complex_fraction_l3689_368921

-- Problem 1
theorem expand_and_simplify (x y : ℝ) : 
  (x + y) * (x - y) + y * (y - 2) = x^2 - 2*y := by sorry

-- Problem 2
theorem simplify_complex_fraction (m : ℝ) (hm2 : m ≠ 2) (hm_2 : m ≠ -2) : 
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 2 / (m - 2) := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_simplify_complex_fraction_l3689_368921


namespace NUMINAMATH_CALUDE_bicycle_car_arrival_l3689_368972

theorem bicycle_car_arrival (x : ℝ) (h : x > 0) : 
  (10 / x - 10 / (2 * x) = 1 / 3) ↔ 
  (10 / x = 10 / (2 * x) + 1 / 3) :=
sorry

end NUMINAMATH_CALUDE_bicycle_car_arrival_l3689_368972


namespace NUMINAMATH_CALUDE_sequence_property_l3689_368909

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = 2^n) →
  a 1 = 1 →
  a 101 = 2^5050 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l3689_368909


namespace NUMINAMATH_CALUDE_antifreeze_concentration_l3689_368922

/-- Proves that the concentration of the certain antifreeze is 100% -/
theorem antifreeze_concentration
  (total_volume : ℝ)
  (final_concentration : ℝ)
  (certain_volume : ℝ)
  (other_concentration : ℝ)
  (h1 : total_volume = 55)
  (h2 : final_concentration = 0.20)
  (h3 : certain_volume = 6.11)
  (h4 : other_concentration = 0.10)
  : ∃ (certain_concentration : ℝ),
    certain_concentration = 1 ∧
    certain_volume * certain_concentration +
    (total_volume - certain_volume) * other_concentration =
    total_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_antifreeze_concentration_l3689_368922


namespace NUMINAMATH_CALUDE_negation_equivalence_l3689_368993

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 8*x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8*x + 18 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3689_368993


namespace NUMINAMATH_CALUDE_carbon_monoxide_weight_l3689_368900

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (c o : ℝ) : ℝ := c + o

/-- Theorem: The molecular weight of Carbon monoxide (CO) is 28.01 g/mol -/
theorem carbon_monoxide_weight : molecular_weight carbon_weight oxygen_weight = 28.01 := by
  sorry

end NUMINAMATH_CALUDE_carbon_monoxide_weight_l3689_368900


namespace NUMINAMATH_CALUDE_remaining_quarters_l3689_368933

-- Define the initial amount, total spent, and value of a quarter
def initial_amount : ℚ := 40
def total_spent : ℚ := 32.25
def quarter_value : ℚ := 0.25

-- Theorem to prove
theorem remaining_quarters : 
  (initial_amount - total_spent) / quarter_value = 31 := by
  sorry

end NUMINAMATH_CALUDE_remaining_quarters_l3689_368933
