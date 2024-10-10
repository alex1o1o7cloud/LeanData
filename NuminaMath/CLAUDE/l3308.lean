import Mathlib

namespace product_of_real_parts_is_two_l3308_330881

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 3*z = -7 + 2*i

-- Theorem statement
theorem product_of_real_parts_is_two :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧
  z₁ ≠ z₂ ∧ (z₁.re * z₂.re = 2) :=
sorry

end product_of_real_parts_is_two_l3308_330881


namespace system_solution_l3308_330834

theorem system_solution (x y : ℝ) : 
  (2 * x + 5 * y = 26 ∧ 4 * x - 2 * y = 4) ↔ (x = 3 ∧ y = 4) :=
by sorry

end system_solution_l3308_330834


namespace x_value_l3308_330824

theorem x_value : ∃ x : ℝ, x = 88 * (1 + 0.5) ∧ x = 132 := by sorry

end x_value_l3308_330824


namespace juggling_contest_winner_l3308_330861

/-- Represents the number of rotations for an object over 4 minutes -/
structure Rotations :=
  (minute1 : ℕ) (minute2 : ℕ) (minute3 : ℕ) (minute4 : ℕ)

/-- Calculates the total rotations for a contestant -/
def totalRotations (obj1Count : ℕ) (obj1Rotations : Rotations) 
                   (obj2Count : ℕ) (obj2Rotations : Rotations) : ℕ :=
  obj1Count * (obj1Rotations.minute1 + obj1Rotations.minute2 + obj1Rotations.minute3 + obj1Rotations.minute4) +
  obj2Count * (obj2Rotations.minute1 + obj2Rotations.minute2 + obj2Rotations.minute3 + obj2Rotations.minute4)

theorem juggling_contest_winner (tobyBaseballs : Rotations) (tobyFrisbees : Rotations)
                                (annaApples : Rotations) (annaOranges : Rotations)
                                (jackTennisBalls : Rotations) (jackWaterBalloons : Rotations) :
  tobyBaseballs = ⟨80, 85, 75, 90⟩ →
  tobyFrisbees = ⟨60, 70, 65, 80⟩ →
  annaApples = ⟨101, 99, 98, 102⟩ →
  annaOranges = ⟨95, 90, 92, 93⟩ →
  jackTennisBalls = ⟨82, 81, 85, 87⟩ →
  jackWaterBalloons = ⟨100, 96, 101, 97⟩ →
  (max (totalRotations 5 tobyBaseballs 3 tobyFrisbees)
       (max (totalRotations 4 annaApples 5 annaOranges)
            (totalRotations 6 jackTennisBalls 4 jackWaterBalloons))) = 3586 := by
  sorry

end juggling_contest_winner_l3308_330861


namespace min_value_expression_l3308_330892

theorem min_value_expression (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0)
  (h5 : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + 4 * y^2 - c = 0 →
    |2 * a + b| ≥ |2 * x + y|) :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + 4 * y^2 - z = 0 →
    3 / x - 4 / y + 5 / z ≥ m :=
by sorry

end min_value_expression_l3308_330892


namespace closure_property_implies_divisibility_characterization_l3308_330886

theorem closure_property_implies_divisibility_characterization 
  (S : Set ℤ) 
  (closure : ∀ a b : ℤ, a ∈ S → b ∈ S → (a + b) ∈ S) 
  (has_negative : ∃ n : ℤ, n < 0 ∧ n ∈ S) 
  (has_positive : ∃ p : ℤ, p > 0 ∧ p ∈ S) : 
  ∃ d : ℤ, ∀ x : ℤ, x ∈ S ↔ d ∣ x := by
sorry

end closure_property_implies_divisibility_characterization_l3308_330886


namespace inequality_solution_l3308_330811

theorem inequality_solution (x : ℝ) : 
  1 / (x * (x + 1)) - 1 / ((x + 2) * (x + 3)) < 1 / 5 ↔ 
  x < -3 ∨ (-2 < x ∧ x < -1) ∨ x > 2 := by sorry

end inequality_solution_l3308_330811


namespace playground_area_is_22500_l3308_330855

/-- Represents a rectangular playground --/
structure Playground where
  width : ℝ
  length : ℝ

/-- Properties of the playground --/
def PlaygroundProperties (p : Playground) : Prop :=
  p.length = 2 * p.width + 25 ∧
  2 * (p.length + p.width) = 650

/-- The area of the playground --/
def playgroundArea (p : Playground) : ℝ :=
  p.length * p.width

/-- Theorem: The area of the playground with given properties is 22,500 square feet --/
theorem playground_area_is_22500 :
  ∀ p : Playground, PlaygroundProperties p → playgroundArea p = 22500 := by
  sorry

end playground_area_is_22500_l3308_330855


namespace quadratic_equation_solution_sum_l3308_330894

theorem quadratic_equation_solution_sum : ∃ (c d : ℝ), 
  (c^2 - 6*c + 15 = 25) ∧ 
  (d^2 - 6*d + 15 = 25) ∧ 
  (c ≥ d) ∧ 
  (3*c + 2*d = 15 + Real.sqrt 19) := by
  sorry

end quadratic_equation_solution_sum_l3308_330894


namespace last_digit_periodic_l3308_330897

theorem last_digit_periodic (n : ℕ) : n^n % 10 = (n + 20)^(n + 20) % 10 := by
  sorry

end last_digit_periodic_l3308_330897


namespace not_passes_third_quadrant_l3308_330840

/-- A linear function f(x) = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.first  => x > 0 ∧ y > 0
  | Quadrant.second => x < 0 ∧ y > 0
  | Quadrant.third  => x < 0 ∧ y < 0
  | Quadrant.fourth => x > 0 ∧ y < 0

/-- A linear function passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passesThroughQuadrant (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: the graph of y = -3x + 2 does not pass through the third quadrant -/
theorem not_passes_third_quadrant :
  ¬ passesThroughQuadrant { m := -3, b := 2 } Quadrant.third := by
  sorry

end not_passes_third_quadrant_l3308_330840


namespace friday_texts_l3308_330838

/-- Represents the number of texts sent to each friend on a given day -/
structure DailyTexts where
  allison : ℕ
  brittney : ℕ
  carol : ℕ
  dylan : ℕ

/-- Calculates the total number of texts sent in a day -/
def totalTexts (d : DailyTexts) : ℕ := d.allison + d.brittney + d.carol + d.dylan

/-- Sydney's texting schedule from Monday to Thursday -/
def textSchedule : List DailyTexts := [
  ⟨5, 5, 5, 5⟩,        -- Monday
  ⟨15, 10, 12, 8⟩,     -- Tuesday
  ⟨20, 18, 7, 14⟩,     -- Wednesday
  ⟨0, 25, 10, 5⟩       -- Thursday
]

/-- Cost of a single text in cents -/
def textCost : ℕ := 10

/-- Weekly budget in cents -/
def weeklyBudget : ℕ := 2000

/-- Theorem: Sydney can send 36 texts on Friday given her schedule and budget -/
theorem friday_texts : 
  (weeklyBudget - (textSchedule.map totalTexts).sum * textCost) / textCost = 36 := by
  sorry

end friday_texts_l3308_330838


namespace triangle_areas_sum_l3308_330845

/-- Given a rectangle and two triangles with specific properties, prove that the combined area of the triangles is 108 cm² -/
theorem triangle_areas_sum (rectangle_length rectangle_width : ℝ)
  (triangle1_area_factor : ℝ)
  (triangle2_base triangle2_base_height_sum : ℝ)
  (h_rectangle_length : rectangle_length = 6)
  (h_rectangle_width : rectangle_width = 4)
  (h_rectangle_triangle1_ratio : (rectangle_length * rectangle_width) / (5 * triangle1_area_factor) = 2 / 5)
  (h_triangle2_base : triangle2_base = 8)
  (h_triangle2_sum : triangle2_base + (triangle2_base_height_sum - triangle2_base) = 20)
  (h_triangle_ratio : (triangle2_base * (triangle2_base_height_sum - triangle2_base)) / (10 * triangle1_area_factor) = 3 / 5) :
  5 * triangle1_area_factor + (triangle2_base * (triangle2_base_height_sum - triangle2_base)) / 2 = 108 := by
  sorry

end triangle_areas_sum_l3308_330845


namespace select_blocks_count_l3308_330821

/-- The number of ways to select 4 blocks from a 6x6 grid with no two in the same row or column -/
def select_blocks : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    with no two in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end select_blocks_count_l3308_330821


namespace lineup_probability_probability_no_more_than_five_girls_between_boys_l3308_330839

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem lineup_probability :
  (C 14 9 + 6 * C 13 8) / C 20 9 =
  (↑(C 14 9 + 6 * C 13 8) : ℚ) / (C 20 9 : ℚ) :=
by sorry

theorem probability_no_more_than_five_girls_between_boys :
  (↑(C 14 9 + 6 * C 13 8) : ℚ) / (C 20 9 : ℚ) =
  9724 / 167960 :=
by sorry

end lineup_probability_probability_no_more_than_five_girls_between_boys_l3308_330839


namespace min_value_expression_min_value_achieved_l3308_330895

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + x * y * z ≥ 2 :=
by sorry

theorem min_value_achieved (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + x * y * z = 2 ↔ 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end min_value_expression_min_value_achieved_l3308_330895


namespace twice_total_credits_l3308_330805

/-- Given the high school credits of three students (Aria, Emily, and Spencer),
    where Emily has 20 credits, Aria has twice as many credits as Emily,
    and Emily has twice as many credits as Spencer,
    prove that twice the total number of credits for all three is 140. -/
theorem twice_total_credits (emily_credits : ℕ) 
  (h1 : emily_credits = 20)
  (h2 : ∃ aria_credits : ℕ, aria_credits = 2 * emily_credits)
  (h3 : ∃ spencer_credits : ℕ, emily_credits = 2 * spencer_credits) :
  2 * (emily_credits + 2 * emily_credits + emily_credits / 2) = 140 :=
by sorry

end twice_total_credits_l3308_330805


namespace remaining_apples_l3308_330849

def initial_apples : ℕ := 127
def given_apples : ℕ := 88

theorem remaining_apples : initial_apples - given_apples = 39 := by
  sorry

end remaining_apples_l3308_330849


namespace sum_of_roots_l3308_330808

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x

-- State the theorem
theorem sum_of_roots (h k : ℝ) (h_root : p h = 1) (k_root : p k = 5) : h + k = 2 := by
  sorry

end sum_of_roots_l3308_330808


namespace regular_polygon_sides_l3308_330828

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) / n = 144 → n = 10 := by
  sorry

end regular_polygon_sides_l3308_330828


namespace f_monotone_and_inequality_l3308_330866

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem f_monotone_and_inequality (a : ℝ) :
  (a > 0 ∧ a ≤ 2) ↔
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x > 0 → (x - 1) * f a x ≥ 0) :=
sorry

end f_monotone_and_inequality_l3308_330866


namespace power_of_power_l3308_330857

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l3308_330857


namespace simplify_algebraic_expression_l3308_330865

theorem simplify_algebraic_expression (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := by
  sorry

end simplify_algebraic_expression_l3308_330865


namespace attendance_theorem_l3308_330859

/-- Represents the admission prices and attendance for a play -/
structure PlayAttendance where
  adult_price : ℕ
  child_price : ℕ
  total_receipts : ℕ
  num_children : ℕ

/-- Calculates the total number of attendees given the play attendance data -/
def total_attendees (p : PlayAttendance) : ℕ :=
  p.num_children + (p.total_receipts - p.num_children * p.child_price) / p.adult_price

/-- Theorem stating that given the specific conditions, the total number of attendees is 610 -/
theorem attendance_theorem (p : PlayAttendance) 
    (h1 : p.adult_price = 2)
    (h2 : p.child_price = 1)
    (h3 : p.total_receipts = 960)
    (h4 : p.num_children = 260) : 
  total_attendees p = 610 := by
  sorry

#eval total_attendees ⟨2, 1, 960, 260⟩

end attendance_theorem_l3308_330859


namespace unique_solution_sum_l3308_330832

theorem unique_solution_sum (x : ℝ) (a b c : ℕ+) : 
  x = Real.sqrt ((Real.sqrt 65) / 2 + 5 / 2) →
  x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 →
  ∃! (a b c : ℕ+), x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 →
  a + b + c = 105 := by
sorry

end unique_solution_sum_l3308_330832


namespace triangle_equilateral_iff_area_condition_l3308_330871

/-- Triangle with vertices A₁, A₂, A₃ -/
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

/-- Altitude of a triangle from a vertex to the opposite side -/
def altitude (T : Triangle) (i : Fin 3) : ℝ := sorry

/-- Area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Length of a side of a triangle -/
def sideLength (T : Triangle) (i j : Fin 3) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def isEquilateral (T : Triangle) : Prop :=
  sideLength T 0 1 = sideLength T 1 2 ∧ sideLength T 1 2 = sideLength T 2 0

/-- Main theorem: A triangle is equilateral iff its area satisfies the given condition -/
theorem triangle_equilateral_iff_area_condition (T : Triangle) :
  isEquilateral T ↔ 
    area T = (1/6) * (sideLength T 0 1 * altitude T 0 + 
                      sideLength T 1 2 * altitude T 1 + 
                      sideLength T 2 0 * altitude T 2) :=
sorry

end triangle_equilateral_iff_area_condition_l3308_330871


namespace minimize_theta_l3308_330891

def angle : ℝ := -495

theorem minimize_theta : 
  ∃ (K : ℤ) (θ : ℝ), 
    angle = K * 360 + θ ∧ 
    ∀ (K' : ℤ) (θ' : ℝ), angle = K' * 360 + θ' → |θ| ≤ |θ'| ∧
    θ = -135 := by
  sorry

end minimize_theta_l3308_330891


namespace angle_complement_theorem_l3308_330816

theorem angle_complement_theorem (x : ℝ) : 
  (90 - x) = (3 * x + 10) → x = 20 := by
  sorry

end angle_complement_theorem_l3308_330816


namespace infinite_sum_equals_9_320_l3308_330889

/-- The sum of the infinite series n / (n^4 + 16) from n=1 to infinity equals 9/320 -/
theorem infinite_sum_equals_9_320 :
  (∑' n : ℕ, n / (n^4 + 16 : ℝ)) = 9 / 320 := by
  sorry

end infinite_sum_equals_9_320_l3308_330889


namespace expression_factorization_l3308_330862

theorem expression_factorization (x : ℝ) : 
  (16 * x^6 - 36 * x^4) - (4 * x^6 - 9 * x^4 + 12) = 3 * x^4 * (2 * x + 3) * (2 * x - 3) - 12 := by
  sorry

end expression_factorization_l3308_330862


namespace phi_range_for_monotonic_interval_l3308_330813

/-- Given a function f(x) = -2 sin(2x + φ) where |φ| < π, 
    if (π/5, 5π/8) is a monotonically increasing interval of f(x),
    then π/10 ≤ φ ≤ π/4 -/
theorem phi_range_for_monotonic_interval (φ : Real) :
  (|φ| < π) →
  (∀ x₁ x₂, π/5 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5*π/8 → 
    (-2 * Real.sin (2*x₁ + φ)) < (-2 * Real.sin (2*x₂ + φ))) →
  π/10 ≤ φ ∧ φ ≤ π/4 := by
  sorry

end phi_range_for_monotonic_interval_l3308_330813


namespace lines_coplanar_iff_k_values_l3308_330814

-- Define the two lines
def line1 (t k : ℝ) : ℝ × ℝ × ℝ := (1 + t, 2 + 2*t, 3 - k*t)
def line2 (u k : ℝ) : ℝ × ℝ × ℝ := (2 + u, 5 + k*u, 6 + u)

-- Define the condition for the lines to be coplanar
def are_coplanar (k : ℝ) : Prop :=
  ∃ t u, line1 t k = line2 u k

-- State the theorem
theorem lines_coplanar_iff_k_values :
  ∀ k : ℝ, are_coplanar k ↔ (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
by sorry

end lines_coplanar_iff_k_values_l3308_330814


namespace savings_calculation_l3308_330876

/-- Calculates the total savings of Thomas and Joseph after 6 years -/
def total_savings (thomas_monthly_savings : ℚ) (years : ℕ) : ℚ :=
  let months : ℕ := years * 12
  let thomas_total : ℚ := thomas_monthly_savings * months
  let joseph_monthly_savings : ℚ := thomas_monthly_savings - (2 / 5) * thomas_monthly_savings
  let joseph_total : ℚ := joseph_monthly_savings * months
  thomas_total + joseph_total

/-- Proves that Thomas and Joseph's combined savings after 6 years equals $4608 -/
theorem savings_calculation : total_savings 40 6 = 4608 := by
  sorry

end savings_calculation_l3308_330876


namespace three_digit_divisible_by_17_l3308_330827

theorem three_digit_divisible_by_17 : 
  (Finset.filter (fun k : ℕ => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 :=
by sorry

end three_digit_divisible_by_17_l3308_330827


namespace sqrt_meaningful_iff_x_geq_one_fifth_l3308_330851

theorem sqrt_meaningful_iff_x_geq_one_fifth (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 5 * x - 1) ↔ x ≥ 1 / 5 := by
  sorry

end sqrt_meaningful_iff_x_geq_one_fifth_l3308_330851


namespace arithmetic_sequence_sum_l3308_330860

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → a (i + 1) - a i = a (j + 1) - a j

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a n →
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a (n - 1) = 38 →
  a (n - 2) + a (n - 3) = 61 := by
  sorry

end arithmetic_sequence_sum_l3308_330860


namespace largest_of_twenty_consecutive_even_integers_with_sum_3000_l3308_330803

/-- Represents a sequence of consecutive even integers -/
structure ConsecutiveEvenIntegers where
  start : ℤ
  count : ℕ
  is_even : Even start

/-- The sum of the sequence -/
def sum_sequence (seq : ConsecutiveEvenIntegers) : ℤ :=
  seq.count * (2 * seq.start + (seq.count - 1) * 2) / 2

/-- The largest integer in the sequence -/
def largest_integer (seq : ConsecutiveEvenIntegers) : ℤ :=
  seq.start + 2 * (seq.count - 1)

theorem largest_of_twenty_consecutive_even_integers_with_sum_3000 :
  ∀ seq : ConsecutiveEvenIntegers,
    seq.count = 20 →
    sum_sequence seq = 3000 →
    largest_integer seq = 169 := by
  sorry

end largest_of_twenty_consecutive_even_integers_with_sum_3000_l3308_330803


namespace min_pouches_is_sixty_l3308_330837

/-- Represents the number of gold coins Flint has. -/
def total_coins : ℕ := 60

/-- Represents the possible number of sailors among whom the coins might be distributed. -/
def possible_sailors : List ℕ := [2, 3, 4, 5]

/-- Defines a valid distribution as one where each sailor receives an equal number of coins. -/
def is_valid_distribution (num_pouches : ℕ) : Prop :=
  ∀ n ∈ possible_sailors, (total_coins / num_pouches) * n = total_coins

/-- States that the number of pouches is minimal if no smaller number satisfies the distribution criteria. -/
def is_minimal (num_pouches : ℕ) : Prop :=
  is_valid_distribution num_pouches ∧
  ∀ k < num_pouches, ¬is_valid_distribution k

/-- The main theorem stating that 60 is the minimum number of pouches required for valid distribution. -/
theorem min_pouches_is_sixty :
  is_minimal total_coins :=
sorry

end min_pouches_is_sixty_l3308_330837


namespace max_sqrt_sum_l3308_330810

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 36) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ Real.sqrt 261 := by
  sorry

end max_sqrt_sum_l3308_330810


namespace equation_solution_l3308_330830

theorem equation_solution : ∃! x : ℝ, (64 : ℝ)^(x - 1) / (4 : ℝ)^(x - 1) = (256 : ℝ)^(2*x) ∧ x = -1/3 := by
  sorry

end equation_solution_l3308_330830


namespace min_jumps_to_cover_race_l3308_330850

/-- Represents the possible jump distances of the cricket -/
inductive JumpDistance where
  | short : JumpDistance -- 8 meters
  | long : JumpDistance  -- 9 meters

/-- The race distance in meters -/
def raceDistance : ℕ := 100

/-- Calculates the total distance covered by a sequence of jumps -/
def totalDistance (jumps : List JumpDistance) : ℕ :=
  jumps.foldl (fun acc jump => acc + match jump with
    | JumpDistance.short => 8
    | JumpDistance.long => 9) 0

/-- Checks if a sequence of jumps exactly covers the race distance -/
def isValidJumpSequence (jumps : List JumpDistance) : Prop :=
  totalDistance jumps = raceDistance

/-- The main theorem to be proved -/
theorem min_jumps_to_cover_race :
  ∃ (jumps : List JumpDistance),
    isValidJumpSequence jumps ∧
    jumps.length = 12 ∧
    ∀ (other_jumps : List JumpDistance),
      isValidJumpSequence other_jumps →
      other_jumps.length ≥ 12 :=
by sorry

end min_jumps_to_cover_race_l3308_330850


namespace ellipse_max_distance_sum_l3308_330870

/-- Given an ellipse with equation x^2/4 + y^2/3 = 1 and foci F₁ and F₂,
    where a line l passing through F₁ intersects the ellipse at points A and B,
    the maximum value of |BF₂| + |AF₂| is 5. -/
theorem ellipse_max_distance_sum (F₁ F₂ A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  (∀ x y, x^2/4 + y^2/3 = 1 → (x, y) ∈ l → (x, y) = A ∨ (x, y) = B) →
  F₁ ∈ l →
  F₁.1 < F₂.1 →
  (∀ x y, x^2/4 + y^2/3 = 1 → dist (x, y) F₁ + dist (x, y) F₂ = 4) →
  dist B F₂ + dist A F₂ ≤ 5 :=
sorry


end ellipse_max_distance_sum_l3308_330870


namespace max_sum_hexagonal_prism_with_pyramid_l3308_330852

/-- Represents a three-dimensional geometric shape -/
structure Shape3D where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- A hexagonal prism -/
def hexagonal_prism : Shape3D :=
  { faces := 8, vertices := 12, edges := 18 }

/-- Adds a pyramid to one face of a given shape -/
def add_pyramid (s : Shape3D) : Shape3D :=
  { faces := s.faces + 5,  -- Loses 1 face, gains 6
    vertices := s.vertices + 1,
    edges := s.edges + 6 }

/-- Calculates the sum of faces, vertices, and edges -/
def shape_sum (s : Shape3D) : ℕ :=
  s.faces + s.vertices + s.edges

/-- Theorem: The maximum sum of faces, vertices, and edges after adding a pyramid to a hexagonal prism is 44 -/
theorem max_sum_hexagonal_prism_with_pyramid :
  shape_sum (add_pyramid hexagonal_prism) = 44 := by
  sorry

end max_sum_hexagonal_prism_with_pyramid_l3308_330852


namespace intersection_point_l3308_330848

theorem intersection_point (a : ℝ) :
  (∃! p : ℝ × ℝ, (p.2 = a * p.1 + a ∧ p.2 = p.1 ∧ p.2 = 2 - 2 * a * p.1)) ↔ (a = 1/2 ∨ a = -2) :=
by sorry

end intersection_point_l3308_330848


namespace jake_weight_loss_l3308_330825

/-- Jake needs to lose weight to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
  (h1 : total_weight = 153)
  (h2 : jake_weight = 113)
  (h3 : total_weight = sister_weight + jake_weight) :
  jake_weight - 2 * sister_weight = 33 := by
sorry

end jake_weight_loss_l3308_330825


namespace three_digit_number_operation_l3308_330867

theorem three_digit_number_operation (a b c : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 →  -- Ensures it's a three-digit number
  a = 2*c - 3 →  -- Hundreds digit is 3 less than twice the units digit
  ((100*a + 10*b + c) - ((100*c + 10*b + a) + 50)) % 10 = 3 :=
by sorry

end three_digit_number_operation_l3308_330867


namespace paul_failed_by_10_marks_l3308_330873

/-- Calculates the number of marks a student failed by in an exam -/
def marksFailed (maxMarks passingPercentage gotMarks : ℕ) : ℕ :=
  let passingMarks := (passingPercentage * maxMarks) / 100
  if gotMarks ≥ passingMarks then 0 else passingMarks - gotMarks

/-- Theorem stating that Paul failed by 10 marks -/
theorem paul_failed_by_10_marks :
  marksFailed 120 50 50 = 10 := by
  sorry

end paul_failed_by_10_marks_l3308_330873


namespace min_x_squared_isosceles_trapezoid_l3308_330879

/-- Represents a trapezoid ABCD with specific properties -/
structure IsoscelesTrapezoid where
  -- Length of base AB
  ab : ℝ
  -- Length of base CD
  cd : ℝ
  -- Length of side AD (equal to BC)
  x : ℝ
  -- Ensures the trapezoid is isosceles
  isIsosceles : ad = bc
  -- Ensures a circle with center on AB is tangent to AD and BC
  hasTangentCircle : ∃ (center : ℝ), 0 ≤ center ∧ center ≤ ab ∧
    ∃ (radius : ℝ), radius > 0 ∧
    (center - radius)^2 + x^2 = (ab/2)^2 ∧
    (center + radius)^2 + x^2 = (ab/2)^2

/-- The theorem stating the minimum value of x^2 for the given trapezoid -/
theorem min_x_squared_isosceles_trapezoid (t : IsoscelesTrapezoid)
  (h1 : t.ab = 50)
  (h2 : t.cd = 14) :
  ∃ (m : ℝ), m^2 = 800 ∧ ∀ (y : ℝ), t.x = y → y^2 ≥ m^2 := by
  sorry

end min_x_squared_isosceles_trapezoid_l3308_330879


namespace series_sum_equals_399002_l3308_330893

/-- The sum of the series 1-2-3+4+5-6-7+8+9-10-11+12+13-...-1994-1995+1996+1997 -/
def seriesSum : ℕ → ℤ
  | 0 => 0
  | n + 1 => seriesSum n + term (n + 1)
where
  term : ℕ → ℤ
  | n => if n % 5 ≤ 2 then -(n : ℤ) else (n : ℤ)

theorem series_sum_equals_399002 : seriesSum 1997 = 399002 := by
  sorry

end series_sum_equals_399002_l3308_330893


namespace journey_ratio_l3308_330836

/-- Proves the ratio of distance after storm to total journey distance -/
theorem journey_ratio (speed : ℝ) (time : ℝ) (storm_distance : ℝ) : 
  speed = 30 ∧ time = 20 ∧ storm_distance = 200 →
  (speed * time - storm_distance) / (2 * speed * time) = 1 / 3 := by
  sorry

end journey_ratio_l3308_330836


namespace area_of_triangle_APQ_l3308_330833

/-- Two perpendicular lines intersecting at A(9,12) with y-intercepts P and Q -/
structure PerpendicularLines where
  A : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular : Bool
  intersect_at_A : Bool
  y_intercept_diff : ℝ

/-- The specific configuration of perpendicular lines for our problem -/
def problem_lines : PerpendicularLines where
  A := (9, 12)
  P := (0, 0)
  Q := (0, 6)
  perpendicular := true
  intersect_at_A := true
  y_intercept_diff := 6

/-- The area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle APQ is 27 -/
theorem area_of_triangle_APQ : 
  triangle_area problem_lines.A problem_lines.P problem_lines.Q = 27 := by sorry

end area_of_triangle_APQ_l3308_330833


namespace total_cows_is_570_l3308_330883

/-- The number of cows owned by Matthews -/
def matthews_cows : ℕ := 60

/-- The number of cows owned by Aaron -/
def aaron_cows : ℕ := 4 * matthews_cows

/-- The number of cows owned by Marovich -/
def marovich_cows : ℕ := aaron_cows + matthews_cows - 30

/-- The total number of cows owned by all three -/
def total_cows : ℕ := aaron_cows + matthews_cows + marovich_cows

theorem total_cows_is_570 : total_cows = 570 := by
  sorry

end total_cows_is_570_l3308_330883


namespace neighboring_cells_difference_l3308_330846

/-- A type representing a cell in an n × n grid --/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- A function that assigns values to cells in the grid --/
def GridAssignment (n : ℕ) := Cell n → Fin (n^2)

/-- Two cells are neighbors if they share at least one point --/
def IsNeighbor {n : ℕ} (c1 c2 : Cell n) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col.val = c2.col.val + 1)

/-- The main theorem to be proved --/
theorem neighboring_cells_difference {n : ℕ} (h : n > 1) (g : GridAssignment n) :
  ∃ (c1 c2 : Cell n), IsNeighbor c1 c2 ∧ 
    (g c1).val ≥ (g c2).val + n + 1 ∨ (g c2).val ≥ (g c1).val + n + 1 :=
sorry

end neighboring_cells_difference_l3308_330846


namespace james_candy_packs_l3308_330880

/-- Given the initial amount, change received, and cost per pack of candy,
    calculate the number of packs of candy bought. -/
def candyPacks (initialAmount change costPerPack : ℕ) : ℕ :=
  (initialAmount - change) / costPerPack

/-- Theorem stating that James bought 3 packs of candy -/
theorem james_candy_packs :
  candyPacks 20 11 3 = 3 := by
  sorry

end james_candy_packs_l3308_330880


namespace perpendicular_tangents_ratio_l3308_330823

-- Define the line ax - by - 2 = 0
def line (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y - 2 = 0

-- Define the curve y = x^3
def curve (x y : ℝ) : Prop := y = x^3

-- Define the point P(1, 1)
def point_P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line to the curve at P
def tangent_slope_curve : ℝ := 3

-- Define the condition that the tangent lines are mutually perpendicular
def perpendicular_tangents (a b : ℝ) : Prop :=
  (a / b) * tangent_slope_curve = -1

theorem perpendicular_tangents_ratio (a b : ℝ) :
  line a b point_P.1 point_P.2 →
  curve point_P.1 point_P.2 →
  perpendicular_tangents a b →
  a / b = -1 / 3 := by
  sorry

end perpendicular_tangents_ratio_l3308_330823


namespace swamp_ecosystem_flies_eaten_l3308_330872

/-- Represents the number of flies eaten per day in a swamp ecosystem -/
def flies_eaten_per_day (
  frog_flies : ℕ)  -- flies eaten by one frog per day
  (fish_frogs : ℕ)  -- frogs eaten by one fish per day
  (gharial_fish : ℕ)  -- fish eaten by one gharial per day
  (heron_frogs : ℕ)  -- frogs eaten by one heron per day
  (heron_fish : ℕ)  -- fish eaten by one heron per day
  (caiman_gharials : ℕ)  -- gharials eaten by one caiman per day
  (caiman_herons : ℕ)  -- herons eaten by one caiman per day
  (num_gharials : ℕ)  -- number of gharials in the swamp
  (num_herons : ℕ)  -- number of herons in the swamp
  (num_caimans : ℕ)  -- number of caimans in the swamp
  : ℕ :=
  sorry

/-- Theorem stating the number of flies eaten per day in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_per_day 30 8 15 5 3 2 2 9 12 7 = 42840 :=
by sorry

end swamp_ecosystem_flies_eaten_l3308_330872


namespace imaginary_part_of_z_l3308_330868

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (3 - 4 * i) / i
  Complex.im z = -3 := by sorry

end imaginary_part_of_z_l3308_330868


namespace arithmetic_sequence_property_l3308_330856

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 9 - (1/3) * a 11 = 16 := by
  sorry

end arithmetic_sequence_property_l3308_330856


namespace problem_solution_l3308_330869

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2^(x / c^2) + 1
  else 0

theorem problem_solution (c : ℝ) :
  (0 < c ∧ c < 1) →
  (f c (c^2) = 9/8) →
  (c = 1/2) ∧
  (∀ x : ℝ, f (1/2) x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 1) :=
by sorry

end problem_solution_l3308_330869


namespace sum_and_double_l3308_330818

theorem sum_and_double : 2 * (2/20 + 3/30 + 4/40) = 0.6 := by
  sorry

end sum_and_double_l3308_330818


namespace rent_utilities_percentage_l3308_330806

-- Define the previous monthly income
def previous_income : ℝ := 1000

-- Define the salary increase
def salary_increase : ℝ := 600

-- Define the percentage spent on rent and utilities after the increase
def new_percentage : ℝ := 0.25

-- Define the function to calculate the amount spent on rent and utilities
def rent_utilities (income : ℝ) (percentage : ℝ) : ℝ := income * percentage

-- Theorem statement
theorem rent_utilities_percentage :
  ∃ (old_percentage : ℝ),
    rent_utilities previous_income old_percentage = 
    rent_utilities (previous_income + salary_increase) new_percentage ∧
    old_percentage = 0.4 :=
by sorry

end rent_utilities_percentage_l3308_330806


namespace sin_cos_sum_21_39_l3308_330847

theorem sin_cos_sum_21_39 : 
  Real.sin (21 * π / 180) * Real.cos (39 * π / 180) + 
  Real.cos (21 * π / 180) * Real.sin (39 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_cos_sum_21_39_l3308_330847


namespace third_team_pieces_l3308_330815

theorem third_team_pieces (total : ℕ) (first_team : ℕ) (second_team : ℕ) 
  (h1 : total = 500) 
  (h2 : first_team = 189) 
  (h3 : second_team = 131) : 
  total - (first_team + second_team) = 180 :=
by
  sorry

end third_team_pieces_l3308_330815


namespace fraction_of_fraction_two_ninths_of_three_fourths_l3308_330882

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem two_ninths_of_three_fourths :
  (2 : ℚ) / 9 / ((3 : ℚ) / 4) = 8 / 27 := by sorry

end fraction_of_fraction_two_ninths_of_three_fourths_l3308_330882


namespace direction_vector_b_value_l3308_330863

/-- Given a line passing through points (-1, 3) and (2, 7) with direction vector (2, b), prove that b = 8/3 -/
theorem direction_vector_b_value (b : ℚ) : 
  let p1 : ℚ × ℚ := (-1, 3)
  let p2 : ℚ × ℚ := (2, 7)
  let direction_vector : ℚ × ℚ := (2, b)
  (∃ (k : ℚ), k • (p2.1 - p1.1, p2.2 - p1.2) = direction_vector) →
  b = 8/3 := by
sorry

end direction_vector_b_value_l3308_330863


namespace f_inequality_range_l3308_330843

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : 
  f (2 * x) > f (x - 1) ↔ x < -1 ∨ x > 1/3 :=
sorry

end f_inequality_range_l3308_330843


namespace tom_program_duration_l3308_330819

def combined_program_duration (bs_duration ph_d_duration : ℕ) : ℕ :=
  bs_duration + ph_d_duration

def accelerated_duration (total_duration : ℕ) (acceleration_factor : ℚ) : ℚ :=
  (total_duration : ℚ) * acceleration_factor

theorem tom_program_duration :
  let bs_duration : ℕ := 3
  let ph_d_duration : ℕ := 5
  let acceleration_factor : ℚ := 3 / 4
  let total_duration := combined_program_duration bs_duration ph_d_duration
  accelerated_duration total_duration acceleration_factor = 6 := by
  sorry

end tom_program_duration_l3308_330819


namespace number_operations_l3308_330885

theorem number_operations (x : ℚ) : (x - 5) / 7 = 7 → (x - 2) / 13 = 4 := by
  sorry

end number_operations_l3308_330885


namespace sum_of_products_of_roots_l3308_330890

theorem sum_of_products_of_roots (p q r : ℂ) : 
  (2 * p^3 + p^2 - 7*p + 2 = 0) → 
  (2 * q^3 + q^2 - 7*q + 2 = 0) → 
  (2 * r^3 + r^2 - 7*r + 2 = 0) → 
  p * q + q * r + r * p = -7/2 := by
  sorry

end sum_of_products_of_roots_l3308_330890


namespace chloe_first_round_score_l3308_330831

/-- Chloe's trivia game score calculation -/
theorem chloe_first_round_score (first_round : ℤ) 
  (h1 : first_round + 50 - 4 = 86) : first_round = 40 := by
  sorry

end chloe_first_round_score_l3308_330831


namespace units_digit_of_k_squared_plus_two_to_k_l3308_330898

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : 
  k = 2008^2 + 2^2008 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l3308_330898


namespace square_nonnegative_is_universal_l3308_330887

/-- The proposition "The square of any real number is non-negative" -/
def square_nonnegative_prop : Prop := ∀ x : ℝ, x^2 ≥ 0

/-- Definition of a universal proposition -/
def is_universal_prop (P : Prop) : Prop := ∃ (α : Type) (Q : α → Prop), P = ∀ x : α, Q x

/-- The square_nonnegative_prop is a universal proposition -/
theorem square_nonnegative_is_universal : is_universal_prop square_nonnegative_prop := by sorry

end square_nonnegative_is_universal_l3308_330887


namespace largest_common_divisor_l3308_330896

theorem largest_common_divisor : 
  ∃ (n : ℕ), n = 35 ∧ 
  n ∣ 420 ∧ n ∣ 385 ∧ 
  ∀ (m : ℕ), m ∣ 420 ∧ m ∣ 385 → m ≤ n :=
by sorry

end largest_common_divisor_l3308_330896


namespace corn_field_fraction_theorem_l3308_330802

/-- Represents a trapezoid field -/
structure TrapezoidField where
  short_side : ℝ
  long_side : ℝ
  angle : ℝ

/-- The fraction of a trapezoid field's area that is closer to its longest side -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

theorem corn_field_fraction_theorem (field : TrapezoidField) 
  (h1 : field.short_side = 120)
  (h2 : field.long_side = 240)
  (h3 : field.angle = 60) :
  fraction_closest_to_longest_side field = 1/3 := by
  sorry

end corn_field_fraction_theorem_l3308_330802


namespace product_and_sum_of_three_two_digit_integers_l3308_330809

theorem product_and_sum_of_three_two_digit_integers : ∃ (a b c : ℕ), 
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  10 ≤ c ∧ c < 100 ∧
  a * b * c = 636405 ∧
  a + b + c = 259 := by
sorry

end product_and_sum_of_three_two_digit_integers_l3308_330809


namespace stairs_ratio_l3308_330877

theorem stairs_ratio (samir veronica : ℕ) (total : ℕ) (h1 : samir = 318) (h2 : total = 495) (h3 : samir + veronica = total) :
  (veronica : ℚ) / (samir / 2 : ℚ) = (total - samir : ℚ) / (samir / 2 : ℚ) := by
sorry

end stairs_ratio_l3308_330877


namespace sector_central_angle_l3308_330841

/-- Given a sector with circumference 10 and area 4, prove that its central angle is π/2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : (1/2) * l * r = 4) :
  l / r = 1/2 := by sorry

end sector_central_angle_l3308_330841


namespace initial_speed_calculation_l3308_330812

/-- Represents a baseball player's training progress -/
structure BaseballTraining where
  initialSpeed : ℝ
  trainingWeeks : ℕ
  speedGainPerWeek : ℝ
  finalSpeedIncrease : ℝ

/-- Theorem stating the initial speed of a baseball player given their training progress -/
theorem initial_speed_calculation (training : BaseballTraining)
  (h1 : training.trainingWeeks = 16)
  (h2 : training.speedGainPerWeek = 1)
  (h3 : training.finalSpeedIncrease = 0.2)
  : training.initialSpeed = 80 := by
  sorry

end initial_speed_calculation_l3308_330812


namespace largest_multiple_of_9_less_than_100_l3308_330829

theorem largest_multiple_of_9_less_than_100 : 
  ∀ n : ℕ, n * 9 < 100 → n * 9 ≤ 99 :=
by
  sorry

end largest_multiple_of_9_less_than_100_l3308_330829


namespace last_two_digits_of_1032_power_1032_l3308_330878

theorem last_two_digits_of_1032_power_1032 : ∃ k : ℕ, 1032^1032 ≡ 76 [ZMOD 100] := by
  sorry

end last_two_digits_of_1032_power_1032_l3308_330878


namespace shaded_triangle_probability_l3308_330817

/-- The total number of triangles in the diagram -/
def total_triangles : ℕ := 10

/-- The number of shaded or partially shaded triangles -/
def shaded_triangles : ℕ := 3

/-- Each triangle has an equal probability of being selected -/
axiom equal_probability : True

/-- The probability of selecting a shaded or partially shaded triangle -/
def shaded_probability : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : 
  shaded_probability = 3 / 10 := by sorry

end shaded_triangle_probability_l3308_330817


namespace mn_minus_n_value_l3308_330854

theorem mn_minus_n_value (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 5/2) (h3 : m * n < 0) : 
  m * n - n = -7.5 ∨ m * n - n = -12.5 := by
sorry

end mn_minus_n_value_l3308_330854


namespace geoffrey_game_cost_l3308_330822

theorem geoffrey_game_cost (initial_money : ℕ) : 
  initial_money + 20 + 25 + 30 = 125 → 
  ∃ (game_cost : ℕ), 
    game_cost * 3 = 125 - 20 ∧ 
    game_cost = 35 := by
  sorry

end geoffrey_game_cost_l3308_330822


namespace min_value_quadratic_form_l3308_330899

theorem min_value_quadratic_form (x y : ℝ) :
  x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ (x^2 + 2*x*y + 2*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end min_value_quadratic_form_l3308_330899


namespace quadratic_equations_solutions_l3308_330801

theorem quadratic_equations_solutions : ∃ (s1 s2 : Set ℝ),
  (∀ x : ℝ, x ∈ s1 ↔ 3 * x^2 = 6 * x) ∧
  (∀ x : ℝ, x ∈ s2 ↔ x^2 - 6 * x + 5 = 0) ∧
  s1 = {0, 2} ∧
  s2 = {5, 1} := by
sorry


end quadratic_equations_solutions_l3308_330801


namespace puzzle_solving_time_l3308_330875

/-- The total time spent solving puzzles given a warm-up puzzle and two longer puzzles -/
theorem puzzle_solving_time (warm_up_time : ℕ) (num_long_puzzles : ℕ) (long_puzzle_factor : ℕ) : 
  warm_up_time = 10 → 
  num_long_puzzles = 2 → 
  long_puzzle_factor = 3 → 
  warm_up_time + num_long_puzzles * (long_puzzle_factor * warm_up_time) = 70 :=
by sorry

end puzzle_solving_time_l3308_330875


namespace fractional_equation_simplification_l3308_330842

theorem fractional_equation_simplification (x : ℝ) :
  (x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)) ↔ (x - 3 * (2 * x - 1) = -2) :=
by sorry

end fractional_equation_simplification_l3308_330842


namespace inequality_solution_set_l3308_330826

theorem inequality_solution_set : 
  ∀ x : ℝ, -x^2 + 3*x + 4 > 0 ↔ -1 < x ∧ x < 4 := by sorry

end inequality_solution_set_l3308_330826


namespace geometric_sequence_middle_term_l3308_330820

theorem geometric_sequence_middle_term (b : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 30 * r = b ∧ b * r = 9/4) → b = 3 * Real.sqrt 30 := by
  sorry

end geometric_sequence_middle_term_l3308_330820


namespace bianca_birthday_money_l3308_330800

theorem bianca_birthday_money (total_amount : ℕ) (num_friends : ℕ) (amount_per_friend : ℕ) 
  (h1 : total_amount = 30) 
  (h2 : num_friends = 5) 
  (h3 : total_amount = num_friends * amount_per_friend) : 
  amount_per_friend = 6 := by
  sorry

end bianca_birthday_money_l3308_330800


namespace plan_D_most_reasonable_l3308_330804

/-- Represents a survey plan for testing vision of junior high school students -/
inductive SurveyPlan
| A  : SurveyPlan  -- Test students in a certain middle school
| B  : SurveyPlan  -- Test all students in a certain district
| C  : SurveyPlan  -- Test all students in the entire city
| D  : SurveyPlan  -- Select 5 schools from each district and test their students

/-- Represents a city with districts and schools -/
structure City where
  numDistricts : Nat
  numSchoolsPerDistrict : Nat

/-- Determines if a survey plan is reasonable based on representativeness and practicality -/
def isReasonable (plan : SurveyPlan) (city : City) : Prop :=
  match plan with
  | SurveyPlan.D => city.numDistricts = 9 ∧ city.numSchoolsPerDistrict ≥ 5
  | _ => False

/-- Theorem stating that plan D is the most reasonable for a city with 9 districts -/
theorem plan_D_most_reasonable (city : City) :
  city.numDistricts = 9 → city.numSchoolsPerDistrict ≥ 5 → 
  ∀ (plan : SurveyPlan), isReasonable plan city → plan = SurveyPlan.D :=
by sorry

end plan_D_most_reasonable_l3308_330804


namespace algebraic_expression_value_l3308_330874

theorem algebraic_expression_value (a b : ℝ) (h : a - b + 3 = 0) :
  2 - 3*a + 3*b = 11 := by
  sorry

end algebraic_expression_value_l3308_330874


namespace total_crayons_l3308_330807

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : 
  crayons_per_child = 5 → num_children = 10 → crayons_per_child * num_children = 50 := by
  sorry

end total_crayons_l3308_330807


namespace right_triangle_shorter_leg_l3308_330888

theorem right_triangle_shorter_leg (a b c m : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0 →  -- Positive lengths
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = 2 * a →  -- One leg is twice the other
  m = 15 →  -- Median to hypotenuse is 15
  m^2 = (c^2) / 4 + (a^2 + b^2) / 4 →  -- Median formula
  a = 6 * Real.sqrt 5 :=
by sorry

end right_triangle_shorter_leg_l3308_330888


namespace surface_area_ratio_of_cubes_l3308_330864

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a / b = 7) :
  (6 * a^2) / (6 * b^2) = 49 := by
  sorry

end surface_area_ratio_of_cubes_l3308_330864


namespace cos_alpha_minus_pi_sixth_eq_zero_l3308_330858

theorem cos_alpha_minus_pi_sixth_eq_zero (α : Real)
  (h1 : 2 * Real.tan α * Real.sin α = 3)
  (h2 : -Real.pi/2 < α)
  (h3 : α < 0) :
  Real.cos (α - Real.pi/6) = 0 := by
  sorry

end cos_alpha_minus_pi_sixth_eq_zero_l3308_330858


namespace remainder_6n_mod_4_l3308_330844

theorem remainder_6n_mod_4 (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 6 * n ≡ 2 [ZMOD 4] := by
  sorry

end remainder_6n_mod_4_l3308_330844


namespace tea_leaves_problem_l3308_330884

theorem tea_leaves_problem (num_plants : ℕ) (initial_leaves : ℕ) (fall_fraction : ℚ) : 
  num_plants = 3 → 
  initial_leaves = 18 → 
  fall_fraction = 1/3 → 
  (num_plants * initial_leaves * (1 - fall_fraction) : ℚ) = 36 := by
  sorry

end tea_leaves_problem_l3308_330884


namespace rectangular_to_polar_l3308_330853

theorem rectangular_to_polar :
  let x : ℝ := 3
  let y : ℝ := 3 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 6 ∧ θ = π / 3 := by
  sorry

end rectangular_to_polar_l3308_330853


namespace orange_pricing_and_purchase_l3308_330835

-- Define variables
variable (x y m : ℝ)

-- Define the theorem
theorem orange_pricing_and_purchase :
  -- Conditions
  (3 * x + 2 * y = 78) →
  (2 * x + 3 * y = 72) →
  (18 * m + 12 * (100 - m) ≤ 1440) →
  (m ≤ 100) →
  -- Conclusions
  (x = 18 ∧ y = 12) ∧
  (∀ n, n ≤ 100 ∧ 18 * n + 12 * (100 - n) ≤ 1440 → n ≤ 40) :=
by sorry

end orange_pricing_and_purchase_l3308_330835
