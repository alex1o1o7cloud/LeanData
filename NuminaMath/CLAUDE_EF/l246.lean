import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_property_l246_24601

theorem cubic_root_property (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 3*x₁ - 1 = 0 →
  x₂^3 - 3*x₂ - 1 = 0 →
  x₃^3 - 3*x₃ - 1 = 0 →
  x₁ < x₂ →
  x₂ < x₃ →
  x₃^2 - x₂^2 = x₃ - x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_property_l246_24601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_range_l246_24660

theorem geometric_sequence_ratio_range (a₁ : ℝ) (q : ℝ) :
  (a₁ > 0) →
  (∀ n : ℕ, n > 0 → (a₁ * (1 - q^(2*n)) / (1 - q)) / (a₁ * (1 - q^n) / (1 - q)) < 5) →
  0 < q ∧ q ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_range_l246_24660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EC_dot_ED_equals_three_l246_24626

/-- Square ABCD with side length 2 and E as midpoint of AB -/
structure SquareABCD where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 ∧
              (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4 ∧
              (C.1 - D.1)^2 + (C.2 - D.2)^2 = 4 ∧
              (D.1 - A.1)^2 + (D.2 - A.2)^2 = 4
  E_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

/-- Theorem: EC · ED = 3 in the given square ABCD -/
theorem EC_dot_ED_equals_three (s : SquareABCD) :
  dot_product (vector s.E s.C) (vector s.E s.D) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_EC_dot_ED_equals_three_l246_24626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gem_value_is_69_l246_24637

/-- Represents a type of gem with its weight and value -/
structure Gem where
  weight : Nat
  value : Nat

/-- The problem setup -/
def gemProblem : Prop :=
  ∃ (gems : List Gem) (maxWeight : Nat),
    gems.length = 3 ∧
    gems = [
      { weight := 3, value := 9 },
      { weight := 6, value := 20 },
      { weight := 2, value := 5 }
    ] ∧
    maxWeight = 21

/-- The maximum value of gems that can be carried -/
def maxGemValue (gems : List Gem) (maxWeight : Nat) : Nat :=
  sorry -- Implementation details omitted

/-- The theorem to prove -/
theorem max_gem_value_is_69 :
  gemProblem → ∃ (gems : List Gem) (maxWeight : Nat), maxGemValue gems maxWeight = 69 :=
by sorry

#check max_gem_value_is_69

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gem_value_is_69_l246_24637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_eq_neg_four_minus_two_i_l246_24688

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function g
noncomputable def g (x : ℂ) : ℂ := (x^5 + 3*x^3 + 2*x) / (x^2 + 2*x + 2)

-- Theorem statement
theorem g_of_i_eq_neg_four_minus_two_i : g i = -4 - 2*i := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_eq_neg_four_minus_two_i_l246_24688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_is_eighteen_l246_24634

-- Define the parameters of the problem
def first_leg_time : ℚ := 2
def first_leg_speed : ℚ := 60
def second_leg_time : ℚ := 3
def second_leg_speed : ℚ := 50
def miles_per_gallon : ℚ := 30
def cost_per_gallon : ℚ := 2

-- Define the function to calculate the total cost
def total_gas_cost : ℚ :=
  let first_leg_distance := first_leg_time * first_leg_speed
  let second_leg_distance := second_leg_time * second_leg_speed
  let total_distance := first_leg_distance + second_leg_distance
  let gallons_used := total_distance / miles_per_gallon
  gallons_used * cost_per_gallon

-- Theorem statement
theorem gas_cost_is_eighteen : total_gas_cost = 18 := by
  -- Unfold the definition of total_gas_cost
  unfold total_gas_cost
  -- Simplify the arithmetic expressions
  simp [first_leg_time, first_leg_speed, second_leg_time, second_leg_speed, miles_per_gallon, cost_per_gallon]
  -- The proof is complete
  rfl

#eval total_gas_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_is_eighteen_l246_24634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l246_24613

def U : Set ℝ := Set.univ

def M : Set ℝ := {a : ℝ | a^2 - 2*a > 0}

theorem complement_of_M_in_U : 
  Mᶜ = Set.Icc (0 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l246_24613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_sum_of_squares_l246_24683

open Real

theorem cosine_power_sum_of_squares :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ),
    (∀ θ : ℝ, (cos θ)^7 = a₁ * cos θ + a₂ * cos (2*θ) + a₃ * cos (3*θ) + 
                         a₄ * cos (4*θ) + a₅ * cos (5*θ) + a₆ * cos (6*θ) + 
                         a₇ * cos (7*θ)) ∧
    a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 + a₆^2 + a₇^2 = 1716 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_sum_of_squares_l246_24683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_is_eight_l246_24687

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℚ
  inningsPlayed : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.initialAverage * (b.inningsPlayed - 1 : ℚ) + b.runsInLastInning) / b.inningsPlayed

/-- Theorem: Given the conditions, prove that the new average is 8 -/
theorem batsman_average_is_eight (b : Batsman) 
    (h1 : b.inningsPlayed = 17)
    (h2 : b.runsInLastInning = 56)
    (h3 : b.averageIncrease = 3)
    (h4 : newAverage b = b.initialAverage + b.averageIncrease) : 
  newAverage b = 8 := by
  sorry

#eval newAverage { initialAverage := 5, inningsPlayed := 17, runsInLastInning := 56, averageIncrease := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_is_eight_l246_24687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l246_24622

noncomputable section

open Real

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a = 36 →  -- Given side length
  c = 64 →  -- Given side length
  C = 2 * A →  -- Given angle relation
  sin C = sin (2 * A) →  -- Consequence of angle relation
  sin A / a = sin B / b →  -- Law of sines
  sin B / b = sin C / c →  -- Law of sines
  b = 52 / 27 :=  -- Conclusion to prove
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l246_24622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_point_zero_one_l246_24663

noncomputable def round_to_hundredths (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

def x : ℝ := 13.165
def y : ℝ := 7.686
def z : ℝ := 11.545

noncomputable def a : ℝ :=
  round_to_hundredths x + round_to_hundredths y + round_to_hundredths z

noncomputable def b : ℝ :=
  round_to_hundredths (x + y + z)

theorem a_minus_b_equals_point_zero_one :
  a - b = 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_equals_point_zero_one_l246_24663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_rate_l246_24620

/-- Calculates the rate of profit given the cost price and selling price -/
noncomputable def rate_of_profit (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 60 is 20% -/
theorem book_profit_rate :
  let cost_price : ℝ := 50
  let selling_price : ℝ := 60
  rate_of_profit cost_price selling_price = 20 := by
  -- Unfold the definition of rate_of_profit
  unfold rate_of_profit
  -- Simplify the expression
  simp
  -- The proof is completed by numerical computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_rate_l246_24620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ahsme_senior_mean_score_l246_24672

/-- The mean score of seniors in the AHSME, given the following conditions:
  * 120 students participated
  * The mean score of all students was 120
  * The number of non-seniors was 80% more than the number of seniors
  * The mean score of seniors was 80% higher than that of non-seniors
-/
theorem ahsme_senior_mean_score :
  ∀ (s n : ℕ) (m_s m_n : ℝ),
  s + n = 120 →
  n = s + (8 * s / 10) →
  (s * m_s + n * m_n) / (s + n) = 120 →
  m_s = m_n + (8 * m_n / 10) →
  ∃ ε > 0, |m_s - 167.87| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ahsme_senior_mean_score_l246_24672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_with_both_colors_l246_24669

theorem percentage_with_both_colors (total_children : ℕ) 
  (h_even : Even total_children)
  (h_blue : (60 : ℚ) / 100 * total_children = (blue_flags : ℕ) / 2)
  (h_red : (70 : ℚ) / 100 * total_children = (red_flags : ℕ) / 2)
  (h_total : blue_flags + red_flags = 2 * total_children) :
  (30 : ℚ) / 100 * total_children = (blue_flags + red_flags - 2 * total_children) / 2 := by
  sorry

#check percentage_with_both_colors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_with_both_colors_l246_24669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_256_l246_24650

theorem modular_inverse_256 (h : (16⁻¹ : ZMod 97) = 10) : (256⁻¹ : ZMod 97) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_256_l246_24650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l246_24649

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + π/2) = 1/3) : Real.cos (2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l246_24649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_theorem_l246_24630

/-- Represents the configuration of a grid with circles on top -/
structure GridWithCircles where
  gridSize : Nat
  squareSize : ℝ
  smallCircleDiameter : ℝ
  largeCircleDiameter : ℝ
  smallCircleCount : Nat
  largeCircleCount : Nat

/-- Calculates the visible shaded area of the grid -/
noncomputable def visibleShadedArea (config : GridWithCircles) : ℝ :=
  let totalArea := (config.gridSize * config.gridSize : ℝ) * config.squareSize ^ 2
  let smallCircleArea := config.smallCircleCount * Real.pi * (config.smallCircleDiameter / 2) ^ 2
  let largeCircleArea := config.largeCircleCount * Real.pi * (config.largeCircleDiameter / 2) ^ 2
  totalArea - (smallCircleArea + largeCircleArea)

/-- The main theorem to prove -/
theorem visible_shaded_area_theorem (config : GridWithCircles) 
  (h1 : config.gridSize = 7)
  (h2 : config.squareSize = 1)
  (h3 : config.smallCircleDiameter = 1)
  (h4 : config.largeCircleDiameter = 3)
  (h5 : config.smallCircleCount = 6)
  (h6 : config.largeCircleCount = 1) :
  visibleShadedArea config = 49 - 3.75 * Real.pi := by
  sorry

#eval (49 : Float) + 3.75  -- Should output approximately 52.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_theorem_l246_24630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l246_24612

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangular region
def triangular_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ line_equation x y

-- Calculate the area of the triangular region
noncomputable def triangle_area : ℝ := 27 / 2

-- Check if a point is inside the triangular region
def point_inside (x y : ℝ) : Prop := triangular_region x y

-- Main theorem
theorem triangle_properties :
  triangle_area = 27 / 2 ∧ ¬ point_inside 1 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l246_24612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l246_24695

/-- Given a hyperbola passing through the point (6, √3) with asymptotes y = ± x/3,
    prove that its equation is x²/9 - y² = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ x^2 / 9 - y^2 = k) →  -- Standard form of hyperbola with asymptotes y = ± x/3
  (6^2 / 9 - (Real.sqrt 3)^2 = 1) →        -- The point (6, √3) satisfies the equation
  x^2 / 9 - y^2 = 1 :=                     -- The equation of the hyperbola
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l246_24695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_result_l246_24631

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original_number : ℝ := 54.68237

/-- Theorem stating that rounding the original number to the nearest hundredth results in 54.68 -/
theorem round_to_hundredth_result :
  round_to_hundredth original_number = 54.68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_result_l246_24631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sergey_tax_refund_l246_24611

/-- Calculates the maximum refundable income tax for Sergey --/
def maxRefundableTax (monthlySalary : ℕ) (treatmentCost : ℕ) (medicationCost : ℕ) (taxRate : ℚ) : ℕ :=
  let annualSalary := monthlySalary * 12
  let totalMedicalExpenses := treatmentCost + medicationCost
  let possibleRefund := (totalMedicalExpenses : ℚ) * taxRate
  let taxPaid := (annualSalary : ℚ) * taxRate
  (min possibleRefund taxPaid).floor.toNat

/-- Theorem stating that Sergey's maximum refundable income tax is 14040 rubles --/
theorem sergey_tax_refund :
  maxRefundableTax 9000 100000 20000 (13 / 100) = 14040 := by
  sorry

#eval maxRefundableTax 9000 100000 20000 (13 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sergey_tax_refund_l246_24611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_is_four_l246_24668

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := 5*(x^2 - 2*x^3) + 3*(x - x^2 + 2*x^4) - (3*x^4 - 2*x^2)

/-- The coefficient of x^2 in the polynomial expression is 4 -/
theorem coeff_x_squared_is_four :
  ∃ (a b c d : ℝ), ∀ x, poly x = a*x^4 + b*x^3 + 4*x^2 + c*x + d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_is_four_l246_24668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_distance_l246_24642

/-- Given a right triangle with hypotenuse 5 miles and one angle 45 degrees,
    and an additional 3 miles perpendicular to one leg of this triangle,
    the distance between the start and end points is √86/2 miles. -/
theorem hiking_distance (h : ℝ) (α : ℝ) :
  h = 5 →
  α = 45 * Real.pi / 180 →
  (Real.sqrt ((h * Real.sin α)^2 + (3 + h * Real.cos α)^2)) = Real.sqrt 86 / 2 := by
  sorry

#check hiking_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_distance_l246_24642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_fractions_l246_24691

theorem positive_difference_of_fractions : 
  |((7^2 + 7^2) / 7 : ℚ) - ((7^2 * 7^2) / 7 : ℚ)| = 329 := by
  -- Convert integer operations to rational
  have h1 : ((7^2 + 7^2) / 7 : ℚ) = 14 := by norm_num
  have h2 : ((7^2 * 7^2) / 7 : ℚ) = 343 := by norm_num
  -- Substitute and calculate
  rw [h1, h2]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_fractions_l246_24691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_kite_area_l246_24610

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a kite with four vertices -/
structure Kite where
  bottomVertex : Point
  topVertex : Point
  leftVertex : Point
  rightVertex : Point

/-- Calculates the area of a kite -/
noncomputable def kiteArea (k : Kite) : ℝ :=
  let base1 := k.leftVertex.x - k.bottomVertex.x
  let height1 := k.topVertex.y - k.bottomVertex.y
  let base2 := k.rightVertex.x - k.topVertex.x
  let height2 := k.topVertex.y - k.bottomVertex.y
  (base1 * height1 + base2 * height2) / 2

/-- Doubles the dimensions of a kite -/
def doubleKite (k : Kite) : Kite :=
  { bottomVertex := { x := k.bottomVertex.x, y := k.bottomVertex.y }
  , topVertex := { x := 2 * k.topVertex.x, y := 2 * k.topVertex.y }
  , leftVertex := { x := 2 * k.leftVertex.x, y := 2 * k.leftVertex.y }
  , rightVertex := { x := 2 * k.rightVertex.x, y := 2 * k.rightVertex.y }
  }

theorem double_kite_area (k : Kite) 
  (h1 : k.bottomVertex = ⟨0, 0⟩)
  (h2 : k.topVertex = ⟨4, 6⟩)
  (h3 : k.leftVertex = ⟨2, 0⟩)
  (h4 : k.rightVertex = ⟨6, 6⟩) :
  kiteArea (doubleKite k) = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_kite_area_l246_24610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l246_24604

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the parallel lines 6x + 8y - 1 = 0 and 6x + 8y - 9 = 0 is 4/5 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 6 8 (-1) (-9) = 4/5 := by
  -- Unfold the definition of distance_between_parallel_lines
  unfold distance_between_parallel_lines
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l246_24604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_b_is_one_range_of_b_when_difference_ge_four_l246_24617

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x + b / x - 3

-- Part 1
theorem range_of_f_when_b_is_one :
  let b := 1
  ∀ x ∈ Set.Icc 1 2, -1 ≤ f b x ∧ f b x ≤ -1/2 := by
  sorry

-- Part 2
theorem range_of_b_when_difference_ge_four :
  ∀ b ≥ 2,
  (∃ M m : ℝ, (∀ x ∈ Set.Icc 1 2, m ≤ f b x ∧ f b x ≤ M) ∧ M - m ≥ 4) →
  b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_b_is_one_range_of_b_when_difference_ge_four_l246_24617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_fee_is_two_l246_24690

/-- Represents the monthly fee for a phone service -/
def monthly_fee : ℝ := 2

/-- Represents the per-minute charge in dollars -/
def per_minute_charge : ℝ := 0.12

/-- Represents the total bill amount in dollars -/
def total_bill : ℝ := 23.36

/-- Represents the number of minutes used -/
def minutes_used : ℕ := 178

/-- Theorem stating that the monthly fee is $2 -/
theorem monthly_fee_is_two :
  monthly_fee = total_bill - (per_minute_charge * minutes_used) :=
by
  -- Unfold the definitions
  unfold monthly_fee total_bill per_minute_charge minutes_used
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_fee_is_two_l246_24690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l246_24693

-- Define the problem setup
variable (A B C D : EuclideanSpace ℝ (Fin 2))
variable (x y : ℝ)

-- State the theorem
theorem vector_relation (h1 : B - C = (5 : ℝ) • (C - D)) 
                        (h2 : A - B = x • (A - C) + y • (A - D)) : 
  x + 2 * y = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l246_24693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_ln_over_x_minus_one_l246_24638

open Real

/-- The function f(x) = ln(x)/(x+1) + 1/x -/
noncomputable def f (x : ℝ) : ℝ := (log x) / (x + 1) + 1 / x

/-- Theorem: For x > 0 and x ≠ 1, f(x) > ln(x)/(x-1) -/
theorem f_greater_than_ln_over_x_minus_one (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  f x > (log x) / (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_ln_over_x_minus_one_l246_24638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l246_24679

noncomputable def f (x : ℝ) : ℝ := Real.log (|x + 3| - |x - 7|)

theorem inequality_solution (m : ℝ) :
  (m = 1 → {x : ℝ | f x < m} = {x : ℝ | 2 < x ∧ x < 7}) ∧
  (∀ x, f x < m ↔ m > 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l246_24679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l246_24665

-- Define the distance between two parallel lines
noncomputable def distance_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_between_lines (c : ℝ) :
  distance_parallel_lines 1 (-2) (-1) (-c) = 2 * Real.sqrt 5 ↔ c = -9 ∨ c = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l246_24665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l246_24646

noncomputable def f (x a b : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (x + a) else Real.cos (x + b)

theorem even_function_condition (a b : ℝ) :
  (∀ x, f x a b = f (-x) a b) →
  (a = π / 3 ∧ b = π / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l246_24646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_multiple_3_or_7_l246_24632

/-- The set of integers from 1 to 50 -/
def S : Finset ℕ := Finset.range 50

/-- The set of multiples of 3 or 7 in S -/
def M : Finset ℕ := S.filter (λ n => 3 ∣ n + 1 ∨ 7 ∣ n + 1)

/-- The probability of selecting a multiple of 3 or 7 from S -/
theorem prob_multiple_3_or_7 : (M.card : ℚ) / (S.card : ℚ) = 21 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_multiple_3_or_7_l246_24632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l246_24640

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := -1/16 * x^2

-- Define the focus of a parabola
def focus : ℝ × ℝ := (0, -4)

-- Theorem statement
theorem parabola_focus :
  ∀ x : ℝ, parabola x = -1/16 * x^2 →
  focus = (0, -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l246_24640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_with_eight_factors_l246_24615

theorem smallest_perfect_square_with_eight_factors : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    (∃ (m : ℕ), n = m^2) ∧ 
    (Finset.filter (λ x => x ∣ n) (Finset.range (n+1))).card = 8 ∧
    (∀ k < n, ¬((∃ (m : ℕ), k = m^2) ∧ 
               (Finset.filter (λ x => x ∣ k) (Finset.range (k+1))).card = 8)) ∧
    n = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_with_eight_factors_l246_24615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_perimeter_not_greater_l246_24600

/-- Represents a polygon in 2D space -/
structure Polygon where
  vertices : List (Real × Real)
  is_closed : vertices.head? = vertices.getLast?

/-- Represents a straight line in 2D space -/
structure Line where
  point1 : Real × Real
  point2 : Real × Real

/-- Calculates the perimeter of a polygon -/
noncomputable def perimeter (p : Polygon) : Real :=
  sorry

/-- Folds a polygon along a line and glues the halves together -/
noncomputable def fold_and_glue (p : Polygon) (l : Line) : Polygon :=
  sorry

/-- Theorem: The perimeter of a polygon after folding and gluing is not greater than the original perimeter -/
theorem folded_perimeter_not_greater (p : Polygon) (l : Line) :
  perimeter (fold_and_glue p l) ≤ perimeter p := by
  sorry

#check folded_perimeter_not_greater

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_perimeter_not_greater_l246_24600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_approximately_9_minutes_l246_24681

/-- Represents the problem of filling a bucket with three taps -/
structure BucketProblem where
  bucket_volume : ℝ
  initial_volume : ℝ
  tap_a_rate : ℝ
  tap_b_fill_fraction : ℝ
  tap_b_fill_time : ℝ
  tap_c_fill_fraction : ℝ
  tap_c_fill_time : ℝ

/-- Calculates the time needed to fill the bucket completely -/
noncomputable def fill_time (problem : BucketProblem) : ℝ :=
  let tap_b_rate := problem.tap_b_fill_fraction * problem.bucket_volume / problem.tap_b_fill_time
  let tap_c_rate := problem.tap_c_fill_fraction * problem.bucket_volume / problem.tap_c_fill_time
  let combined_rate := problem.tap_a_rate + tap_b_rate + tap_c_rate
  let remaining_volume := problem.bucket_volume - problem.initial_volume
  remaining_volume / combined_rate

/-- Theorem stating that the fill time for the given problem is approximately 9 minutes -/
theorem fill_time_approximately_9_minutes (problem : BucketProblem) 
  (h1 : problem.bucket_volume = 50)
  (h2 : problem.initial_volume = 8)
  (h3 : problem.tap_a_rate = 3)
  (h4 : problem.tap_b_fill_fraction = 1/3)
  (h5 : problem.tap_b_fill_time = 20)
  (h6 : problem.tap_c_fill_fraction = 1/2)
  (h7 : problem.tap_c_fill_time = 30) :
  ∃ ε > 0, |fill_time problem - 9| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_approximately_9_minutes_l246_24681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_NNM_l246_24625

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Condition: M × M ≡ M (mod 10) -/
def satisfies_modulo_condition (M : Digit) : Prop :=
  (M.val * M.val) % 10 = M.val

/-- Represents a two-digit number MM where both digits are M -/
def two_digit_MM (M : Digit) : ℕ :=
  10 * M.val + M.val

/-- Represents a three-digit number NNM -/
def three_digit_NNM (N M : Digit) : ℕ :=
  100 * N.val + 10 * N.val + M.val

/-- The product of MM and M equals NNM -/
def product_equals_NNM (N M : Digit) : Prop :=
  two_digit_MM M * M.val = three_digit_NNM N M

theorem greatest_NNM :
  ∃ (N M : Digit), satisfies_modulo_condition M ∧
                   product_equals_NNM N M ∧
                   (∀ (N' M' : Digit), satisfies_modulo_condition M' ∧ product_equals_NNM N' M' →
                                       three_digit_NNM N' M' ≤ three_digit_NNM N M) ∧
                   three_digit_NNM N M = 396 := by
  sorry

#check greatest_NNM

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_NNM_l246_24625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l246_24677

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 20 * x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (5, 0)

-- Define the asymptote of the hyperbola
def hyperbola_asymptote (x y a b : ℝ) : Prop := b * x + a * y = 0

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ := 
  (|a * x + b * y + c|) / Real.sqrt (a^2 + b^2)

-- The main theorem
theorem hyperbola_equation (a b : ℝ) :
  a > b ∧ b > 0 ∧
  (∃ (x y : ℝ), hyperbola x y a b ∧ (x, y) = parabola_focus) ∧
  (∀ (x y : ℝ), hyperbola_asymptote x y a b → 
    distance_point_to_line 5 0 b a 0 = 4) →
  a = 3 ∧ b = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l246_24677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_right_angle_possible_m_l246_24661

/-- The set of possible values for m -/
def possible_m : Set ℕ := {4, 5, 6}

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 1

/-- Theorem stating the possible values of m -/
theorem circle_right_angle_possible_m (m : ℕ) :
  (∃ (P : ℝ × ℝ), circle_C P.1 P.2 ∧
    (P.1 + m) * (P.1 - m) + P.2^2 = 0) →
  m ∈ possible_m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_right_angle_possible_m_l246_24661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangle_area_l246_24644

theorem regular_triangle_area (s r : ℝ) (h1 : r = 4) :
  let A := (Real.sqrt 3 / 4) * s^2
  r = s * Real.sqrt 3 / 6 →
  A = 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangle_area_l246_24644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_partner_contribution_l246_24635

/-- Calculates the partner's contribution percentage for a theater project --/
theorem theater_partner_contribution
  (cost_per_sqft : ℚ)
  (space_per_seat : ℚ)
  (num_seats : ℕ)
  (tom_contribution : ℚ)
  (h1 : cost_per_sqft = 5)
  (h2 : space_per_seat = 12)
  (h3 : num_seats = 500)
  (h4 : tom_contribution = 54000) :
  let total_sqft : ℚ := space_per_seat * num_seats
  let land_cost : ℚ := cost_per_sqft * total_sqft
  let construction_cost : ℚ := 2 * land_cost
  let total_cost : ℚ := land_cost + construction_cost
  let partner_contribution : ℚ := total_cost - tom_contribution
  partner_contribution / total_cost = 2/5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_partner_contribution_l246_24635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_roots_l246_24643

open Real

theorem trigonometric_roots (α β : ℝ) (hα : α ∈ Set.Ioo 0 π) (hβ : β ∈ Set.Ioo 0 π)
  (h_roots : {tan α, tan β} = {x | x^2 - 5*x + 6 = 0}) :
  α + β = 3 * π / 4 ∧ cos (α - β) = 7 * Real.sqrt 2 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_roots_l246_24643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l246_24616

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem circle_properties
  (C : ℝ × ℝ)  -- Center of the circle
  (r : ℝ)      -- Radius of the circle
  (h1 : C ∈ Line 1 1 (-1))  -- Center lies on x + y - 1 = 0
  (h2 : (-1, 1) ∈ Circle C r)  -- Circle passes through (-1, 1)
  (h3 : (-2, -2) ∈ Circle C r)  -- Circle passes through (-2, -2)
  : 
  -- 1. Standard equation of the circle
  (C = (3, -2) ∧ r = 5) ∧
  -- 2. Minimum distance between circle and line x - y + 5 = 0
  (∃ (P Q : ℝ × ℝ), 
    P ∈ Circle C r ∧ 
    Q ∈ Line 1 (-1) 5 ∧
    (∀ (P' Q' : ℝ × ℝ), 
      P' ∈ Circle C r → 
      Q' ∈ Line 1 (-1) 5 → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5 * Real.sqrt 2 - 5) ∧
  -- 3. Equation of tangent line through (0, 3)
  (∃ (k : ℝ), (k = 15/8 ∨ k = 0) ∧
    (∀ (x y : ℝ), (x, y) ∈ Line k (-1) 3 → 
      ((x - C.1)^2 + (y - C.2)^2 = r^2 → 
        ∀ (x' y' : ℝ), (x', y') ∈ Line k (-1) 3 → 
          (x' - C.1)^2 + (y' - C.2)^2 ≥ r^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l246_24616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l246_24654

-- Define the radius of the sphere
noncomputable def p : ℝ := sorry

-- Define the volume of a sphere
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * (radius ^ 3)

-- Define the volume of a hemisphere
noncomputable def hemisphere_volume (radius : ℝ) : ℝ := (1 / 2) * (4 / 3) * Real.pi * (radius ^ 3)

-- Theorem stating the ratio of volumes
theorem volume_ratio :
  (sphere_volume p) / (hemisphere_volume (3 * p)) = 2 / 27 := by
  sorry

#check volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l246_24654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_45_75_l246_24614

theorem common_divisors_45_75 : 
  (Finset.filter (λ x => x ∣ 45 ∧ x ∣ 75) (Finset.range 151)).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_45_75_l246_24614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_grouping_l246_24682

/-- Given a total number of students and a maximum group size, 
    calculate the minimum number of equal-sized groups. -/
def minGroups (totalStudents : ℕ) (maxGroupSize : ℕ) : ℕ :=
  let validDivisors := (List.range totalStudents).filter (fun d => 
    d > 0 && d ≤ maxGroupSize && totalStudents % d = 0)
  match validDivisors.maximum? with
  | none => totalStudents
  | some maxDivisor => totalStudents / maxDivisor

theorem teacher_grouping :
  minGroups 30 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_grouping_l246_24682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l246_24655

/-- Two lines are parallel if their slopes are equal -/
def parallel (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

/-- Distance between two parallel lines -/
noncomputable def distance (a b c1 c2 : ℝ) : ℝ :=
  |c2 - c1| / Real.sqrt (a^2 + b^2)

/-- Theorem: Given parallel lines with specific coefficients and distance, prove the value of c -/
theorem parallel_lines_distance (c : ℝ) :
  parallel 3 (-2) 6 (-4) →
  distance 3 (-2) (-1) (c/2) = 2 * Real.sqrt 13 / 13 →
  c = 2 ∨ c = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l246_24655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_calculation_l246_24664

/-- Calculates the total amount paid for fruits given specific quantities, rates, discounts, and taxes -/
theorem fruit_purchase_calculation (grapes_quantity : ℝ) (grapes_rate : ℝ) (grapes_discount : ℝ) (grapes_tax : ℝ)
                                   (mangoes_quantity : ℝ) (mangoes_rate : ℝ) (mangoes_tax : ℝ) :
  grapes_quantity = 8 →
  grapes_rate = 70 →
  grapes_discount = 0.1 →
  grapes_tax = 0.05 →
  mangoes_quantity = 9 →
  mangoes_rate = 50 →
  mangoes_tax = 0.08 →
  (grapes_quantity * grapes_rate * (1 - grapes_discount) * (1 + grapes_tax)) +
  (mangoes_quantity * mangoes_rate * (1 + mangoes_tax)) = 1015.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_calculation_l246_24664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_k_range_l246_24628

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval [5,8]
def interval : Set ℝ := Set.Icc 5 8

-- Define monotonicity on an interval
def monotonic_on (g : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → g x ≤ g y

-- State the theorem
theorem monotonic_f_k_range :
  ∀ k : ℝ, (monotonic_on (f k) interval) → k ∈ Set.Iic 40 ∪ Set.Ici 64 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_k_range_l246_24628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_k_value_l246_24603

/-- A function f: ℝ → ℝ is linear if there exist a, b ∈ ℝ such that f(x) = ax + b for all x ∈ ℝ -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

/-- The function y = x^(k-1) + 2 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^(k-1) + 2

theorem linear_function_k_value :
  (∃ k : ℝ, IsLinearFunction (f k)) → (∃ k : ℝ, k = 2 ∧ IsLinearFunction (f k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_k_value_l246_24603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_linear_term_l246_24686

theorem coefficient_of_linear_term (x : ℝ) : 
  let expansion := (3 / x^2 + x + 2)^5
  ∃ a b c d e f : ℝ, expansion = a*x^4 + b*x^3 + c*x^2 + 200*x + d + e/x + f/x^2 + 
                    (fun y : ℝ => 1/y^3 + 1/y^4 + 1/y^5 + 1/y^6) x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_linear_term_l246_24686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_from_bankers_gain_l246_24657

/-- Banker's gain calculation -/
noncomputable def bankers_gain (present_worth : ℝ) (years : ℕ) (rate : ℝ) : ℝ :=
  let compound_interest := present_worth * ((1 + rate / 100) ^ years - 1)
  let simple_interest := present_worth * rate * (years : ℝ) / 100
  compound_interest - simple_interest

/-- Present worth calculation given banker's gain -/
theorem present_worth_from_bankers_gain 
  (gain : ℝ) (years : ℕ) (rate : ℝ) :
  ∃ (pw : ℝ), bankers_gain pw years rate = gain ∧ 
  (abs (pw - 1161.29) < 0.01) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_worth_from_bankers_gain_l246_24657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depletion_rate_l246_24618

/-- The value depletion rate of a machine given its initial value, value after a certain time, and the time elapsed. -/
noncomputable def valueDepletionRate (initialValue : ℝ) (finalValue : ℝ) (time : ℝ) : ℝ :=
  1 - (finalValue / initialValue) ^ (1 / time)

/-- Theorem stating that the value depletion rate of a machine with initial value $1200 and final value $972 after 2 years is 0.1 -/
theorem machine_depletion_rate :
  valueDepletionRate 1200 972 2 = 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depletion_rate_l246_24618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_workers_count_l246_24696

theorem company_workers_count (total : ℕ) 
  (h1 : total % 3 = 0)  -- Ensures total is divisible by 3
  (h2 : (total / 3) % 5 = 0)  -- Ensures (total / 3) is divisible by 5
  (h3 : (2 * total / 3) % 5 = 0)  -- Ensures (2 * total / 3) is divisible by 5
  (h4 : 4 * (2 * total / 3) / 5 + 4 * (total / 3) / 5 = 160) :
  total - 160 = 140 := by
  sorry

#check company_workers_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_workers_count_l246_24696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_can_reach_any_area_grasshopper_can_fall_into_hole_l246_24675

/-- Represents a square meadow with a circular hole -/
structure Meadow where
  side_length : ℝ
  hole_radius : ℝ
  side_positive : side_length > 0
  hole_fits : hole_radius < side_length / 2

/-- Represents the position of the grasshopper -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a jump of the grasshopper -/
noncomputable def jump (p : Position) (vertex : Position) : Position :=
  { x := p.x + (vertex.x - p.x) / 2,
    y := p.y + (vertex.y - p.y) / 2 }

/-- Theorem stating that the grasshopper can reach any small area in the meadow -/
theorem grasshopper_can_reach_any_area (m : Meadow) :
  ∀ ε > 0, ∃ n : ℕ, ∀ p : Position,
    p.x ≥ 0 ∧ p.x ≤ m.side_length ∧ p.y ≥ 0 ∧ p.y ≤ m.side_length →
    ∃ q : Position, 
      q.x ≥ 0 ∧ q.x ≤ m.side_length ∧ q.y ≥ 0 ∧ q.y ≤ m.side_length ∧
      |q.x - p.x| < ε ∧ |q.y - p.y| < ε ∧
      (∃ k : ℕ, ∃ vertices : List Position, 
        vertices.length = k ∧
        q = (vertices.foldl jump p)) :=
by
  sorry

/-- Theorem stating that the grasshopper can fall into the hole -/
theorem grasshopper_can_fall_into_hole (m : Meadow) :
  ∃ p : Position, p.x^2 + p.y^2 < m.hole_radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_can_reach_any_area_grasshopper_can_fall_into_hole_l246_24675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_dilution_l246_24621

/-- Represents the amount of water added to dilute an acid solution -/
noncomputable def water_added (s : ℝ) : ℝ :=
  15 * s / (s + 15)

/-- Theorem stating the correct amount of water to add to dilute an acid solution -/
theorem acid_dilution (s : ℝ) (y : ℝ) (h1 : s > 30) :
  (s * s / 100 = (s - 15) / 100 * (s + y)) → y = water_added s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_dilution_l246_24621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_trig_expression_value_l246_24698

-- Part 1
theorem tan_alpha_value (α : ℝ) (m : ℝ) :
  Real.cos α = -1/3 → (m, 1) ∈ {p : ℝ × ℝ | p.1 * Real.cos α - p.2 * Real.sin α = 0} →
  Real.tan α = -2 * Real.sqrt 2 :=
by sorry

-- Part 2
theorem trig_expression_value :
  Real.tan (150 * π / 180) * Real.cos (-210 * π / 180) * Real.sin (-420 * π / 180) /
  (Real.sin (1050 * π / 180) * Real.cos (-600 * π / 180)) = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_trig_expression_value_l246_24698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_mixed_base_expression_l246_24627

theorem calculate_mixed_base_expression : 
  let base_10 : ℤ → ℤ := λ n => n
  let base_3 : ℤ → ℤ := λ n => (n % 3) + 3 * ((n / 3) % 3) + 9 * (n / 9)
  let base_9 : ℤ → ℤ := λ n => (n % 9) + 9 * ((n / 9) % 9) + 81 * ((n / 81) % 9) + 729 * (n / 729)
  let base_7 : ℤ → ℤ := λ n => (n % 7) + 7 * ((n / 7) % 7) + 49 * ((n / 49) % 7) + 343 * (n / 343)
  
  (base_10 2468) / (base_3 111) - (base_9 3471) + (base_7 1234) = -1919
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_mixed_base_expression_l246_24627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_coffee_consumption_l246_24671

/-- The number of coffees John bought per day before the price increase -/
def original_coffees : ℕ := 4

/-- The original price of each coffee in dollars -/
def original_price : ℚ := 2

/-- The price increase percentage -/
def price_increase : ℚ := (1 / 2)

/-- The new price of each coffee after the increase -/
def new_price : ℚ := original_price * (1 + price_increase)

/-- The number of coffees John buys after the price increase -/
def new_coffees : ℚ := (original_coffees : ℚ) / 2

/-- The amount John saves per day after changing his coffee consumption -/
def savings : ℚ := 2

theorem john_coffee_consumption :
  original_coffees * original_price - new_coffees * new_price = savings ∧
  original_coffees = 4 := by
  sorry

#eval original_coffees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_coffee_consumption_l246_24671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teapot_teacup_cost_comparison_l246_24666

/-- Represents the cost of purchasing teapots and teacups under different options -/
def CostFunction (x : ℝ) : ℝ × ℝ := (4 * x + 80, 3.6 * x + 90)

/-- Represents a purchasing plan -/
structure PurchasePlan where
  teapots_option1 : ℕ
  teacups_option1 : ℕ
  teapots_option2 : ℕ
  teacups_option2 : ℕ

/-- Calculates the total cost of a purchasing plan -/
def totalCost (plan : PurchasePlan) : ℝ :=
  20 * plan.teapots_option1 + 4 * (plan.teacups_option1 - plan.teapots_option1) +
  0.9 * (20 * plan.teapots_option2 + 4 * plan.teacups_option2)

theorem teapot_teacup_cost_comparison :
  ∀ x : ℝ, x > 5 →
    let (cost1, cost2) := CostFunction x
    (cost1 < cost2 ∧ x = 20) ∧
    ∃ (plan : PurchasePlan),
      totalCost plan < (CostFunction 20).1 ∧
      plan.teapots_option1 + plan.teapots_option2 = 5 ∧
      plan.teacups_option1 + plan.teacups_option2 = 20 :=
by sorry

#check teapot_teacup_cost_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teapot_teacup_cost_comparison_l246_24666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_sum_ratio_l246_24641

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else n % 10 + sum_of_digits (n / 10)

/-- Theorem: 1099 maximizes s(n)/n for four-digit numbers -/
theorem max_digit_sum_ratio :
  ∀ n : ℕ, 1000 ≤ n → n ≤ 9999 → 
    (sum_of_digits n : ℚ) / n ≤ (sum_of_digits 1099 : ℚ) / 1099 := by
  sorry

#eval sum_of_digits 1099  -- Should output 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_sum_ratio_l246_24641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetric_points_distance_l246_24659

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the parabola y = x^2 - 7 -/
def on_parabola (p : Point) : Prop :=
  p.y = p.x^2 - 7

/-- Checks if two points are symmetric about the line x + y = 0 -/
def symmetric_about_line (a b : Point) : Prop :=
  a.x + a.y = -(b.x + b.y)

/-- Calculates the distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Main theorem -/
theorem parabola_symmetric_points_distance :
  ∃ (a b : Point), a ≠ b ∧ on_parabola a ∧ on_parabola b ∧
  symmetric_about_line a b ∧ distance a b = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetric_points_distance_l246_24659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l246_24685

noncomputable section

-- Define the curve
noncomputable def x (t : Real) : Real := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def y (t : Real) : Real := Real.arccos (1 / Real.sqrt (1 + t^2))

-- Define the point at t₀ = 1
def t₀ : Real := 1
noncomputable def x₀ : Real := x t₀
noncomputable def y₀ : Real := y t₀

-- Define the derivative of x with respect to t
noncomputable def dx_dt (t : Real) : Real := 1 / (1 + t^2)

-- Define the derivative of y with respect to t
noncomputable def dy_dt (t : Real) : Real := t / Real.sqrt (1 + t^2)

-- Define the slope of the tangent line at t₀
noncomputable def m_tangent : Real := (dy_dt t₀) / (dx_dt t₀)

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∀ x y : Real, y = m_tangent * (x - x₀) + y₀ ↔ y = 2 * x - Real.pi / 4 := by
  sorry

-- Theorem for the normal line equation
theorem normal_line_equation :
  ∀ x y : Real, y = (-1 / m_tangent) * (x - x₀) + y₀ ↔ y = -x / 2 + 3 * Real.pi / 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l246_24685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_length_ratio_l246_24697

/-- A rectangle with width w, length 10, and perimeter 30 has a width to length ratio of 1:2 -/
theorem rectangle_width_length_ratio :
  ∀ w : ℚ,
  w > 0 →
  2 * w + 2 * 10 = 30 →
  w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_length_ratio_l246_24697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equivalence_l246_24619

def same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₁ = θ₂ + 2 * Real.pi * k

theorem angle_equivalence (k : ℤ) :
  same_terminal_side (k * 360 * (Real.pi / 180) - 315 * (Real.pi / 180)) (9 * Real.pi / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equivalence_l246_24619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_result_proof_l246_24684

theorem sixth_result_proof (n : ℕ) (total_avg first_six_avg last_six_avg sixth_result : ℚ) : 
  n = 11 ∧ 
  total_avg = 52 ∧ 
  first_six_avg = 49 ∧ 
  last_six_avg = 52 ∧ 
  (n : ℚ) * total_avg = 6 * first_six_avg + 6 * last_six_avg - sixth_result →
  sixth_result = 34 := by
  intro h
  -- Proof steps would go here
  sorry

#check sixth_result_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_result_proof_l246_24684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_enrolled_l246_24653

theorem students_not_enrolled (total : ℕ) (biology_percent : ℚ) (chemistry_percent : ℚ) :
  total = 880 →
  biology_percent = 1/2 →
  chemistry_percent = 3/10 →
  (total : ℚ) * biology_percent + (total : ℚ) * chemistry_percent ≤ total →
  total - (↑total * biology_percent).floor - (↑total * chemistry_percent).floor = 176 := by
  sorry

#check students_not_enrolled

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_enrolled_l246_24653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_y_coordinate_l246_24694

-- Define the line equation
def line (x : ℝ) : ℝ := x - 1

-- Define the parabola equation (noncomputable due to sqrt)
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (8 * x)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = line x ∧ p.2^2 = 8 * x}

-- Theorem statement
theorem midpoint_y_coordinate :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  (A.2 + B.2) / 2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_y_coordinate_l246_24694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l246_24678

structure Plane :=
  (Point : Type)
  (Circle : Type)
  (Line : Type)
  (on_circle : Point → Circle → Prop)
  (on_line : Point → Line → Prop)
  (intersect_circles : Circle → Circle → Set Point)
  (intersect_line_circle : Line → Circle → Set Point)
  (midpoint_arc : Point → Point → Circle → Point)
  (tangent_at : Point → Circle → Line)
  (mk_line : Point → Point → Line)

theorem circle_tangency (π : Plane) 
  (O₁ O₂ : π.Circle) 
  (A B M P Q : π.Point) 
  (l₁ l₂ : π.Line) :
  A ∈ π.intersect_circles O₁ O₂ →
  B ∈ π.intersect_circles O₁ O₂ →
  M = π.midpoint_arc A B O₁ →
  P ∈ π.intersect_line_circle (π.mk_line M P) O₁ →
  Q ∈ π.intersect_line_circle (π.mk_line M P) O₂ →
  l₁ = π.tangent_at P O₁ →
  l₂ = π.tangent_at Q O₂ →
  ∃ (C : π.Circle), (∀ X : π.Point, π.on_circle X C ↔ 
    (π.on_line X l₁ ∨ π.on_line X l₂ ∨ (π.on_line X (π.mk_line A B)))) ∧
    ∃ (Y : π.Point), Y ∈ π.intersect_circles C O₂ ∧ 
      ∀ (Z : π.Point), Z ∈ π.intersect_circles C O₂ → Z = Y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l246_24678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l246_24662

/-- Represents a parabola y = ax^2 -/
structure Parabola where
  a : ℝ
  pos_a : a > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := 0, y := 1 / (4 * p.a) }

/-- A line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  ∃ t : ℝ, p = { x := l.p1.x + t * (l.p2.x - l.p1.x), y := l.p1.y + t * (l.p2.y - l.p1.y) }

/-- Distance between two points -/
noncomputable def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a parabola y = ax^2 (a > 0), if a line through its focus F
    intersects the parabola at points P and Q, and the distances PF and FQ
    are p and q respectively, then 1/p + 1/q = 4a -/
theorem parabola_intersection_sum (para : Parabola) 
  (l : Line) (P Q : Point) (p q : ℝ) 
  (h1 : l.p1 = focus para) 
  (h2 : P.onLine l ∧ Q.onLine l) 
  (h3 : P.y = para.a * P.x^2 ∧ Q.y = para.a * Q.x^2) 
  (h4 : dist P (focus para) = p ∧ dist Q (focus para) = q) : 
  1/p + 1/q = 4 * para.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_sum_l246_24662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l246_24670

theorem integral_equality : 
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), (Real.sqrt (1 - x^2) + x + x^3) = (Real.pi + 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l246_24670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l246_24602

noncomputable section

open Real

def f (x : ℝ) : ℝ := (log x + 1) / exp x

def h (x : ℝ) : ℝ := 1 - x - x * log x

def g (x : ℝ) : ℝ := x * deriv f x

theorem function_properties (x : ℝ) (hx : x > 0) :
  (∃ (max_h : ℝ), max_h = 1 + exp (-2) ∧ ∀ y > 0, h y ≤ max_h) ∧
  g x < 1 + exp (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l246_24602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_above_threshold_l246_24623

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add a case for 0
  | 1 => 1/2
  | (n+2) => let a_n := sequence_a (n+1)
             let a_prev := sequence_a n
             (2 * a_prev) / (a_prev + 1)

theorem largest_k_above_threshold : 
  (∀ n : ℕ, n ≥ 2 → sequence_a n * sequence_a (n-1) - 2 * sequence_a n + sequence_a (n-1) = 0) → 
  (∃ k : ℕ, k = 11 ∧ 
    sequence_a k > 1/2017 ∧ 
    ∀ m : ℕ, m > k → sequence_a m ≤ 1/2017) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_above_threshold_l246_24623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l246_24699

/-- Hyperbola C₁ -/
def C₁ (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Hyperbola C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 / 4 - y^2 / 16 = 1

/-- The asymptotes of a hyperbola with equation x²/a² - y²/b² = 1 -/
def asymptotes (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b/a) * x ∨ y = -(b/a) * x}

/-- The right focus of a hyperbola with equation x²/a² - y²/b² = 1 -/
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

theorem hyperbola_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  asymptotes a b = asymptotes 2 4 ∧ right_focus a b = (Real.sqrt 5, 0) →
  a = 1 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l246_24699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_pasture_optimization_l246_24658

/-- A rectangular cow pasture problem -/
theorem cow_pasture_optimization (barn_length : ℝ) (fence_cost : ℝ) (total_budget : ℝ)
  (h1 : barn_length = 500)
  (h2 : fence_cost = 7)
  (h3 : total_budget = 1470)
  (h4 : 0 < fence_cost)
  (h5 : 0 < total_budget) :
  ∃ (parallel_side : ℝ), 
    (parallel_side ≥ 103 ∧ parallel_side ≤ 105) ∧ 
    ∀ (other_parallel : ℝ), 
      0 ≤ other_parallel ∧ 
      other_parallel ≤ barn_length / 2 ∧
      (total_budget / fence_cost - 2 * (barn_length / 2 - other_parallel) / 2) * other_parallel ≤ 
      (total_budget / fence_cost - 2 * (barn_length / 2 - parallel_side) / 2) * parallel_side :=
by sorry

#check cow_pasture_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_pasture_optimization_l246_24658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_squared_plus_two_equality_l246_24645

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_squared_plus_two_equality (x : ℝ) : 
  floor (x^2 + 2*x) = (floor x)^2 + 2*(floor x) ↔ ∃ n : ℤ, x = ↑n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_squared_plus_two_equality_l246_24645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_and_line_l246_24648

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 1/a^2 + 3/(4*b^2) = 1

/-- A line intersecting the ellipse with specific properties -/
structure IntersectingLine (E : SpecialEllipse) where
  k : ℝ
  m : ℝ
  h_slope_sum : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/E.a^2 + y₁^2/E.b^2 = 1 → 
    x₂^2/E.a^2 + y₂^2/E.b^2 = 1 → 
    y₁ = k*x₁ + m → 
    y₂ = k*x₂ + m → 
    y₁/x₁ + y₂/x₂ = 2
  h_tangent : m^2 / (k^2 + 1) = 1

theorem special_ellipse_and_line (E : SpecialEllipse) : 
  E.a^2 = 4 ∧ E.b^2 = 1 ∧ 
  ∃ (l : IntersectingLine E), l.k = -1 ∧ (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_and_line_l246_24648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l246_24605

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |triangle_area 26 24 20 - 228| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l246_24605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l246_24633

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) + 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 - Real.sqrt 3

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 6) + 1

theorem f_properties (ω : ℝ) (h_ω : ω > 0) (h_period : ∀ x, f ω (x + Real.pi) = f ω x) :
  ω = 1 ∧
  (∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * Real.pi + 5 * Real.pi / 12) (k * Real.pi + 11 * Real.pi / 12))) ∧
  (∀ b : ℝ, (∃ S : Finset ℝ, S.card ≥ 10 ∧ (∀ x ∈ S, x ∈ Set.Icc 0 b ∧ g ω x = 0)) → b ≥ 59 * Real.pi / 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l246_24633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_P_is_tangent_tangent_M_are_tangents_tangent_length_from_M_l246_24656

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define points P and M
def P : ℝ × ℝ := (Real.sqrt 2 + 1, 2 - Real.sqrt 2)
def M : ℝ × ℝ := (3, 1)

-- Define the tangent line through P
def tangent_P (x y : ℝ) : Prop := x - y + 1 - 2 * Real.sqrt 2 = 0

-- Define the tangent lines through M
def tangent_M1 (x : ℝ) : Prop := x - 3 = 0
def tangent_M2 (x y : ℝ) : Prop := 3 * x - 4 * y - 5 = 0

-- Theorem statements
theorem tangent_P_is_tangent :
  ∃ (x y : ℝ), circle_C x y ∧ tangent_P x y ∧ (x, y) = P :=
sorry

theorem tangent_M_are_tangents :
  (∃ (x y : ℝ), circle_C x y ∧ tangent_M1 x) ∧
  (∃ (x y : ℝ), circle_C x y ∧ tangent_M2 x y) ∧
  tangent_M1 M.1 ∧ tangent_M2 M.1 M.2 :=
sorry

theorem tangent_length_from_M :
  ∃ (x y : ℝ), circle_C x y ∧ tangent_M1 x ∧
  Real.sqrt ((x - M.1)^2 + (y - M.2)^2) = 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_P_is_tangent_tangent_M_are_tangents_tangent_length_from_M_l246_24656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l246_24689

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

-- State the theorem
theorem zero_in_interval :
  (∀ x y, 0 < x ∧ x < y → f x < f y) →  -- f is increasing on (0, +∞)
  f 2 < 0 →                            -- f(2) < 0
  f 3 > 0 →                            -- f(3) > 0
  ∃! x, 2 < x ∧ x < 3 ∧ f x = 0 :=     -- There exists a unique zero in (2, 3)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l246_24689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_45_degrees_l246_24673

/-- The area of a figure formed by rotating a semicircle around one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (1/2) * (2*R)^2 * α

theorem rotated_semicircle_area_45_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (π/4) = π * R^2 / 2 := by
  -- Unfold the definition of rotated_semicircle_area
  unfold rotated_semicircle_area
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_45_degrees_l246_24673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l246_24629

theorem triangle_angle_B (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (condition : a^2 + c^2 - b^2 = Real.sqrt 3 * a * c) : 
  Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l246_24629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_quadratic_l246_24674

/-- Given a linear function f(x) = ax + b with a root of 1, 
    prove that the roots of g(x) = bx^2 + ax are 0 and 1 -/
theorem roots_of_quadratic (a b : ℝ) : 
  (∃ f : ℝ → ℝ, f = (λ x ↦ a * x + b) ∧ f 1 = 0) →
  (∃ g : ℝ → ℝ, g = (λ x ↦ b * x^2 + a * x) ∧ 
   ∀ x, g x = 0 ↔ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_quadratic_l246_24674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leghorn_hens_count_l246_24624

/-- Represents the number of hens for each breed of chicken on a farm -/
structure ChickenFarm where
  total_chickens : ℕ
  black_copper_marans_percent : ℚ
  rhode_island_reds_percent : ℚ
  leghorns_percent : ℚ
  black_copper_marans_hens_percent : ℚ
  rhode_island_reds_hens_percent : ℚ
  leghorns_hens_percent : ℚ

/-- Calculates the number of Leghorn hens on the farm -/
def leghorn_hens (farm : ChickenFarm) : ℕ :=
  Int.toNat ((farm.total_chickens : ℚ) * farm.leghorns_percent * farm.leghorns_hens_percent).floor

/-- Theorem stating that the number of Leghorn hens on the given farm is 105 -/
theorem leghorn_hens_count (farm : ChickenFarm) 
  (h1 : farm.total_chickens = 500)
  (h2 : farm.black_copper_marans_percent = 1/4)
  (h3 : farm.rhode_island_reds_percent = 2/5)
  (h4 : farm.leghorns_percent = 7/20)
  (h5 : farm.black_copper_marans_hens_percent = 13/20)
  (h6 : farm.rhode_island_reds_hens_percent = 11/20)
  (h7 : farm.leghorns_hens_percent = 3/5) :
  leghorn_hens farm = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leghorn_hens_count_l246_24624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_composition_l246_24680

-- Define the function g
def g : ℕ → ℕ := sorry

-- Define the inverse function g⁻¹
def g_inv : ℕ → ℕ := sorry

-- State the properties of g
axiom g_2 : g 2 = 8
axiom g_3 : g 3 = 15
axiom g_4 : g 4 = 24
axiom g_5 : g 5 = 35
axiom g_6 : g 6 = 48

-- State the properties of g⁻¹
axiom g_inv_48 : g_inv 48 = 6
axiom g_inv_24 : g_inv 24 = 4
axiom g_inv_15 : g_inv 15 = 3

-- State the continuation of the pattern
axiom g_7 : g 7 = 61

-- State the theorem to be proved
theorem g_inv_composition : 
  g_inv (g_inv 48 + g_inv 24 - g_inv 15) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_composition_l246_24680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_more_heads_than_tails_nine_coins_l246_24609

theorem probability_more_heads_than_tails_nine_coins :
  let n : ℕ := 9
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := (Finset.range 5).sum (λ i ↦ Nat.choose n (n - i))
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_more_heads_than_tails_nine_coins_l246_24609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l246_24651

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2) + 1 / Real.sqrt (5 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 2 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l246_24651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l246_24676

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (10, Real.pi / 6, 2)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 =
  (5 * Real.sqrt 3, 5, 2) := by
  sorry

#check cylindrical_to_rectangular_conversion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l246_24676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_numbers_l246_24647

theorem sum_of_two_numbers (A B : ℕ) : 
  A + B = 888888 ∧ 
  A / 100000 % 10 = 2 ∧ A / 10 % 10 = 2 ∧
  B / 100000 % 10 = 6 ∧ B / 10 % 10 = 6 ∧
  (A - A / 100000 * 100000 - A / 10 % 10 * 10) = 
    3 * (B - B / 100000 * 100000 - B / 10 % 10 * 10) →
  A = 626626 ∧ B = 262262 := by
  sorry

#eval 626626 + 262262  -- Should output 888888
#eval 626626 / 100000 % 10  -- Should output 2
#eval 626626 / 10 % 10  -- Should output 2
#eval 262262 / 100000 % 10  -- Should output 6
#eval 262262 / 10 % 10  -- Should output 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_numbers_l246_24647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_identity_l246_24607

theorem tangent_identity (α : ℝ) : 
  Real.tan (π / 4 + α) = 1 → (2 * Real.sin α + Real.cos α) / (3 * Real.cos α - Real.sin α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_identity_l246_24607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_l246_24692

noncomputable def matrix_elements (i j : Fin 3) : ℝ := 
  Real.sin (((i.val * 3 + j.val + 1 : ℕ) : ℝ) + Real.pi / 4)

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ := 
  Matrix.of (λ i j ↦ matrix_elements i j)

theorem determinant_zero : Matrix.det A = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_l246_24692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l246_24608

/-- Represents a triangle with an angle bisector -/
structure TriangleWithBisector where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  is_triangle : Prop
  is_angle_bisector : Prop

/-- The length of a line segment between two points -/
noncomputable def length (p q : ℝ × ℝ) : ℝ := sorry

/-- The circumcenter of a triangle -/
noncomputable def circumcenter (p q r : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Checks if a point lies on a line segment -/
def lies_on_segment (p q r : ℝ × ℝ) : Prop := sorry

theorem triangle_side_length 
  (t : TriangleWithBisector)
  (h1 : length t.B t.P = 16)
  (h2 : length t.P t.C = 20)
  (h3 : lies_on_segment t.A t.C (circumcenter t.A t.B t.P)) :
  length t.A t.B = (144 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l246_24608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplifies_to_one_third_l246_24639

-- Define the value of a
noncomputable def a : ℝ := |(-6)| - (1/2)⁻¹

-- Define the original expression
noncomputable def original_expression (x : ℝ) : ℝ :=
  x / (x + 2) - (x + 3) / (x^2 - 4) / ((2*x + 6) / (2*x^2 - 8*x + 8))

-- Theorem statement
theorem expression_simplifies_to_one_third :
  original_expression a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplifies_to_one_third_l246_24639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l246_24606

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 6 * Real.cos (ω * x / 2)^2 - 3

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (3 * x)

theorem function_properties (ω θ : ℝ) (h_ω_pos : ω > 0) (h_θ_bounds : 0 < θ ∧ θ < Real.pi / 2) :
  (∀ x, f ω (x + θ) = f ω (-x + θ) ∧ 
   ∀ y, f ω (y + θ + Real.pi) = f ω (y + θ) ∧
   ∀ z, z > 0 → z < Real.pi → (f ω (z + θ) ≠ f ω (θ))) →
  (ω = 2 ∧ θ = Real.pi / 12) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 3 → g ω x < g ω y) →
  ω ≤ 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l246_24606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_society_theorem_l246_24667

/-- Represents a group of people and their acquaintances. -/
structure Society where
  people : Finset Nat
  knows : Nat → Nat → Prop

/-- The property that for any two people, exactly one other person knows both of them. -/
def HasUniqueCommonAcquaintance (s : Society) : Prop :=
  ∀ a b, a ∈ s.people → b ∈ s.people → a ≠ b →
    ∃! c, c ∈ s.people ∧ c ≠ a ∧ c ≠ b ∧ s.knows c a ∧ s.knows c b

/-- The existence of a person who knows everyone else. -/
def HasUniversalAcquaintance (s : Society) : Prop :=
  ∃ a, a ∈ s.people ∧ ∀ b, b ∈ s.people → b ≠ a → s.knows a b

/-- The main theorem: In a society of 11 people, if for any two people there exists
    exactly one person who knows both of them, then there exists at least one person
    who knows everyone else. -/
theorem society_theorem (s : Society)
    (size_eq : s.people.card = 11)
    (unique_common : HasUniqueCommonAcquaintance s) :
    HasUniversalAcquaintance s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_society_theorem_l246_24667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_E_l246_24652

/-- Represents a square in the cube net --/
inductive Square
| A | B | C | D | E | F

/-- Represents the cube net --/
structure CubeNet where
  squares : List Square
  adjacent : Square → Square → Prop
  opposite_in_net : Square → Square → Prop

/-- Represents the folded cube --/
structure Cube where
  faces : List Square
  opposite_in_cube : Square → Square → Prop

/-- Given a cube net, folds it into a cube --/
def fold (net : CubeNet) : Cube :=
  sorry

theorem opposite_face_of_E (net : CubeNet) (cube : Cube) :
  net.squares = [Square.A, Square.B, Square.C, Square.D, Square.E, Square.F] →
  net.adjacent Square.A Square.B →
  net.adjacent Square.A Square.C →
  net.opposite_in_net Square.B Square.D →
  cube = fold net →
  cube.opposite_in_cube Square.E Square.F :=
by
  sorry

#check opposite_face_of_E

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_E_l246_24652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l246_24636

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- four-digit positive integer
  (∀ d ∈ n.digits 10, d > 1) ∧  -- smallest digit > 1
  (∀ d ∈ n.digits 10, n % d = 0) ∧  -- divisible by each digit
  (n.digits 10).Nodup  -- all digits are different

theorem smallest_valid_number : 
  is_valid_number 3246 ∧ 
  ∀ m, is_valid_number m → m ≥ 3246 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l246_24636
