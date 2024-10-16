import Mathlib

namespace NUMINAMATH_CALUDE_simplify_absolute_value_expression_l3797_379766

theorem simplify_absolute_value_expression 
  (a b c : ℝ) 
  (ha : |a| + a = 0) 
  (hab : |a * b| = a * b) 
  (hc : |c| - c = 0) : 
  |b| - |a + b| - |c - b| + |a - c| = b := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_expression_l3797_379766


namespace NUMINAMATH_CALUDE_ticket_sales_solution_l3797_379726

/-- Represents the number of tickets sold for each type -/
structure TicketSales where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the conditions of the ticket sales problem -/
def validTicketSales (s : TicketSales) : Prop :=
  s.a + s.b + s.c = 400 ∧
  50 * s.a + 40 * s.b + 30 * s.c = 15500 ∧
  s.b = s.c

/-- Theorem stating the solution to the ticket sales problem -/
theorem ticket_sales_solution :
  ∃ (s : TicketSales), validTicketSales s ∧ s.a = 100 ∧ s.b = 150 ∧ s.c = 150 := by
  sorry


end NUMINAMATH_CALUDE_ticket_sales_solution_l3797_379726


namespace NUMINAMATH_CALUDE_simplified_expression_l3797_379705

theorem simplified_expression (b : ℚ) : (3 * b + 7 - 5 * b) / 3 = -2/3 * b + 7/3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l3797_379705


namespace NUMINAMATH_CALUDE_sales_prediction_at_34_l3797_379798

/-- Represents the linear regression equation for predicting cold drink sales based on temperature -/
def predict_sales (x : ℝ) : ℝ := 2 * x + 60

/-- Theorem stating that when the temperature is 34°C, the predicted sales volume is 128 cups -/
theorem sales_prediction_at_34 :
  predict_sales 34 = 128 := by
  sorry

end NUMINAMATH_CALUDE_sales_prediction_at_34_l3797_379798


namespace NUMINAMATH_CALUDE_arc_length_for_specific_circle_l3797_379785

/-- Given a circle with radius π and a central angle of 120°, the arc length is (2π²)/3 -/
theorem arc_length_for_specific_circle :
  let r : ℝ := Real.pi
  let θ : ℝ := 120
  let l : ℝ := (θ / 180) * Real.pi * r
  l = (2 * Real.pi^2) / 3 := by sorry

end NUMINAMATH_CALUDE_arc_length_for_specific_circle_l3797_379785


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3797_379716

/-- Given a journey with a distance and time, calculate the average speed -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 210) (h2 : time = 35/6) :
  distance / time = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3797_379716


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3797_379799

/-- Represents a rectangular field with a square pond -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ
  length_is_double_width : field_length = 2 * field_width
  length_is_96 : field_length = 96
  pond_side_is_8 : pond_side = 8

/-- The ratio of the pond area to the field area is 1:72 -/
theorem pond_to_field_area_ratio (f : FieldWithPond) :
  (f.pond_side^2) / (f.field_length * f.field_width) = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3797_379799


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l3797_379793

-- Define the number to factorize
def n : ℕ := 65535

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l3797_379793


namespace NUMINAMATH_CALUDE_black_squares_eaten_l3797_379787

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Defines whether a square is black -/
def isBlack (s : Square) : Bool :=
  (s.row + s.col) % 2 = 0

/-- The list of squares eaten by termites -/
def eatenSquares : List Square := [
  ⟨3, 1⟩, ⟨4, 6⟩, ⟨3, 7⟩,
  ⟨4, 1⟩, ⟨2, 3⟩, ⟨2, 4⟩, ⟨4, 3⟩,
  ⟨3, 5⟩, ⟨3, 2⟩, ⟨4, 7⟩,
  ⟨3, 6⟩, ⟨2, 6⟩
]

/-- Counts the number of black squares in a list of squares -/
def countBlackSquares (squares : List Square) : Nat :=
  squares.filter isBlack |>.length

/-- Theorem stating that the number of black squares eaten is 12 -/
theorem black_squares_eaten :
  countBlackSquares eatenSquares = 12 := by
  sorry


end NUMINAMATH_CALUDE_black_squares_eaten_l3797_379787


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l3797_379732

theorem trigonometric_expression_value (α : Real) (h : α = -35 * π / 6) :
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l3797_379732


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3797_379706

theorem quadratic_root_implies_a (a : ℝ) :
  (3^2 + a*3 + a - 1 = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3797_379706


namespace NUMINAMATH_CALUDE_coefficient_of_fifth_power_l3797_379781

theorem coefficient_of_fifth_power (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (2*x - 1)^5 * (x + 2) = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 →
  a₅ = 176 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_fifth_power_l3797_379781


namespace NUMINAMATH_CALUDE_total_strings_is_40_l3797_379734

/-- The number of strings on all instruments in Francis' family -/
def total_strings : ℕ :=
  let ukulele_count : ℕ := 2
  let guitar_count : ℕ := 4
  let violin_count : ℕ := 2
  let strings_per_ukulele : ℕ := 4
  let strings_per_guitar : ℕ := 6
  let strings_per_violin : ℕ := 4
  ukulele_count * strings_per_ukulele +
  guitar_count * strings_per_guitar +
  violin_count * strings_per_violin

theorem total_strings_is_40 : total_strings = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_strings_is_40_l3797_379734


namespace NUMINAMATH_CALUDE_tangent_chord_length_l3797_379789

/-- The circle with equation x^2 + y^2 - 6x - 8y + 20 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + 20 = 0}

/-- The origin point (0, 0) -/
def Origin : ℝ × ℝ := (0, 0)

/-- A point is on the circle if it satisfies the circle equation -/
def IsOnCircle (p : ℝ × ℝ) : Prop := p ∈ Circle

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def IsTangentLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), IsOnCircle p ∧ l ∩ Circle = {p}

/-- The theorem stating that the length of the chord formed by two tangent lines
    from the origin to the circle is 4√5 -/
theorem tangent_chord_length :
  ∃ (A B : ℝ × ℝ) (OA OB : Set (ℝ × ℝ)),
    IsOnCircle A ∧ IsOnCircle B ∧
    IsTangentLine OA ∧ IsTangentLine OB ∧
    Origin ∈ OA ∧ Origin ∈ OB ∧
    A ∈ OA ∧ B ∈ OB ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_chord_length_l3797_379789


namespace NUMINAMATH_CALUDE_molecular_weight_AlOH3_l3797_379776

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define the composition of Al(OH)3
def Al_count : ℕ := 1
def O_count : ℕ := 3
def H_count : ℕ := 3

-- Define the number of moles
def moles : ℝ := 7

-- Theorem statement
theorem molecular_weight_AlOH3 :
  let molecular_weight := Al_count * atomic_weight_Al + O_count * atomic_weight_O + H_count * atomic_weight_H
  moles * molecular_weight = 546.07 := by
  sorry


end NUMINAMATH_CALUDE_molecular_weight_AlOH3_l3797_379776


namespace NUMINAMATH_CALUDE_exclusive_or_implication_l3797_379784

theorem exclusive_or_implication :
  let statement1 := ¬p ∧ ¬q
  let statement2 := ¬p ∧ q
  let statement3 := p ∧ ¬q
  let statement4 := p ∧ q
  let exclusive_condition := ¬(p ∧ q)
  (statement1 → exclusive_condition) ∧
  (statement2 → exclusive_condition) ∧
  (statement3 → exclusive_condition) ∧
  ¬(statement4 → exclusive_condition) := by
  sorry

end NUMINAMATH_CALUDE_exclusive_or_implication_l3797_379784


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3797_379715

theorem trigonometric_identity : 
  (Real.cos (10 * π / 180)) / (Real.tan (20 * π / 180)) + 
  Real.sqrt 3 * (Real.sin (10 * π / 180)) * (Real.tan (70 * π / 180)) - 
  2 * (Real.cos (40 * π / 180)) = 2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3797_379715


namespace NUMINAMATH_CALUDE_intersection_theorem_l3797_379711

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | x < 1}

theorem intersection_theorem : M ∩ (Set.univ \ N) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3797_379711


namespace NUMINAMATH_CALUDE_min_value_two_over_x_plus_x_over_two_min_value_achievable_l3797_379737

theorem min_value_two_over_x_plus_x_over_two (x : ℝ) (hx : x > 0) :
  2/x + x/2 ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > 0 ∧ 2/x + x/2 = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_two_over_x_plus_x_over_two_min_value_achievable_l3797_379737


namespace NUMINAMATH_CALUDE_vacant_seats_l3797_379758

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 60 / 100) : 
  (1 - filled_percentage) * total_seats = 240 := by
sorry

end NUMINAMATH_CALUDE_vacant_seats_l3797_379758


namespace NUMINAMATH_CALUDE_carls_cupcake_goal_l3797_379763

/-- Given Carl's cupcake selling goal and payment obligation, prove the number of cupcakes he must sell per day. -/
theorem carls_cupcake_goal (goal : ℕ) (days : ℕ) (payment : ℕ) (cupcakes_per_day : ℕ) 
    (h1 : goal = 96) 
    (h2 : days = 2) 
    (h3 : payment = 24) 
    (h4 : cupcakes_per_day * days = goal + payment) : 
  cupcakes_per_day = 60 := by
  sorry

#check carls_cupcake_goal

end NUMINAMATH_CALUDE_carls_cupcake_goal_l3797_379763


namespace NUMINAMATH_CALUDE_train_speed_proof_l3797_379739

/-- Proves that a train of given length, crossing a bridge of given length in a given time, has a specific speed. -/
theorem train_speed_proof (train_length bridge_length : ℝ) (crossing_time_minutes : ℝ) :
  train_length = 120 →
  bridge_length = 240 →
  crossing_time_minutes = 3 →
  (train_length + bridge_length) / (crossing_time_minutes * 60) = 2 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l3797_379739


namespace NUMINAMATH_CALUDE_abs_neg_two_neg_two_pow_zero_l3797_379736

-- Prove that the absolute value of -2 is equal to 2
theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

-- Prove that -2 raised to the power of 0 is equal to 1
theorem neg_two_pow_zero : (-2 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_neg_two_pow_zero_l3797_379736


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l3797_379724

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let total_distance := 2 * distance
  let completed_distance := distance + 0.1 * distance
  (completed_distance / total_distance) * 100 = 55 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l3797_379724


namespace NUMINAMATH_CALUDE_negative_root_implies_inequality_l3797_379794

theorem negative_root_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 3*a + 9 = 0 ∧ x < 0) → (a - 4) * (a - 5) > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_root_implies_inequality_l3797_379794


namespace NUMINAMATH_CALUDE_plumber_toilet_charge_l3797_379747

def sink_charge : ℕ := 30
def shower_charge : ℕ := 40

def job1_earnings (toilet_charge : ℕ) : ℕ := 3 * toilet_charge + 3 * sink_charge
def job2_earnings (toilet_charge : ℕ) : ℕ := 2 * toilet_charge + 5 * sink_charge
def job3_earnings (toilet_charge : ℕ) : ℕ := toilet_charge + 2 * shower_charge + 3 * sink_charge

def max_earnings : ℕ := 250

theorem plumber_toilet_charge :
  ∃ (toilet_charge : ℕ),
    (job1_earnings toilet_charge ≤ max_earnings) ∧
    (job2_earnings toilet_charge ≤ max_earnings) ∧
    (job3_earnings toilet_charge ≤ max_earnings) ∧
    ((job1_earnings toilet_charge = max_earnings) ∨
     (job2_earnings toilet_charge = max_earnings) ∨
     (job3_earnings toilet_charge = max_earnings)) ∧
    toilet_charge = 50 :=
by sorry

end NUMINAMATH_CALUDE_plumber_toilet_charge_l3797_379747


namespace NUMINAMATH_CALUDE_parabola_properties_line_parabola_intersection_l3797_379718

/-- Parabola C: y^2 = -4x -/
def parabola (x y : ℝ) : Prop := y^2 = -4*x

/-- Line l: y = kx - k + 2, passing through (1, 2) -/
def line (k x y : ℝ) : Prop := y = k*x - k + 2

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (-1, 0)

/-- Directrix of the parabola -/
def directrix (x : ℝ) : Prop := x = 1

/-- Distance from focus to directrix -/
def focus_directrix_distance : ℝ := 2

/-- Theorem about the parabola and its properties -/
theorem parabola_properties :
  (∀ x y, parabola x y → (focus.1 = -1 ∧ focus.2 = 0)) ∧
  (∀ x, directrix x ↔ x = 1) ∧
  focus_directrix_distance = 2 :=
sorry

/-- Theorem about the intersection of the line and parabola -/
theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, parabola x y ∧ line k x y →
    (k = 0 ∨ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) ↔
      (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2)) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) ↔
    (∃ p q : ℝ × ℝ, p ≠ q ∧ parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ line k p.1 p.2 ∧ line k q.1 q.2)) ∧
  (k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) ↔
    (∀ x y, ¬(parabola x y ∧ line k x y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_line_parabola_intersection_l3797_379718


namespace NUMINAMATH_CALUDE_complex_number_equality_l3797_379767

theorem complex_number_equality : (((1 : ℂ) + I) * ((3 : ℂ) + 4*I)) / I = (7 : ℂ) + I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3797_379767


namespace NUMINAMATH_CALUDE_divisor_problem_l3797_379713

theorem divisor_problem (n d : ℕ) : 
  (n % d = 3) → (n^2 % d = 3) → d = 6 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3797_379713


namespace NUMINAMATH_CALUDE_total_skips_is_450_l3797_379740

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of times Sally can skip a rock -/
def sally_skips : ℕ := 18

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for all three people -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped + sally_skips * rocks_skipped

theorem total_skips_is_450 : total_skips = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_skips_is_450_l3797_379740


namespace NUMINAMATH_CALUDE_only_hexagonal_prism_no_circular_cross_section_l3797_379719

-- Define the types of geometric shapes
inductive GeometricShape
  | Sphere
  | Cone
  | Cylinder
  | HexagonalPrism

-- Define a property for shapes that can have circular cross-sections
def has_circular_cross_section (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | GeometricShape.Cone => true
  | GeometricShape.Cylinder => true
  | GeometricShape.HexagonalPrism => false

-- Theorem statement
theorem only_hexagonal_prism_no_circular_cross_section :
  ∀ (shape : GeometricShape),
    ¬(has_circular_cross_section shape) ↔ shape = GeometricShape.HexagonalPrism :=
by
  sorry

end NUMINAMATH_CALUDE_only_hexagonal_prism_no_circular_cross_section_l3797_379719


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3797_379709

variable (a b : ℝ × ℝ)

theorem vector_sum_magnitude 
  (h1 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 4)
  (h2 : a.1 * b.1 + a.2 * b.2 = 1) :
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3797_379709


namespace NUMINAMATH_CALUDE_fraction_division_result_l3797_379701

theorem fraction_division_result : (3 / 8) / (5 / 9) = 27 / 40 := by sorry

end NUMINAMATH_CALUDE_fraction_division_result_l3797_379701


namespace NUMINAMATH_CALUDE_corrected_mean_l3797_379779

/-- Given a set of observations, calculate the corrected mean after fixing an error in one observation -/
theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n > 0 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  (corrected_sum / n) = 36.14 →
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 30 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3797_379779


namespace NUMINAMATH_CALUDE_solution_difference_l3797_379743

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 + 3*r - 18) = r + 3) →
  ((6 * s - 18) / (s^2 + 3*s - 18) = s + 3) →
  r ≠ s →
  r > s →
  r - s = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l3797_379743


namespace NUMINAMATH_CALUDE_total_weight_is_600_l3797_379774

/-- Proves that the total weight of Verna, Sherry, Jake, and Laura is 600 pounds given the specified conditions. -/
theorem total_weight_is_600 (haley_weight : ℝ) (verna_weight : ℝ) (sherry_weight : ℝ) (jake_weight : ℝ) (laura_weight : ℝ) : 
  haley_weight = 103 →
  verna_weight = haley_weight + 17 →
  verna_weight = sherry_weight / 2 →
  jake_weight = 3/5 * (haley_weight + verna_weight) →
  laura_weight = sherry_weight - jake_weight →
  verna_weight + sherry_weight + jake_weight + laura_weight = 600 := by
  sorry

#check total_weight_is_600

end NUMINAMATH_CALUDE_total_weight_is_600_l3797_379774


namespace NUMINAMATH_CALUDE_triangle_inequality_l3797_379762

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * (-a + b + c) + b^2 * (a - b + c) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3797_379762


namespace NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_l3797_379722

theorem abs_x_lt_2_sufficient_not_necessary :
  (∃ x : ℝ, (abs x < 2 → x^2 - x - 6 < 0) ∧ 
            ¬(x^2 - x - 6 < 0 → abs x < 2)) :=
sorry

end NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_l3797_379722


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3797_379744

/-- The area of a square with perimeter 24 is 36 -/
theorem square_area_from_perimeter : 
  ∀ s : Real, 
  (s > 0) → 
  (4 * s = 24) → 
  (s * s = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3797_379744


namespace NUMINAMATH_CALUDE_bagel_savings_l3797_379723

/-- The cost of an individual bagel in cents -/
def individual_cost : ℕ := 225

/-- The cost of a dozen bagels in dollars -/
def dozen_cost : ℕ := 24

/-- The number of bagels in a dozen -/
def bagels_per_dozen : ℕ := 12

/-- The savings per bagel in cents when buying a dozen -/
theorem bagel_savings : ℕ := by
  -- Convert individual cost to cents
  -- Calculate cost per bagel when buying a dozen
  -- Convert dozen cost to cents
  -- Calculate the difference
  sorry

end NUMINAMATH_CALUDE_bagel_savings_l3797_379723


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3797_379727

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom limit_condition : ∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, 
  |((f (x + 1) - f 1) / (2 * x)) - 3| < ε

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f 1) = 6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3797_379727


namespace NUMINAMATH_CALUDE_jamies_mean_is_88_5_l3797_379768

/-- Represents a test score series for two students -/
structure TestScores where
  scores : List Nat
  alex_count : Nat
  jamie_count : Nat
  alex_mean : Rat

/-- Calculates Jamie's mean score given the test scores -/
def jamies_mean (ts : TestScores) : Rat :=
  let total_sum := ts.scores.sum
  let alex_sum := ts.alex_mean * ts.alex_count
  let jamie_sum := total_sum - alex_sum
  jamie_sum / ts.jamie_count

/-- Theorem: Jamie's mean score is 88.5 given the conditions -/
theorem jamies_mean_is_88_5 (ts : TestScores) 
  (h1 : ts.scores = [75, 80, 85, 90, 92, 97])
  (h2 : ts.alex_count = 4)
  (h3 : ts.jamie_count = 2)
  (h4 : ts.alex_mean = 85.5)
  : jamies_mean ts = 88.5 := by
  sorry

end NUMINAMATH_CALUDE_jamies_mean_is_88_5_l3797_379768


namespace NUMINAMATH_CALUDE_ralph_peanuts_l3797_379738

/-- Represents the number of peanuts Ralph starts with -/
def initial_peanuts : ℕ := sorry

/-- Represents the number of peanuts Ralph loses -/
def lost_peanuts : ℕ := 59

/-- Represents the number of peanuts Ralph ends up with -/
def final_peanuts : ℕ := 15

/-- Theorem stating that Ralph started with 74 peanuts -/
theorem ralph_peanuts : initial_peanuts = 74 :=
by
  sorry

end NUMINAMATH_CALUDE_ralph_peanuts_l3797_379738


namespace NUMINAMATH_CALUDE_farm_has_six_cows_l3797_379730

/-- Represents the number of animals of each type on the farm -/
structure FarmAnimals where
  cows : ℕ
  chickens : ℕ
  sheep : ℕ

/-- Calculates the total number of legs for given farm animals -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  4 * animals.cows + 2 * animals.chickens + 4 * animals.sheep

/-- Calculates the total number of heads for given farm animals -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.cows + animals.chickens + animals.sheep

/-- Theorem stating that the farm with the given conditions has 6 cows -/
theorem farm_has_six_cows :
  ∃ (animals : FarmAnimals),
    totalLegs animals = 100 ∧
    totalLegs animals = 3 * totalHeads animals + 20 ∧
    animals.cows = 6 := by
  sorry


end NUMINAMATH_CALUDE_farm_has_six_cows_l3797_379730


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3797_379759

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ (1 - 1 / (2^n : ℚ) > 315 / 412) ∧ ∀ (m : ℕ), m > 0 ∧ m < n → 1 - 1 / (2^m : ℚ) ≤ 315 / 412 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3797_379759


namespace NUMINAMATH_CALUDE_tan_inequality_solution_set_l3797_379764

theorem tan_inequality_solution_set : 
  let S := {x : ℝ | ∃ k : ℤ, k * π - π / 3 < x ∧ x < k * π + Real.arctan 2}
  ∀ x : ℝ, x ∈ S ↔ -Real.sqrt 3 < Real.tan x ∧ Real.tan x < 2 :=
by sorry

end NUMINAMATH_CALUDE_tan_inequality_solution_set_l3797_379764


namespace NUMINAMATH_CALUDE_rental_miles_driven_l3797_379761

-- Define the rental parameters
def daily_rate : ℚ := 29
def mile_rate : ℚ := 0.08
def total_paid : ℚ := 46.12

-- Define the function to calculate miles driven
def miles_driven (daily_rate mile_rate total_paid : ℚ) : ℚ :=
  (total_paid - daily_rate) / mile_rate

-- Theorem statement
theorem rental_miles_driven :
  miles_driven daily_rate mile_rate total_paid = 214 := by
  sorry

end NUMINAMATH_CALUDE_rental_miles_driven_l3797_379761


namespace NUMINAMATH_CALUDE_sum_even_integers_minus15_to_5_l3797_379748

def sum_even_integers (a b : Int) : Int :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  let num_terms := (last_even - first_even) / 2 + 1
  (first_even + last_even) * num_terms / 2

theorem sum_even_integers_minus15_to_5 :
  sum_even_integers (-15) 5 = -50 := by
sorry

end NUMINAMATH_CALUDE_sum_even_integers_minus15_to_5_l3797_379748


namespace NUMINAMATH_CALUDE_three_digit_permutation_average_l3797_379775

def isValidNumber (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    n = 37 * (a + b + c)

def validSet : Set ℕ :=
  {111, 222, 333, 407, 444, 518, 555, 592, 666, 777, 888, 999}

theorem three_digit_permutation_average (n : ℕ) :
  isValidNumber n ↔ n ∈ validSet :=
sorry

end NUMINAMATH_CALUDE_three_digit_permutation_average_l3797_379775


namespace NUMINAMATH_CALUDE_min_k_for_A_cannot_win_l3797_379750

/-- Represents a position on the infinite hexagonal grid --/
structure HexPosition

/-- Represents the game state --/
structure GameState where
  board : HexPosition → Option Bool  -- True for A's counter, False for empty
  turn : Bool  -- True for A's turn, False for B's turn

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : HexPosition) : Prop := sorry

/-- Checks if there are k consecutive counters in a line --/
def consecutive_counters (state : GameState) (k : ℕ) : Prop := sorry

/-- Represents a valid move by player A --/
def valid_move_A (state : GameState) (p1 p2 : HexPosition) : Prop :=
  adjacent p1 p2 ∧ state.board p1 = none ∧ state.board p2 = none ∧ state.turn

/-- Represents a valid move by player B --/
def valid_move_B (state : GameState) (p : HexPosition) : Prop :=
  state.board p = some true ∧ ¬state.turn

/-- The main theorem stating that 6 is the minimum k for which A cannot win --/
theorem min_k_for_A_cannot_win :
  (∀ k < 6, ∃ (strategy : GameState → HexPosition × HexPosition),
    ∀ (counter_strategy : GameState → HexPosition),
      ∃ (n : ℕ), ∃ (final_state : GameState),
        consecutive_counters final_state k) ∧
  (∀ (strategy : GameState → HexPosition × HexPosition),
    ∃ (counter_strategy : GameState → HexPosition),
      ∀ (n : ℕ), ∀ (final_state : GameState),
        ¬consecutive_counters final_state 6) :=
sorry

end NUMINAMATH_CALUDE_min_k_for_A_cannot_win_l3797_379750


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l3797_379760

/-- Calculate the number of poles needed for a rectangular fence --/
def poles_needed (length width long_spacing short_spacing : ℕ) : ℕ :=
  let long_poles := (length / long_spacing + 1) * 2
  let short_poles := (width / short_spacing + 1) * 2
  long_poles + short_poles - 4

/-- Theorem: The number of poles needed for the given rectangular plot is 70 --/
theorem rectangular_plot_poles :
  poles_needed 90 70 4 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l3797_379760


namespace NUMINAMATH_CALUDE_investment_interest_proof_l3797_379712

/-- Calculates the simple interest earned on an investment. -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest_proof (total_investment : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (investment1 : ℝ) (time : ℝ) 
  (h1 : total_investment = 15000)
  (h2 : rate1 = 0.06)
  (h3 : rate2 = 0.075)
  (h4 : investment1 = 8200)
  (h5 : time = 1)
  (h6 : investment1 ≤ total_investment) :
  simple_interest investment1 rate1 time + 
  simple_interest (total_investment - investment1) rate2 time = 1002 := by
  sorry

#check investment_interest_proof

end NUMINAMATH_CALUDE_investment_interest_proof_l3797_379712


namespace NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_integers_l3797_379707

theorem sum_of_six_consecutive_odd_integers (S : ℤ) :
  (∃ n : ℤ, S = 6*n + 30 ∧ Odd n) ↔ (∃ k : ℤ, S - 30 = 6*k ∧ Odd k) :=
sorry

end NUMINAMATH_CALUDE_sum_of_six_consecutive_odd_integers_l3797_379707


namespace NUMINAMATH_CALUDE_basketball_weight_l3797_379756

theorem basketball_weight (basketball_weight bicycle_weight : ℝ) 
  (h1 : 9 * basketball_weight = 6 * bicycle_weight)
  (h2 : 4 * bicycle_weight = 120) : 
  basketball_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_basketball_weight_l3797_379756


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_prove_mean_equality_implies_y_value_l3797_379745

theorem mean_equality_implies_y_value : ℝ → Prop :=
  fun y =>
    (6 + 9 + 18) / 3 = (12 + y) / 2 →
    y = 10

-- The proof is omitted
theorem prove_mean_equality_implies_y_value :
  ∃ y : ℝ, mean_equality_implies_y_value y :=
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_prove_mean_equality_implies_y_value_l3797_379745


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3797_379765

/-- The sum of the intercepts of the line 2x - 3y + 6 = 0 on the coordinate axes is -1 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (2 * x - 3 * y + 6 = 0) → 
  (∃ x_intercept y_intercept : ℝ, 
    (2 * x_intercept + 6 = 0) ∧ 
    (-3 * y_intercept + 6 = 0) ∧ 
    (x_intercept + y_intercept = -1)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3797_379765


namespace NUMINAMATH_CALUDE_flagpole_height_l3797_379777

/-- Given a tree and a flagpole casting shadows at the same time, 
    we can determine the height of the flagpole. -/
theorem flagpole_height 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : tree_height = 3.6) 
  (h2 : tree_shadow = 0.6) 
  (h3 : flagpole_shadow = 1.5) : 
  ∃ (flagpole_height : ℝ), flagpole_height = 9 := by
sorry

end NUMINAMATH_CALUDE_flagpole_height_l3797_379777


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_sum_l3797_379790

/-- Represents a rectangular parallelepiped with dimensions l, w, and h -/
structure Parallelepiped where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a parallelepiped -/
def volume_extended_parallelepiped (p : Parallelepiped) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def relatively_prime (a b : ℕ) : Prop := sorry

theorem extended_parallelepiped_volume_sum (m n p : ℕ) :
  (∃ (parallelepiped : Parallelepiped),
    parallelepiped.l = 3 ∧
    parallelepiped.w = 4 ∧
    parallelepiped.h = 5 ∧
    volume_extended_parallelepiped parallelepiped = (m + n * Real.pi) / p ∧
    m > 0 ∧ n > 0 ∧ p > 0 ∧
    relatively_prime n p) →
  m + n + p = 505 := by sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_sum_l3797_379790


namespace NUMINAMATH_CALUDE_digit_B_value_l3797_379700

theorem digit_B_value (A B : ℕ) (h : 100 * A + 10 * B + 2 - 41 = 591) : B = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_B_value_l3797_379700


namespace NUMINAMATH_CALUDE_circle_largest_area_l3797_379749

-- Define the shapes
def triangle_area (side : Real) (angle1 : Real) (angle2 : Real) : Real :=
  -- Area calculation for triangle
  sorry

def rhombus_area (d1 : Real) (d2 : Real) (angle : Real) : Real :=
  -- Area calculation for rhombus
  sorry

def circle_area (radius : Real) : Real :=
  -- Area calculation for circle
  sorry

def square_area (diagonal : Real) : Real :=
  -- Area calculation for square
  sorry

-- Theorem statement
theorem circle_largest_area :
  let triangle_a := triangle_area (Real.sqrt 2) (60 * π / 180) (45 * π / 180)
  let rhombus_a := rhombus_area (Real.sqrt 2) (Real.sqrt 3) (75 * π / 180)
  let circle_a := circle_area 1
  let square_a := square_area 2.5
  circle_a > triangle_a ∧ circle_a > rhombus_a ∧ circle_a > square_a :=
by sorry


end NUMINAMATH_CALUDE_circle_largest_area_l3797_379749


namespace NUMINAMATH_CALUDE_evaluate_expression_l3797_379731

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 3)^2 = 144 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3797_379731


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_range_of_linear_combination_l3797_379772

-- Part 1
theorem compare_quadratic_expressions (a : ℝ) : 
  (a - 2) * (a - 6) < (a - 3) * (a - 5) := by sorry

-- Part 2
theorem range_of_linear_combination (x y : ℝ) 
  (hx : -2 < x ∧ x < 1) (hy : 1 < y ∧ y < 2) : 
  -6 < 2 * x - y ∧ 2 * x - y < 1 := by sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_range_of_linear_combination_l3797_379772


namespace NUMINAMATH_CALUDE_remainder_three_to_forty_plus_five_mod_five_l3797_379780

theorem remainder_three_to_forty_plus_five_mod_five :
  (3^40 + 5) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_forty_plus_five_mod_five_l3797_379780


namespace NUMINAMATH_CALUDE_bridge_length_l3797_379751

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 255 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3797_379751


namespace NUMINAMATH_CALUDE_dartboard_angle_l3797_379778

theorem dartboard_angle (P : ℝ) (θ : ℝ) : 
  P = 1/8 → θ = P * 360 → θ = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_l3797_379778


namespace NUMINAMATH_CALUDE_inequality_proof_l3797_379710

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3797_379710


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_true_implication_l3797_379714

theorem sufficient_condition_implies_true_implication (p q : Prop) :
  (p → q) → (p → q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_true_implication_l3797_379714


namespace NUMINAMATH_CALUDE_negation_of_positive_square_is_false_l3797_379728

theorem negation_of_positive_square_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) = False :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_is_false_l3797_379728


namespace NUMINAMATH_CALUDE_find_number_l3797_379797

theorem find_number : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3797_379797


namespace NUMINAMATH_CALUDE_convex_broken_line_in_triangle_l3797_379735

/-- A convex broken line in 2D space -/
structure ConvexBrokenLine where
  points : List (Real × Real)
  is_convex : sorry
  length : Real

/-- An equilateral triangle in 2D space -/
structure EquilateralTriangle where
  center : Real × Real
  side_length : Real

/-- A function to check if a broken line is enclosed within a triangle -/
def is_enclosed (line : ConvexBrokenLine) (triangle : EquilateralTriangle) : Prop :=
  sorry

theorem convex_broken_line_in_triangle 
  (line : ConvexBrokenLine) 
  (triangle : EquilateralTriangle) : 
  line.length = 1 → 
  triangle.side_length = 1 → 
  is_enclosed line triangle :=
sorry

end NUMINAMATH_CALUDE_convex_broken_line_in_triangle_l3797_379735


namespace NUMINAMATH_CALUDE_derivative_of_f_l3797_379754

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  (f ∘ Real.cos ∘ (fun x => 2 * x)) x = 1 - 2 * (Real.sin x) ^ 2 → 
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3797_379754


namespace NUMINAMATH_CALUDE_quadratic_solution_l3797_379725

theorem quadratic_solution (x a : ℝ) : x = 3 ∧ x^2 = a → a = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3797_379725


namespace NUMINAMATH_CALUDE_congruence_problem_l3797_379704

theorem congruence_problem (x y n : ℤ) : 
  x ≡ 45 [ZMOD 60] →
  y ≡ 98 [ZMOD 60] →
  n ∈ Finset.Icc 150 210 →
  (x - y ≡ n [ZMOD 60]) ↔ n = 187 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l3797_379704


namespace NUMINAMATH_CALUDE_wendy_distance_difference_l3797_379786

/-- The distance Wendy ran in miles -/
def distance_run : ℝ := 19.83

/-- The distance Wendy walked in miles -/
def distance_walked : ℝ := 9.17

/-- The difference between the distance Wendy ran and walked -/
def distance_difference : ℝ := distance_run - distance_walked

theorem wendy_distance_difference :
  distance_difference = 10.66 := by sorry

end NUMINAMATH_CALUDE_wendy_distance_difference_l3797_379786


namespace NUMINAMATH_CALUDE_equation_solution_l3797_379703

theorem equation_solution : ∃ x : ℚ, x - (x + 2) / 2 = (2 * x - 1) / 3 - 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3797_379703


namespace NUMINAMATH_CALUDE_sum_of_abc_l3797_379782

theorem sum_of_abc (a b c : ℝ) 
  (h1 : a^2 - 2*b = -2) 
  (h2 : b^2 + 6*c = 7) 
  (h3 : c^2 - 8*a = -31) : 
  a + b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_l3797_379782


namespace NUMINAMATH_CALUDE_work_completion_time_l3797_379746

/-- The number of days y needs to finish the work alone -/
def y_days : ℕ := 24

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 18

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 36

theorem work_completion_time :
  x_days = 36 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3797_379746


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l3797_379742

/-- The capacity of a bucket in the first scenario, in litres. -/
def first_bucket_capacity : ℝ := 13.5

/-- The number of buckets required to fill the tank in the first scenario. -/
def first_scenario_buckets : ℕ := 28

/-- The number of buckets required to fill the tank in the second scenario. -/
def second_scenario_buckets : ℕ := 42

/-- The capacity of a bucket in the second scenario, in litres. -/
def second_bucket_capacity : ℝ := 9

theorem bucket_capacity_proof :
  first_bucket_capacity * first_scenario_buckets =
  second_bucket_capacity * second_scenario_buckets := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_proof_l3797_379742


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3797_379741

theorem arithmetic_mean_problem (x : ℝ) : 
  (8 + 16 + 21 + 7 + x) / 5 = 12 → x = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3797_379741


namespace NUMINAMATH_CALUDE_average_and_square_difference_l3797_379753

theorem average_and_square_difference (y : ℝ) : 
  (45 + y) / 2 = 50 → (y - 45)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_average_and_square_difference_l3797_379753


namespace NUMINAMATH_CALUDE_percentage_not_sold_is_25_percent_l3797_379733

-- Define the initial stock and daily sales
def initial_stock : ℕ := 600
def monday_sales : ℕ := 25
def tuesday_sales : ℕ := 70
def wednesday_sales : ℕ := 100
def thursday_sales : ℕ := 110
def friday_sales : ℕ := 145

-- Define the total sales
def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

-- Define the number of bags not sold
def bags_not_sold : ℕ := initial_stock - total_sales

-- Theorem to prove
theorem percentage_not_sold_is_25_percent :
  (bags_not_sold : ℚ) / (initial_stock : ℚ) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_percentage_not_sold_is_25_percent_l3797_379733


namespace NUMINAMATH_CALUDE_savings_calculation_l3797_379769

def num_machines : ℕ := 25
def bearings_per_machine : ℕ := 45
def regular_price : ℚ := 125/100
def sale_price : ℚ := 80/100
def discount_first_20 : ℚ := 25/100
def discount_remaining : ℚ := 35/100
def first_batch : ℕ := 20

def total_bearings : ℕ := num_machines * bearings_per_machine

def regular_total_cost : ℚ := (total_bearings : ℚ) * regular_price

def sale_cost_before_discount : ℚ := (total_bearings : ℚ) * sale_price

def first_batch_bearings : ℕ := first_batch * bearings_per_machine
def remaining_bearings : ℕ := total_bearings - first_batch_bearings

def first_batch_cost : ℚ := (first_batch_bearings : ℚ) * sale_price * (1 - discount_first_20)
def remaining_cost : ℚ := (remaining_bearings : ℚ) * sale_price * (1 - discount_remaining)

def total_discounted_cost : ℚ := first_batch_cost + remaining_cost

theorem savings_calculation : 
  regular_total_cost - total_discounted_cost = 74925/100 :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l3797_379769


namespace NUMINAMATH_CALUDE_centroid_distance_sum_l3797_379702

/-- Given a triangle DEF with centroid G, prove that if the sum of squared distances
    from G to the vertices is 90, then the sum of squared side lengths is 270. -/
theorem centroid_distance_sum (D E F G : ℝ × ℝ) : 
  (G = ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3)) →  -- G is the centroid
  ((G.1 - D.1)^2 + (G.2 - D.2)^2 + 
   (G.1 - E.1)^2 + (G.2 - E.2)^2 + 
   (G.1 - F.1)^2 + (G.2 - F.2)^2 = 90) →  -- Sum of squared distances from G to vertices is 90
  ((D.1 - E.1)^2 + (D.2 - E.2)^2 + 
   (D.1 - F.1)^2 + (D.2 - F.2)^2 + 
   (E.1 - F.1)^2 + (E.2 - F.2)^2 = 270)  -- Sum of squared side lengths is 270
:= by sorry

end NUMINAMATH_CALUDE_centroid_distance_sum_l3797_379702


namespace NUMINAMATH_CALUDE_isabel_earnings_l3797_379771

def bead_necklaces : ℕ := 3
def gemstone_necklaces : ℕ := 3
def bead_price : ℚ := 4
def gemstone_price : ℚ := 8
def sales_tax_rate : ℚ := 0.05
def discount_rate : ℚ := 0.10

def total_earned : ℚ :=
  let total_before_tax := bead_necklaces * bead_price + gemstone_necklaces * gemstone_price
  let tax_amount := total_before_tax * sales_tax_rate
  let total_after_tax := total_before_tax + tax_amount
  let discount_amount := total_after_tax * discount_rate
  total_after_tax - discount_amount

theorem isabel_earnings : total_earned = 34.02 := by
  sorry

end NUMINAMATH_CALUDE_isabel_earnings_l3797_379771


namespace NUMINAMATH_CALUDE_sqrt_floor_equality_l3797_379729

theorem sqrt_floor_equality (n : ℕ) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ :=
by sorry

end NUMINAMATH_CALUDE_sqrt_floor_equality_l3797_379729


namespace NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l3797_379791

/-- A letter lock with a given number of rings and letters per ring -/
structure LetterLock where
  num_rings : ℕ
  letters_per_ring : ℕ

/-- The number of distinct unsuccessful attempts for a given letter lock -/
def unsuccessfulAttempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem: For a letter lock with 3 rings and 6 letters per ring, 
    the number of distinct unsuccessful attempts is 215 -/
theorem letter_lock_unsuccessful_attempts :
  ∃ (lock : LetterLock), lock.num_rings = 3 ∧ lock.letters_per_ring = 6 ∧ 
  unsuccessfulAttempts lock = 215 := by
  sorry

end NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l3797_379791


namespace NUMINAMATH_CALUDE_scale_division_l3797_379755

/-- Given a scale of length 80 inches divided into 5 equal parts, 
    prove that the length of each part is 16 inches. -/
theorem scale_division (total_length : ℕ) (num_parts : ℕ) (part_length : ℕ) 
  (h1 : total_length = 80) 
  (h2 : num_parts = 5) 
  (h3 : part_length * num_parts = total_length) : 
  part_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l3797_379755


namespace NUMINAMATH_CALUDE_staircase_extension_l3797_379752

/-- Calculates the number of additional toothpicks needed to extend a staircase -/
def additional_toothpicks (initial_steps : ℕ) (final_steps : ℕ) (initial_toothpicks : ℕ) : ℕ :=
  let base_increase := initial_toothpicks / initial_steps + 2
  let num_new_steps := final_steps - initial_steps
  (num_new_steps * (2 * base_increase + (num_new_steps - 1) * 2)) / 2

theorem staircase_extension :
  additional_toothpicks 4 7 28 = 42 :=
by sorry

end NUMINAMATH_CALUDE_staircase_extension_l3797_379752


namespace NUMINAMATH_CALUDE_ratio_composition_l3797_379783

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 2 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_composition_l3797_379783


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l3797_379717

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 120

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of 'A's in "BANANA" -/
def num_a : ℕ := 3

/-- Theorem stating that the number of unique arrangements of "BANANA" is correct -/
theorem banana_arrangement_count :
  banana_arrangements = (total_letters.factorial) / (num_a.factorial) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l3797_379717


namespace NUMINAMATH_CALUDE_checkerboard_chips_l3797_379770

/-- The total number of chips on an n × n checkerboard where each square (i, j) has |i - j| chips -/
def total_chips (n : ℕ) : ℕ := n * (n + 1) * (n - 1) / 3

/-- Theorem stating that if the total number of chips is 2660, then n = 20 -/
theorem checkerboard_chips (n : ℕ) : total_chips n = 2660 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_chips_l3797_379770


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3797_379792

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 20) * (x / 80) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3797_379792


namespace NUMINAMATH_CALUDE_factorization_equality_l3797_379788

theorem factorization_equality (x y : ℝ) : 6*x^2*y - 3*x*y = 3*x*y*(2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3797_379788


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l3797_379773

theorem conic_section_eccentricity (m : ℝ) : 
  (m^2 = 5 * (16/5)) → 
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
   ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
   ((x : ℝ) → (y : ℝ) → x^2 / m + y^2 = 1 → 
    (e = c / a ∧ ((m > 0 → a^2 - b^2 = c^2) ∧ (m < 0 → b^2 - a^2 = c^2))))) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l3797_379773


namespace NUMINAMATH_CALUDE_product_inequality_l3797_379721

theorem product_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3797_379721


namespace NUMINAMATH_CALUDE_box_height_proof_l3797_379796

/-- Proves that a box with given dimensions and cube requirements has a specific height -/
theorem box_height_proof (length width : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) (height : ℝ) : 
  length = 10 →
  width = 13 →
  cube_volume = 5 →
  min_cubes = 130 →
  height = (min_cubes : ℝ) * cube_volume / (length * width) →
  height = 5 := by
sorry

end NUMINAMATH_CALUDE_box_height_proof_l3797_379796


namespace NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l3797_379720

/-- A function that checks if a natural number is a nine-digit number -/
def isNineDigit (n : ℕ) : Prop :=
  100000000 ≤ n ∧ n < 1000000000

/-- A function that checks if a natural number ends with 5 -/
def endsWithFive (n : ℕ) : Prop :=
  n % 10 = 5

/-- A function that checks if a natural number contains each of the digits 1-9 exactly once -/
def containsEachDigitOnce (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] →
    (∃! p : ℕ, p < 9 ∧ (n / 10^p) % 10 = d)

/-- The main theorem stating that no number satisfying the given conditions is a perfect square -/
theorem no_perfect_square_with_conditions :
  ¬∃ n : ℕ, isNineDigit n ∧ endsWithFive n ∧ containsEachDigitOnce n ∧ ∃ m : ℕ, n = m^2 := by
  sorry


end NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l3797_379720


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3797_379757

/-- The common ratio of the geometric sequence representing 0.72̄ -/
def q : ℚ := 1 / 100

/-- The first term of the geometric sequence representing 0.72̄ -/
def a₁ : ℚ := 72 / 100

/-- The sum of the infinite geometric series representing 0.72̄ -/
def S : ℚ := a₁ / (1 - q)

/-- The repeating decimal 0.72̄ as a rational number -/
def repeating_decimal : ℚ := 8 / 11

theorem repeating_decimal_equals_fraction : S = repeating_decimal := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3797_379757


namespace NUMINAMATH_CALUDE_problem_statement_l3797_379795

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : y = x / (3 * x + 1)) :
  (x - y + 3 * x * y) / (x * y) = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3797_379795


namespace NUMINAMATH_CALUDE_problem_statement_l3797_379708

theorem problem_statement (x y z a : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : x^2 - 1/y = y^2 - 1/z ∧ y^2 - 1/z = z^2 - 1/x ∧ z^2 - 1/x = a) :
  (x + y + z) * x * y * z = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3797_379708
