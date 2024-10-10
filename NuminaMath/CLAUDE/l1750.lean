import Mathlib

namespace factorial_square_root_square_l1750_175063

-- Definition of factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_square_root_square :
  (((factorial 5 + 1) * factorial 4).sqrt ^ 2 : ℕ) = 2904 := by
  sorry

end factorial_square_root_square_l1750_175063


namespace ninth_term_is_negative_256_l1750_175022

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℤ, a (n + 1) = a n * q
  prod_condition : a 2 * a 5 = -32
  sum_condition : a 3 + a 4 = 4

/-- The theorem stating that a₉ = -256 for the given geometric sequence -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

end ninth_term_is_negative_256_l1750_175022


namespace clock_hand_alignments_in_day_l1750_175004

/-- Represents a traditional 12-hour analog clock -/
structure AnalogClock where
  hourHand : ℝ
  minuteHand : ℝ
  secondHand : ℝ

/-- The number of times the clock hands align in a 12-hour period -/
def alignmentsIn12Hours : ℕ := 1

/-- The number of 12-hour periods in a day -/
def periodsInDay : ℕ := 2

/-- Theorem: The number of times all three hands align in a 24-hour period is 2 -/
theorem clock_hand_alignments_in_day :
  alignmentsIn12Hours * periodsInDay = 2 := by sorry

end clock_hand_alignments_in_day_l1750_175004


namespace f_2019_equals_2016_l1750_175030

def f : ℕ → ℕ
| x => if x ≤ 2015 then x + 2 else f (x - 5)

theorem f_2019_equals_2016 : f 2019 = 2016 := by
  sorry

end f_2019_equals_2016_l1750_175030


namespace selection_ways_l1750_175027

/-- The number of students in the group -/
def num_students : ℕ := 5

/-- The number of positions to be filled (representative and vice-president) -/
def num_positions : ℕ := 2

/-- Theorem: The number of ways to select one representative and one vice-president
    from a group of 5 students is equal to 20 -/
theorem selection_ways : (num_students * (num_students - 1)) = 20 := by
  sorry

end selection_ways_l1750_175027


namespace binary_divisible_by_seven_l1750_175034

def K (x y z : Fin 2) : ℕ :=
  524288 + 131072 + 65536 + 16384 + 4096 + 1024 + 256 + 64 * y.val + 32 * x.val + 16 * z.val + 8 + 2

theorem binary_divisible_by_seven (x y z : Fin 2) :
  K x y z % 7 = 0 → x = 0 ∧ y = 1 ∧ z = 0 := by
  sorry

end binary_divisible_by_seven_l1750_175034


namespace contrapositive_equivalence_l1750_175006

theorem contrapositive_equivalence (f : ℝ → ℝ) (a : ℝ) :
  (a ≥ (1/2) → ∀ x ≥ 0, f x ≥ 0) ↔
  (∃ x ≥ 0, f x < 0 → a < (1/2)) :=
sorry

end contrapositive_equivalence_l1750_175006


namespace jerrie_carrie_difference_l1750_175093

/-- The number of sit-ups Barney can perform in one minute -/
def barney_rate : ℕ := 45

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_rate : ℕ := 2 * barney_rate

/-- The number of minutes Barney performs sit-ups -/
def barney_time : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_time : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_time : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 510

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_rate : ℕ := (total_situps - (barney_rate * barney_time + carrie_rate * carrie_time)) / jerrie_time

theorem jerrie_carrie_difference :
  jerrie_rate - carrie_rate = 5 :=
sorry

end jerrie_carrie_difference_l1750_175093


namespace fraction_simplification_l1750_175095

theorem fraction_simplification : (1 : ℚ) / 462 + 19 / 42 = 5 / 11 := by
  sorry

end fraction_simplification_l1750_175095


namespace hyperbola_asymptotes_l1750_175079

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def real_axis_length (a : ℝ) : ℝ := 2 * a

def imaginary_axis_length (b : ℝ) : ℝ := 2 * b

def asymptote_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : real_axis_length a = 2 * Real.sqrt 2)
  (h4 : imaginary_axis_length b = 2) :
  ∀ (x y : ℝ), asymptote_equation a b x y ↔ y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x :=
sorry

end hyperbola_asymptotes_l1750_175079


namespace tens_digit_of_23_pow_2057_l1750_175086

theorem tens_digit_of_23_pow_2057 : ∃ n : ℕ, 23^2057 ≡ 60 + n [ZMOD 100] ∧ n < 10 :=
sorry

end tens_digit_of_23_pow_2057_l1750_175086


namespace equation_proof_l1750_175029

theorem equation_proof : (5568 / 87 : ℝ)^(1/3) + (72 * 2 : ℝ)^(1/2) = (256 : ℝ)^(1/2) := by
  sorry

end equation_proof_l1750_175029


namespace roots_equal_magnitude_opposite_sign_l1750_175019

theorem roots_equal_magnitude_opposite_sign (a b c m : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = -x ∧
    (x^2 - b*x) / (a*x - c) = (m - 1) / (m + 1) ∧
    (y^2 - b*y) / (a*y - c) = (m - 1) / (m + 1)) →
  m = (a - b) / (a + b) :=
by sorry

end roots_equal_magnitude_opposite_sign_l1750_175019


namespace tire_circumference_l1750_175038

/-- Calculates the circumference of a car tire given the car's speed and tire rotation rate. -/
theorem tire_circumference (speed : ℝ) (rotations : ℝ) : 
  speed = 168 → rotations = 400 → (speed * 1000 / 60) / rotations = 7 := by
  sorry

end tire_circumference_l1750_175038


namespace house_transaction_loss_l1750_175072

theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  gain_percent = 0.20 → 
  initial_value * (1 - loss_percent) * (1 + gain_percent) - initial_value = 240 := by
  sorry

end house_transaction_loss_l1750_175072


namespace primitive_root_extension_l1750_175017

theorem primitive_root_extension (p : ℕ) (x : ℤ) (h_p : Nat.Prime p) (h_p_odd : p % 2 = 1)
  (h_primitive_root_p2 : IsPrimitiveRoot x (p^2)) :
  ∀ α : ℕ, α ≥ 2 → IsPrimitiveRoot x (p^α) :=
by sorry

end primitive_root_extension_l1750_175017


namespace total_glasses_displayed_l1750_175058

/-- Represents the number of cupboards of each type -/
def num_tall_cupboards : ℕ := 2
def num_wide_cupboards : ℕ := 2
def num_narrow_cupboards : ℕ := 2

/-- Represents the capacity of each type of cupboard -/
def tall_cupboard_capacity : ℕ := 30
def wide_cupboard_capacity : ℕ := 2 * tall_cupboard_capacity
def narrow_cupboard_capacity : ℕ := 45

/-- Represents the number of shelves in a narrow cupboard -/
def shelves_per_narrow_cupboard : ℕ := 3

/-- Represents the number of broken shelves -/
def broken_shelves : ℕ := 1

/-- Theorem stating the total number of glasses displayed -/
theorem total_glasses_displayed : 
  num_tall_cupboards * tall_cupboard_capacity +
  num_wide_cupboards * wide_cupboard_capacity +
  (num_narrow_cupboards * narrow_cupboard_capacity - 
   broken_shelves * (narrow_cupboard_capacity / shelves_per_narrow_cupboard)) = 255 := by
  sorry

end total_glasses_displayed_l1750_175058


namespace sin_theta_value_l1750_175023

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) 
  (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : Real.sin θ = 3/4 := by
  sorry

end sin_theta_value_l1750_175023


namespace area_of_right_isosceles_triangle_l1750_175082

/-- A right-angled isosceles triangle with the sum of the areas of squares on its sides equal to 72 -/
structure RightIsoscelesTriangle where
  /-- The length of each of the two equal sides -/
  side : ℝ
  /-- The sum of the areas of squares on the sides is 72 -/
  sum_of_squares : side^2 + side^2 + (2 * side^2) = 72

/-- The area of a right-angled isosceles triangle with the given property is 9 -/
theorem area_of_right_isosceles_triangle (t : RightIsoscelesTriangle) : 
  (1/2 : ℝ) * t.side * t.side = 9 := by
  sorry

end area_of_right_isosceles_triangle_l1750_175082


namespace laura_drives_234_miles_per_week_l1750_175068

/-- Calculates the total miles driven per week based on Laura's travel habits -/
def total_miles_per_week (school_round_trip : ℕ) (supermarket_extra : ℕ) (gym_distance : ℕ) (friend_distance : ℕ) : ℕ :=
  let school_miles := school_round_trip * 5
  let supermarket_miles := (school_round_trip + 2 * supermarket_extra) * 2
  let gym_miles := 2 * gym_distance * 3
  let friend_miles := 2 * friend_distance
  school_miles + supermarket_miles + gym_miles + friend_miles

/-- Theorem stating that Laura drives 234 miles per week -/
theorem laura_drives_234_miles_per_week :
  total_miles_per_week 20 10 5 12 = 234 := by
  sorry

end laura_drives_234_miles_per_week_l1750_175068


namespace largest_integer_quadratic_inequality_six_satisfies_inequality_seven_does_not_satisfy_inequality_l1750_175013

theorem largest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 9*n + 14 < 0 → n ≤ 6 :=
by
  sorry

theorem six_satisfies_inequality :
  (6 : ℤ)^2 - 9*6 + 14 < 0 :=
by
  sorry

theorem seven_does_not_satisfy_inequality :
  (7 : ℤ)^2 - 9*7 + 14 ≥ 0 :=
by
  sorry

end largest_integer_quadratic_inequality_six_satisfies_inequality_seven_does_not_satisfy_inequality_l1750_175013


namespace vowel_probability_is_three_thirteenths_l1750_175052

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The set of vowels including W -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'W'}

/-- The probability of selecting a vowel from the alphabet -/
def vowel_probability : ℚ := (Finset.card vowels : ℚ) / alphabet_size

theorem vowel_probability_is_three_thirteenths : 
  vowel_probability = 3 / 13 := by sorry

end vowel_probability_is_three_thirteenths_l1750_175052


namespace no_common_root_for_rational_coefficients_l1750_175024

theorem no_common_root_for_rational_coefficients :
  ∀ (a b : ℚ), ¬∃ (x : ℂ), (x^5 - x - 1 = 0) ∧ (x^2 + a*x + b = 0) :=
by sorry

end no_common_root_for_rational_coefficients_l1750_175024


namespace product_abcd_l1750_175015

theorem product_abcd (a b c d : ℚ) 
  (eq1 : 4*a + 5*b + 7*c + 9*d = 82)
  (eq2 : d + c = 2*b)
  (eq3 : 2*b + 2*c = 3*a)
  (eq4 : c - 2 = d) :
  a * b * c * d = 276264960 / 14747943 := by
sorry

end product_abcd_l1750_175015


namespace sector_radius_l1750_175011

/-- Given a circular sector with area 240π and arc length 20π, prove that its radius is 24. -/
theorem sector_radius (A : ℝ) (L : ℝ) (r : ℝ) : 
  A = 240 * Real.pi → L = 20 * Real.pi → A = (1/2) * r^2 * (L/r) → r = 24 := by
sorry

end sector_radius_l1750_175011


namespace imaginary_iff_m_neq_zero_pure_imaginary_iff_m_eq_two_or_three_not_in_second_quadrant_l1750_175040

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := (m^2 - 5*m + 6) - 3*m*Complex.I

-- Theorem 1: z is an imaginary number iff m ≠ 0
theorem imaginary_iff_m_neq_zero (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m ≠ 0 :=
sorry

-- Theorem 2: z is a pure imaginary number iff m = 2 or m = 3
theorem pure_imaginary_iff_m_eq_two_or_three (m : ℝ) :
  (z m).re = 0 ↔ m = 2 ∨ m = 3 :=
sorry

-- Theorem 3: z cannot be in the second quadrant for any real m
theorem not_in_second_quadrant (m : ℝ) :
  ¬((z m).re < 0 ∧ (z m).im > 0) :=
sorry

end imaginary_iff_m_neq_zero_pure_imaginary_iff_m_eq_two_or_three_not_in_second_quadrant_l1750_175040


namespace min_value_trig_expression_equality_condition_l1750_175026

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ ≥ 3 * Real.sqrt 2 :=
by sorry

theorem equality_condition (θ : Real) (h : θ = π / 4) :
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ = 3 * Real.sqrt 2 :=
by sorry

end min_value_trig_expression_equality_condition_l1750_175026


namespace complex_power_of_four_l1750_175097

theorem complex_power_of_four :
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 =
  -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_of_four_l1750_175097


namespace midpoint_coordinate_sum_l1750_175057

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, -2) and (-4, 10) is 6. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := -2
  let x₂ : ℝ := -4
  let y₂ : ℝ := 10
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 6 := by sorry

end midpoint_coordinate_sum_l1750_175057


namespace sticker_distribution_l1750_175043

/-- The number of ways to distribute indistinguishable objects into distinguishable groups -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 1365 ways to distribute 11 indistinguishable stickers into 5 distinguishable sheets of paper -/
theorem sticker_distribution : distribute 11 5 = 1365 := by
  sorry

end sticker_distribution_l1750_175043


namespace square_difference_characterization_l1750_175060

theorem square_difference_characterization (N : ℕ+) :
  (∃ k : ℕ, (2^N.val : ℕ) - 2 * N.val = k^2) ↔ N = 1 ∨ N = 2 := by
  sorry

end square_difference_characterization_l1750_175060


namespace question_differentiates_inhabitants_l1750_175056

-- Define the types of inhabitants
inductive InhabitantType
  | TruthTeller
  | Liar

-- Define the possible answers
inductive Answer
  | Yes
  | No

-- Function to determine how an inhabitant would answer the question
def answer_question (inhabitant_type : InhabitantType) : Answer :=
  match inhabitant_type with
  | InhabitantType.TruthTeller => Answer.No
  | InhabitantType.Liar => Answer.Yes

-- Theorem stating that the question can differentiate between truth-tellers and liars
theorem question_differentiates_inhabitants :
  ∀ (t : InhabitantType),
    (t = InhabitantType.TruthTeller ↔ answer_question t = Answer.No) ∧
    (t = InhabitantType.Liar ↔ answer_question t = Answer.Yes) :=
by sorry

end question_differentiates_inhabitants_l1750_175056


namespace sqrt_six_times_sqrt_two_equals_two_sqrt_three_l1750_175005

theorem sqrt_six_times_sqrt_two_equals_two_sqrt_three :
  Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_six_times_sqrt_two_equals_two_sqrt_three_l1750_175005


namespace sum_of_sequences_l1750_175031

/-- Sum of arithmetic sequence with 5 terms -/
def arithmetic_sum (a₁ : ℕ) : ℕ := a₁ + (a₁ + 10) + (a₁ + 20) + (a₁ + 30) + (a₁ + 40)

/-- The sum of two specific arithmetic sequences equals 270 -/
theorem sum_of_sequences : arithmetic_sum 3 + arithmetic_sum 11 = 270 := by
  sorry

end sum_of_sequences_l1750_175031


namespace asymptote_sum_l1750_175096

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x, x^3 + A*x^2 + B*x + C = (x + 3)*(x - 1)*(x - 3)) → A + B + C = 15 :=
by sorry

end asymptote_sum_l1750_175096


namespace quadratic_inequality_solution_parametric_quadratic_inequality_solution_l1750_175012

-- Define the quadratic function
def f (x : ℝ) := -x^2 - 2*x + 3

-- Define the parametric quadratic function
def g (a : ℝ) (x : ℝ) := -x^2 - 2*x + a

theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

theorem parametric_quadratic_inequality_solution (a : ℝ) :
  ({x : ℝ | g a x < 0} = Set.univ) ↔ a < -1 := by sorry

end quadratic_inequality_solution_parametric_quadratic_inequality_solution_l1750_175012


namespace problem_polygon_area_l1750_175089

-- Define a point on a 2D grid
structure GridPoint where
  x : Int
  y : Int

-- Define a polygon as a list of grid points
def Polygon := List GridPoint

-- Function to calculate the area of a polygon given its vertices
def polygonArea (p : Polygon) : ℚ :=
  sorry

-- Define the specific polygon from the problem
def problemPolygon : Polygon := [
  ⟨0, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨30, 0⟩, ⟨40, 0⟩, ⟨40, 10⟩,
  ⟨40, 20⟩, ⟨30, 30⟩, ⟨20, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨0, 10⟩
]

-- Theorem statement
theorem problem_polygon_area :
  polygonArea problemPolygon = 15/2 := by sorry

end problem_polygon_area_l1750_175089


namespace vector_addition_proof_l1750_175042

def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

theorem vector_addition_proof : 
  (2 : ℝ) • a + b = (7, -3, 2) := by sorry

end vector_addition_proof_l1750_175042


namespace batting_bowling_average_change_l1750_175094

/-- Represents a batsman's performance in a cricket inning -/
structure InningPerformance where
  runs : ℕ
  boundaries : ℕ
  sixes : ℕ
  strike_rate : ℝ
  wickets : ℕ

/-- Calculates the new batting average after an inning -/
def new_batting_average (initial_average : ℝ) (performance : InningPerformance) : ℝ :=
  initial_average + 5

/-- Calculates the new bowling average after an inning -/
def new_bowling_average (initial_average : ℝ) (performance : InningPerformance) : ℝ :=
  initial_average - 3

theorem batting_bowling_average_change 
  (A B : ℝ) 
  (performance : InningPerformance) 
  (h1 : performance.runs = 100) 
  (h2 : performance.boundaries = 12) 
  (h3 : performance.sixes = 2) 
  (h4 : performance.strike_rate = 130) 
  (h5 : performance.wickets = 1) :
  new_batting_average A performance = A + 5 ∧ 
  new_bowling_average B performance = B - 3 := by
  sorry


end batting_bowling_average_change_l1750_175094


namespace cube_surface_area_l1750_175048

/-- The surface area of a cube with side length 8 centimeters is 384 square centimeters. -/
theorem cube_surface_area : 
  let side_length : ℝ := 8
  let surface_area : ℝ := 6 * side_length * side_length
  surface_area = 384 := by sorry

end cube_surface_area_l1750_175048


namespace paper_tearing_theorem_l1750_175080

/-- Represents the number of parts after n tears -/
def num_parts (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that the number of parts is always odd and can never be 100 -/
theorem paper_tearing_theorem :
  ∀ n : ℕ, ∃ k : ℕ, num_parts n = 2 * k + 1 ∧ num_parts n ≠ 100 :=
sorry

end paper_tearing_theorem_l1750_175080


namespace sequence_with_geometric_differences_formula_l1750_175092

def sequence_with_geometric_differences (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = (1/3)^(n-1)

theorem sequence_with_geometric_differences_formula (a : ℕ → ℝ) :
  sequence_with_geometric_differences a →
  ∀ n : ℕ, n ≥ 1 → a n = 3/2 * (1 - (1/3)^n) :=
by sorry

end sequence_with_geometric_differences_formula_l1750_175092


namespace ellipse_and_line_theorem_l1750_175098

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2

-- Define the arithmetic sequence property
def arithmetic_sequence (P : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), Real.sqrt ((P.1 + 1)^2 + P.2^2) = 2 - d ∧
             Real.sqrt ((P.1 - 1)^2 + P.2^2) = 2 + d

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (3 * Real.sqrt 7 / 7) * (x - 1) ∨
  y = -(3 * Real.sqrt 7 / 7) * (x - 1)

-- Define the perpendicular property
def perpendicular_property (P Q : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (Q.1 - F₁.1) + (P.2 - F₁.2) * (Q.2 - F₁.2) = 0

theorem ellipse_and_line_theorem :
  ∀ (P : ℝ × ℝ),
    point_on_ellipse P →
    arithmetic_sequence P →
    ∀ (Q : ℝ × ℝ),
      point_on_ellipse Q →
      Q.1 = F₂.1 →
      perpendicular_property P Q →
      line_m P.1 P.2 ∧ line_m Q.1 Q.2 :=
sorry

end ellipse_and_line_theorem_l1750_175098


namespace probability_sum_binary_digits_not_exceed_eight_l1750_175066

/-- The maximum number in the set of possible values -/
def max_num : ℕ := 2016

/-- Function to calculate the sum of binary digits of a natural number -/
def sum_binary_digits (n : ℕ) : ℕ := sorry

/-- The count of numbers from 1 to max_num with sum of binary digits not exceeding 8 -/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the probability of a randomly chosen number from 1 to max_num 
    having a sum of binary digits not exceeding 8 -/
theorem probability_sum_binary_digits_not_exceed_eight :
  (count_valid_numbers : ℚ) / max_num = 655 / 672 := by sorry

end probability_sum_binary_digits_not_exceed_eight_l1750_175066


namespace sports_club_membership_l1750_175065

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 80 →
  badminton = 48 →
  tennis = 46 →
  neither = 7 →
  badminton + tennis - (total - neither) = 21 :=
by
  sorry

end sports_club_membership_l1750_175065


namespace highest_score_is_103_l1750_175007

def base_score : ℕ := 100

def score_adjustments : List ℤ := [3, -8, 0]

def actual_scores : List ℕ := score_adjustments.map (λ x => (base_score : ℤ) + x |>.toNat)

theorem highest_score_is_103 : actual_scores.maximum? = some 103 := by
  sorry

end highest_score_is_103_l1750_175007


namespace sum_reciprocal_inequality_max_nonnegative_l1750_175077

-- Problem 1
theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

-- Problem 2
theorem max_nonnegative (x : ℝ) :
  let a := x^2 - 1
  let b := 2*x + 2
  max a b ≥ 0 := by
  sorry

end sum_reciprocal_inequality_max_nonnegative_l1750_175077


namespace sum_of_ages_l1750_175009

/-- Given the ages of Eunji, Yuna, and Eunji's uncle, prove that the sum of Eunji's and Yuna's ages is 35 years. -/
theorem sum_of_ages (uncle_age : ℕ) (eunji_age : ℕ) (yuna_age : ℕ)
  (h1 : uncle_age = 41)
  (h2 : uncle_age = eunji_age + 25)
  (h3 : yuna_age = eunji_age + 3) :
  eunji_age + yuna_age = 35 :=
by sorry

end sum_of_ages_l1750_175009


namespace correct_tax_distribution_l1750_175073

-- Define the types of taxes
inductive TaxType
  | PropertyTax
  | FederalTax
  | ProfitTax
  | RegionalTax
  | TransportTax

-- Define the budget levels
inductive BudgetLevel
  | Federal
  | Regional

-- Function to map tax types to budget levels
def taxDistribution : TaxType → BudgetLevel
  | TaxType.PropertyTax => BudgetLevel.Regional
  | TaxType.FederalTax => BudgetLevel.Federal
  | TaxType.ProfitTax => BudgetLevel.Regional
  | TaxType.RegionalTax => BudgetLevel.Regional
  | TaxType.TransportTax => BudgetLevel.Regional

-- Theorem stating the correct distribution of taxes
theorem correct_tax_distribution :
  (taxDistribution TaxType.PropertyTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.FederalTax = BudgetLevel.Federal) ∧
  (taxDistribution TaxType.ProfitTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.RegionalTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.TransportTax = BudgetLevel.Regional) :=
by sorry

end correct_tax_distribution_l1750_175073


namespace total_money_proof_l1750_175064

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
sorry

end total_money_proof_l1750_175064


namespace units_digit_pow_two_cycle_units_digit_pow_two_2015_l1750_175061

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_pow_two_cycle (n : ℕ) (h : n ≥ 1) : 
  units_digit (2^n) = units_digit (2^((n - 1) % 4 + 1)) :=
sorry

theorem units_digit_pow_two_2015 : units_digit (2^2015) = 8 :=
sorry

end units_digit_pow_two_cycle_units_digit_pow_two_2015_l1750_175061


namespace stewart_farm_ratio_l1750_175085

/-- The Stewart farm scenario -/
structure StewartFarm where
  total_horse_food : ℕ
  horse_food_per_horse : ℕ
  num_sheep : ℕ

/-- Calculate the number of horses on the farm -/
def num_horses (farm : StewartFarm) : ℕ :=
  farm.total_horse_food / farm.horse_food_per_horse

/-- Calculate the ratio of sheep to horses -/
def sheep_to_horses_ratio (farm : StewartFarm) : ℚ :=
  farm.num_sheep / (num_horses farm)

/-- Theorem: The ratio of sheep to horses on the Stewart farm is 6:7 -/
theorem stewart_farm_ratio :
  let farm := StewartFarm.mk 12880 230 48
  sheep_to_horses_ratio farm = 6 / 7 := by
  sorry

end stewart_farm_ratio_l1750_175085


namespace triangle_area_bounds_l1750_175059

/-- The parabola function y = x^2 - 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 1

/-- The line function y = r -/
def line (r : ℝ) (x : ℝ) : ℝ := r

/-- The area of the triangle formed by the vertex of the parabola and its intersections with the line y = r -/
def triangleArea (r : ℝ) : ℝ := (r + 1)^(3/2)

theorem triangle_area_bounds (r : ℝ) :
  (8 ≤ triangleArea r ∧ triangleArea r ≤ 64) → (3 ≤ r ∧ r ≤ 15) :=
by sorry

end triangle_area_bounds_l1750_175059


namespace max_imaginary_part_of_z_l1750_175071

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem max_imaginary_part_of_z (z : ℂ) 
  (h : is_purely_imaginary ((z - 6) / (z - 8*I))) : 
  (⨆ (z : ℂ), |z.im|) = 9 := by sorry

end max_imaginary_part_of_z_l1750_175071


namespace arrangements_count_l1750_175033

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- The number of elderly persons --/
def num_elderly : ℕ := 1

/-- The total number of people --/
def total_people : ℕ := num_volunteers + num_elderly

/-- The position of the elderly person --/
def elderly_position : ℕ := (total_people + 1) / 2

theorem arrangements_count :
  (num_volunteers.factorial * (num_volunteers + 1 - elderly_position).factorial) = 24 :=
sorry

end arrangements_count_l1750_175033


namespace fraction_equality_l1750_175035

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 4) :
  (4 * a + 3 * b) / (4 * a - 3 * b) = 4 := by
  sorry

end fraction_equality_l1750_175035


namespace first_class_students_l1750_175014

theorem first_class_students (x : ℕ) : 
  (∃ (total_students : ℕ),
    total_students = x + 50 ∧
    (50 * x + 60 * 50 : ℚ) / total_students = 56.25) →
  x = 30 := by
sorry

end first_class_students_l1750_175014


namespace equation_solution_l1750_175088

theorem equation_solution :
  ∃ (a b c d : ℝ), 
    2 * a^2 + b^2 + 2 * c^2 + 2 = 3 * d + Real.sqrt (2 * a + b + 2 * c - 3 * d) ∧
    d = 2/3 ∧ a = 1/2 ∧ b = 1 ∧ c = 1/2 := by
  sorry

end equation_solution_l1750_175088


namespace max_equal_covering_is_three_l1750_175049

/-- Represents a square covering on a cube face -/
structure SquareCovering where
  position : Fin 6 × Fin 6
  folded : Bool

/-- Represents the cube and its covering -/
structure CubeCovering where
  squares : List SquareCovering

/-- Check if a cell is covered by a square -/
def covers (s : SquareCovering) (cell : Fin 6 × Fin 6) : Bool :=
  sorry

/-- Count how many squares cover a given cell -/
def coverCount (cc : CubeCovering) (cell : Fin 6 × Fin 6 × Fin 3) : Nat :=
  sorry

/-- Check if the covering is valid (no overlaps, all 2x2) -/
def isValidCovering (cc : CubeCovering) : Bool :=
  sorry

/-- Check if all cells are covered equally -/
def isEqualCovering (cc : CubeCovering) : Bool :=
  sorry

/-- The main theorem -/
theorem max_equal_covering_is_three :
  ∀ (cc : CubeCovering),
    isValidCovering cc →
    isEqualCovering cc →
    ∃ (n : Nat), (∀ (cell : Fin 6 × Fin 6 × Fin 3), coverCount cc cell = n) ∧ n ≤ 3 :=
  sorry

end max_equal_covering_is_three_l1750_175049


namespace f_properties_l1750_175041

def f (x : ℝ) := (x - 2)^2

theorem f_properties :
  (∀ x, f (x + 2) = f (-x - 2)) ∧
  (∀ x < 2, ∀ y < x, f y < f x) ∧
  (∀ x > 2, ∀ y > x, f y > f x) ∧
  (∀ x y, x < y → f (y + 2) - f y > f (x + 2) - f x) :=
by sorry

end f_properties_l1750_175041


namespace peters_horses_food_l1750_175078

/-- The amount of food needed for Peter's horses over 5 days -/
def food_needed (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) 
                (grain_per_meal : ℕ) (grain_meals_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_horses * (oats_per_meal * oats_meals_per_day + grain_per_meal * grain_meals_per_day) * num_days

/-- Theorem stating the total amount of food needed for Peter's horses -/
theorem peters_horses_food : 
  food_needed 6 5 3 4 2 5 = 690 := by
  sorry

end peters_horses_food_l1750_175078


namespace tens_digit_of_2031_pow_2024_minus_2033_l1750_175054

theorem tens_digit_of_2031_pow_2024_minus_2033 :
  ∃ n : ℕ, n < 10 ∧ (2031^2024 - 2033) % 100 = 80 + n :=
by
  -- The proof goes here
  sorry

end tens_digit_of_2031_pow_2024_minus_2033_l1750_175054


namespace taxi_charge_calculation_l1750_175051

/-- Calculates the total charge for a taxi trip with given conditions -/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_two_fifths_mile : ℚ) 
  (trip_distance : ℚ) (non_peak_discount : ℚ) (standard_car_discount : ℚ) : ℚ :=
  let base_charge := initial_fee + (trip_distance / (2/5)) * charge_per_two_fifths_mile
  let discount := base_charge * (non_peak_discount + standard_car_discount)
  base_charge - discount

/-- The total charge for the taxi trip is $4.95 -/
theorem taxi_charge_calculation :
  let initial_fee : ℚ := 235/100
  let charge_per_two_fifths_mile : ℚ := 35/100
  let trip_distance : ℚ := 36/10
  let non_peak_discount : ℚ := 7/100
  let standard_car_discount : ℚ := 3/100
  calculate_taxi_charge initial_fee charge_per_two_fifths_mile trip_distance 
    non_peak_discount standard_car_discount = 495/100 := by
  sorry

end taxi_charge_calculation_l1750_175051


namespace oil_distribution_l1750_175067

theorem oil_distribution (a b c : ℝ) : 
  c = 48 →
  (2/3 * a = 4/5 * (b + 1/3 * a)) →
  (2/3 * a = 48 + 1/5 * (b + 1/3 * a)) →
  a = 96 ∧ b = 48 := by
sorry

end oil_distribution_l1750_175067


namespace car_distance_theorem_l1750_175090

/-- Calculates the total distance covered by a car given its uphill and downhill speeds and times. -/
def total_distance (uphill_speed downhill_speed uphill_time downhill_time : ℝ) : ℝ :=
  uphill_speed * uphill_time + downhill_speed * downhill_time

/-- Theorem stating that under the given conditions, the total distance covered by the car is 400 km. -/
theorem car_distance_theorem :
  let uphill_speed : ℝ := 30
  let downhill_speed : ℝ := 50
  let uphill_time : ℝ := 5
  let downhill_time : ℝ := 5
  total_distance uphill_speed downhill_speed uphill_time downhill_time = 400 := by
  sorry

end car_distance_theorem_l1750_175090


namespace rectangle_area_perimeter_ratio_max_l1750_175070

theorem rectangle_area_perimeter_ratio_max (A P : ℝ) (h1 : A > 0) (h2 : P > 0) : 
  A / P^2 ≤ 1 / 16 := by
  sorry

end rectangle_area_perimeter_ratio_max_l1750_175070


namespace max_discount_percentage_l1750_175036

theorem max_discount_percentage (cost : ℝ) (price : ℝ) (min_margin : ℝ) :
  cost = 400 →
  price = 500 →
  min_margin = 0.0625 →
  ∃ x : ℝ, x = 15 ∧
    ∀ y : ℝ, 0 ≤ y → y ≤ x →
      price * (1 - y / 100) - cost ≥ cost * min_margin ∧
      ∀ z : ℝ, z > x →
        price * (1 - z / 100) - cost < cost * min_margin :=
by sorry

end max_discount_percentage_l1750_175036


namespace research_budget_allocation_l1750_175010

theorem research_budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (industrial_lubricants : ℝ) (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 20 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 18 →
  ∃ (genetically_modified_microorganisms : ℝ),
    genetically_modified_microorganisms = 29 ∧
    microphotonics + home_electronics + food_additives + industrial_lubricants + 
    (basic_astrophysics_degrees / 360 * 100) + genetically_modified_microorganisms = 100 :=
by sorry

end research_budget_allocation_l1750_175010


namespace problems_solved_l1750_175055

theorem problems_solved (first last : ℕ) (h : first = 78 ∧ last = 125) : 
  (last - first + 1 : ℕ) = 49 := by
  sorry

end problems_solved_l1750_175055


namespace sin_cos_sixth_power_sum_l1750_175028

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.tan θ = 1/6) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11/12 := by
  sorry

end sin_cos_sixth_power_sum_l1750_175028


namespace inequality_solution_sets_l1750_175046

theorem inequality_solution_sets 
  (a b : ℝ) 
  (h1 : ∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by sorry

end inequality_solution_sets_l1750_175046


namespace joint_account_final_amount_l1750_175037

/-- Calculates the final amount in a joint account after one year with changing interest rates and tax --/
theorem joint_account_final_amount 
  (deposit_lopez : ℝ) 
  (deposit_johnson : ℝ) 
  (initial_rate : ℝ) 
  (changed_rate : ℝ) 
  (tax_rate : ℝ) 
  (h1 : deposit_lopez = 100)
  (h2 : deposit_johnson = 150)
  (h3 : initial_rate = 0.20)
  (h4 : changed_rate = 0.18)
  (h5 : tax_rate = 0.05) : 
  ∃ (final_amount : ℝ), abs (final_amount - 272.59) < 0.01 := by
  sorry

end joint_account_final_amount_l1750_175037


namespace sum_of_solutions_quadratic_l1750_175045

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x - 24) → (∃ y : ℝ, y^2 = 10*y - 24 ∧ x + y = 10) :=
by sorry

end sum_of_solutions_quadratic_l1750_175045


namespace meaningful_square_root_range_l1750_175084

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end meaningful_square_root_range_l1750_175084


namespace min_hat_flips_min_hat_flips_1000_l1750_175000

theorem min_hat_flips (n : ℕ) (h : n = 1000) : ℕ :=
  let elf_count := n
  let initial_red_count : ℕ := n - 1
  let initial_blue_count : ℕ := 1
  let final_red_count : ℕ := 1
  let final_blue_count : ℕ := n - 1
  let min_flips := initial_red_count - final_red_count
  min_flips

/-- The minimum number of hat flips required for 1000 elves to satisfy the conditions is 998. -/
theorem min_hat_flips_1000 : min_hat_flips 1000 (by rfl) = 998 := by
  sorry

end min_hat_flips_min_hat_flips_1000_l1750_175000


namespace installment_value_approximation_l1750_175020

def tv_price : ℕ := 15000
def num_installments : ℕ := 20
def interest_rate : ℚ := 6 / 100
def last_installment : ℕ := 13000

def calculate_installment_value (price : ℕ) (num_inst : ℕ) (rate : ℚ) (last_inst : ℕ) : ℚ :=
  let avg_balance : ℚ := price / 2
  let interest : ℚ := avg_balance * rate
  let total_amount : ℚ := price + interest
  (total_amount - last_inst) / (num_inst - 1)

theorem installment_value_approximation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_installment_value tv_price num_installments interest_rate last_installment - 129| < ε :=
sorry

end installment_value_approximation_l1750_175020


namespace reciprocal_of_negative_fraction_l1750_175053

theorem reciprocal_of_negative_fraction (a b : ℚ) (h : b ≠ 0) :
  ((-a) / b)⁻¹ = -(b / a) :=
by sorry

end reciprocal_of_negative_fraction_l1750_175053


namespace child_workers_count_l1750_175087

/-- Represents the number of child workers employed by the contractor. -/
def num_child_workers : ℕ := 5

/-- Represents the number of male workers employed by the contractor. -/
def num_male_workers : ℕ := 20

/-- Represents the number of female workers employed by the contractor. -/
def num_female_workers : ℕ := 15

/-- Represents the daily wage of a male worker in rupees. -/
def male_wage : ℕ := 25

/-- Represents the daily wage of a female worker in rupees. -/
def female_wage : ℕ := 20

/-- Represents the daily wage of a child worker in rupees. -/
def child_wage : ℕ := 8

/-- Represents the average daily wage paid by the contractor in rupees. -/
def average_wage : ℕ := 21

/-- Theorem stating that the number of child workers is 5, given the conditions. -/
theorem child_workers_count :
  (num_male_workers * male_wage + num_female_workers * female_wage + num_child_workers * child_wage) / 
  (num_male_workers + num_female_workers + num_child_workers) = average_wage := by
  sorry

end child_workers_count_l1750_175087


namespace equation_solution_l1750_175069

theorem equation_solution : 
  ∃ x : ℝ, ((x * 5) / 2.5) - (8 * 2.25) = 5.5 ∧ x = 11.75 := by
  sorry

end equation_solution_l1750_175069


namespace survey_sample_size_l1750_175099

/-- Represents a survey with its characteristics -/
structure Survey where
  surveyors : ℕ
  households : ℕ
  questionnaires : ℕ

/-- Definition of sample size for a survey -/
def sampleSize (s : Survey) : ℕ := s.questionnaires

/-- Theorem stating that the sample size is equal to the number of questionnaires -/
theorem survey_sample_size (s : Survey) : sampleSize s = s.questionnaires := by
  sorry

/-- The specific survey described in the problem -/
def cityCenterSurvey : Survey := {
  surveyors := 400,
  households := 10000,
  questionnaires := 30000
}

#eval sampleSize cityCenterSurvey

end survey_sample_size_l1750_175099


namespace C_power_50_l1750_175039

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_power_50 : C^50 = !![(-299 : ℤ), -100; 800, 251] := by sorry

end C_power_50_l1750_175039


namespace last_four_average_l1750_175076

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 63.75 :=
by sorry

end last_four_average_l1750_175076


namespace double_age_in_two_years_l1750_175074

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

/-- Theorem: Given the son's age is 22 and the age difference is 24,
    the number of years until the man's age is twice his son's age is 2 -/
theorem double_age_in_two_years :
  years_until_double_age 22 24 = 2 := by
  sorry

end double_age_in_two_years_l1750_175074


namespace ratio_is_five_thirds_l1750_175002

/-- Given a diagram with triangles, some shaded and some unshaded -/
structure TriangleDiagram where
  shaded : ℕ
  unshaded : ℕ

/-- The ratio of shaded to unshaded triangles -/
def shaded_unshaded_ratio (d : TriangleDiagram) : ℚ :=
  d.shaded / d.unshaded

theorem ratio_is_five_thirds (d : TriangleDiagram) 
  (h1 : d.shaded = 5) 
  (h2 : d.unshaded = 3) : 
  shaded_unshaded_ratio d = 5 / 3 := by
  sorry

end ratio_is_five_thirds_l1750_175002


namespace tony_distance_behind_l1750_175021

-- Define the slope length
def slope_length : ℝ := 700

-- Define the meeting point distance from the top
def meeting_point : ℝ := 70

-- Define Maria's and Tony's uphill speeds as variables
variable (maria_uphill_speed tony_uphill_speed : ℝ)

-- Define the theorem
theorem tony_distance_behind (maria_uphill_speed tony_uphill_speed : ℝ) 
  (h_positive : maria_uphill_speed > 0 ∧ tony_uphill_speed > 0) :
  let maria_total_distance := slope_length + slope_length / 2
  let tony_total_distance := maria_total_distance * (tony_uphill_speed / maria_uphill_speed)
  let distance_behind := maria_total_distance - tony_total_distance
  2 * distance_behind = 300 := by sorry

end tony_distance_behind_l1750_175021


namespace correct_fraction_l1750_175003

theorem correct_fraction (number : ℚ) (incorrect_fraction : ℚ) (difference : ℚ) :
  number = 96 →
  incorrect_fraction = 5 / 6 →
  incorrect_fraction * number = number * x + difference →
  difference = 50 →
  x = 5 / 16 := by
  sorry

end correct_fraction_l1750_175003


namespace apple_tree_yield_l1750_175001

theorem apple_tree_yield (total : ℕ) : 
  (total / 5 : ℚ) +             -- First day
  (2 * (total / 5) : ℚ) +       -- Second day
  (total / 5 + 20 : ℚ) +        -- Third day
  20 = total →                  -- Remaining apples
  total = 200 := by
sorry

end apple_tree_yield_l1750_175001


namespace triangle_angle_tangent_difference_l1750_175083

theorem triangle_angle_tangent_difference (A B : Real) (cosA tanB : Real) 
  (h1 : cosA = -Real.sqrt 2 / 2)
  (h2 : tanB = 1 / 3) :
  Real.tan (A - B) = -2 := by
  sorry

end triangle_angle_tangent_difference_l1750_175083


namespace smallest_prime_sum_of_five_primes_l1750_175025

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if a list of natural numbers contains distinct elements -/
def isDistinct (list : List ℕ) : Prop := list.Nodup

/-- The theorem stating that 43 is the smallest prime that is the sum of five distinct primes -/
theorem smallest_prime_sum_of_five_primes :
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ),
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ ∧ isPrime p₅ ∧
    isDistinct [p₁, p₂, p₃, p₄, p₅] ∧
    p₁ + p₂ + p₃ + p₄ + p₅ = 43 ∧
    isPrime 43 ∧
    (∀ (q : ℕ), q < 43 →
      ¬∃ (q₁ q₂ q₃ q₄ q₅ : ℕ),
        isPrime q₁ ∧ isPrime q₂ ∧ isPrime q₃ ∧ isPrime q₄ ∧ isPrime q₅ ∧
        isDistinct [q₁, q₂, q₃, q₄, q₅] ∧
        q₁ + q₂ + q₃ + q₄ + q₅ = q ∧
        isPrime q) :=
by sorry


end smallest_prime_sum_of_five_primes_l1750_175025


namespace third_row_is_4213_l1750_175044

/-- Represents a 4x4 grid of numbers -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a number is the first odd or even in a list -/
def isFirstOddOrEven (n : Fin 4) (list : List (Fin 4)) : Prop :=
  n.val % 2 ≠ list.head!.val % 2 ∧ 
  ∀ m ∈ list, m.val < n.val → m.val % 2 = list.head!.val % 2

/-- The constraints of the grid puzzle -/
structure GridConstraints (grid : Grid) : Prop where
  unique_in_row : ∀ i j k, j ≠ k → grid i j ≠ grid i k
  unique_in_col : ∀ i j k, i ≠ k → grid i j ≠ grid k j
  top_indicators : ∀ j, isFirstOddOrEven (grid 0 j) [grid 1 j, grid 2 j, grid 3 j]
  left_indicators : ∀ i, isFirstOddOrEven (grid i 0) [grid i 1, grid i 2, grid i 3]
  right_indicators : ∀ i, isFirstOddOrEven (grid i 3) [grid i 2, grid i 1, grid i 0]
  bottom_indicators : ∀ j, isFirstOddOrEven (grid 3 j) [grid 2 j, grid 1 j, grid 0 j]

/-- The main theorem stating that the third row must be [4, 2, 1, 3] -/
theorem third_row_is_4213 (grid : Grid) (h : GridConstraints grid) :
  (grid 2 0 = 4) ∧ (grid 2 1 = 2) ∧ (grid 2 2 = 1) ∧ (grid 2 3 = 3) := by
  sorry

end third_row_is_4213_l1750_175044


namespace jerrys_average_increase_l1750_175081

theorem jerrys_average_increase :
  ∀ (original_average new_average : ℚ),
  original_average = 94 →
  (3 * original_average + 102) / 4 = new_average →
  new_average - original_average = 2 := by
sorry

end jerrys_average_increase_l1750_175081


namespace bob_wins_for_S_l1750_175018

/-- A set of lattice points in the Cartesian plane -/
def LatticeSet := Set (ℤ × ℤ)

/-- The set S defined by m and n -/
def S (m n : ℕ) : LatticeSet :=
  {p : ℤ × ℤ | m ≤ p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 ≤ n}

/-- Count of points on a line -/
def LineCount := ℕ

/-- Information provided by Alice: counts of points on horizontal, vertical, and diagonal lines -/
structure AliceInfo :=
  (horizontal : ℤ → LineCount)
  (vertical : ℤ → LineCount)
  (diagonalPos : ℤ → LineCount)  -- y = x + k
  (diagonalNeg : ℤ → LineCount)  -- y = -x + k

/-- Generate AliceInfo from a given set -/
def getAliceInfo (s : LatticeSet) : AliceInfo :=
  sorry

/-- Bob's winning condition -/
def BobCanWin (s : LatticeSet) : Prop :=
  ∀ t : LatticeSet, getAliceInfo s = getAliceInfo t → s = t

/-- Main theorem -/
theorem bob_wins_for_S (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  BobCanWin (S m n) :=
sorry

end bob_wins_for_S_l1750_175018


namespace quadratic_factorization_l1750_175016

theorem quadratic_factorization (a : ℝ) : a^2 - 8*a + 16 = (a - 4)^2 := by
  sorry

end quadratic_factorization_l1750_175016


namespace parallel_line_segment_length_l1750_175008

/-- Given a triangle with sides a, b, c, and lines parallel to the sides drawn through an interior point,
    if the segments of these lines within the triangle are equal in length x, then
    x = 2 / (1/a + 1/b + 1/c) -/
theorem parallel_line_segment_length (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  x > 0 → x = 2 / (1/a + 1/b + 1/c) := by
  sorry

end parallel_line_segment_length_l1750_175008


namespace arctan_equation_solution_l1750_175050

theorem arctan_equation_solution (y : ℝ) : 
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = π/4 → y = 31/9 := by
sorry

end arctan_equation_solution_l1750_175050


namespace solve_euro_equation_l1750_175032

-- Define the operation €
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem solve_euro_equation (x : ℝ) :
  (euro 6 (euro x 5) = 480) → x = 4 := by
  sorry

end solve_euro_equation_l1750_175032


namespace square_difference_ends_with_two_l1750_175047

theorem square_difference_ends_with_two (a b : ℕ) (h1 : a^2 > b^2) 
  (h2 : ∃ (m n : ℕ), a^2 = m^2 ∧ b^2 = n^2) 
  (h3 : (a^2 - b^2) % 10 = 2) :
  a^2 % 10 = 6 ∧ b^2 % 10 = 4 := by
sorry

end square_difference_ends_with_two_l1750_175047


namespace max_daily_profit_l1750_175091

/-- Represents the daily profit function for a store selling football souvenir books. -/
def daily_profit (x : ℝ) : ℝ :=
  (x - 40) * (-10 * x + 740)

/-- Theorem stating the maximum daily profit and the corresponding selling price. -/
theorem max_daily_profit :
  let cost_price : ℝ := 40
  let initial_price : ℝ := 44
  let initial_sales : ℝ := 300
  let price_range : Set ℝ := {x | 44 ≤ x ∧ x ≤ 52}
  let sales_decrease_rate : ℝ := 10
  ∃ (max_price : ℝ), max_price ∈ price_range ∧
    ∀ (x : ℝ), x ∈ price_range →
      daily_profit x ≤ daily_profit max_price ∧
      daily_profit max_price = 2640 ∧
      max_price = 52 :=
by sorry


end max_daily_profit_l1750_175091


namespace soccer_ball_white_patches_l1750_175062

/-- Represents a soccer ball with hexagonal and pentagonal patches -/
structure SoccerBall where
  total_patches : ℕ
  white_patches : ℕ
  black_patches : ℕ
  white_black_borders : ℕ

/-- Conditions for a valid soccer ball configuration -/
def is_valid_soccer_ball (ball : SoccerBall) : Prop :=
  ball.total_patches = 32 ∧
  ball.white_patches + ball.black_patches = ball.total_patches ∧
  ball.white_black_borders = 3 * ball.white_patches ∧
  ball.white_black_borders = 5 * ball.black_patches

/-- Theorem stating that a valid soccer ball has 20 white patches -/
theorem soccer_ball_white_patches (ball : SoccerBall) 
  (h : is_valid_soccer_ball ball) : ball.white_patches = 20 := by
  sorry

#check soccer_ball_white_patches

end soccer_ball_white_patches_l1750_175062


namespace b_squared_neq_ac_sufficient_not_necessary_l1750_175075

-- Define what it means for three numbers to form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b / a = c / b) ∨ (a = 0 ∧ b = 0) ∨ (b = 0 ∧ c = 0)

-- State the theorem
theorem b_squared_neq_ac_sufficient_not_necessary :
  (∀ a b c : ℝ, b^2 ≠ a*c → ¬is_geometric_sequence a b c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_sequence a b c) := by sorry

end b_squared_neq_ac_sufficient_not_necessary_l1750_175075
