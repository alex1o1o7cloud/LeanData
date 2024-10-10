import Mathlib

namespace history_book_cost_l179_17968

theorem history_book_cost 
  (total_books : ℕ) 
  (math_books : ℕ) 
  (math_book_cost : ℚ) 
  (total_price : ℚ) 
  (h1 : total_books = 90)
  (h2 : math_books = 60)
  (h3 : math_book_cost = 4)
  (h4 : total_price = 390) :
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 := by
  sorry

end history_book_cost_l179_17968


namespace three_consecutive_days_without_class_l179_17901

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in the month -/
structure Day where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a month with its properties -/
structure Month where
  days : List Day
  startDay : DayOfWeek
  totalDays : Nat
  classSchedule : List Nat

/-- Main theorem to prove -/
theorem three_consecutive_days_without_class 
  (november2017 : Month)
  (h1 : november2017.startDay = DayOfWeek.Wednesday)
  (h2 : november2017.totalDays = 30)
  (h3 : november2017.classSchedule.length = 11)
  (h4 : ∀ d ∈ november2017.days, 
    d.dayOfWeek = DayOfWeek.Saturday ∨ d.dayOfWeek = DayOfWeek.Sunday → 
    d.dayNumber ∉ november2017.classSchedule) :
  ∃ d1 d2 d3 : Day, 
    d1 ∈ november2017.days ∧ 
    d2 ∈ november2017.days ∧ 
    d3 ∈ november2017.days ∧ 
    d1.dayNumber + 1 = d2.dayNumber ∧ 
    d2.dayNumber + 1 = d3.dayNumber ∧ 
    d1.dayNumber ∉ november2017.classSchedule ∧ 
    d2.dayNumber ∉ november2017.classSchedule ∧ 
    d3.dayNumber ∉ november2017.classSchedule :=
by sorry

end three_consecutive_days_without_class_l179_17901


namespace arithmetic_sequence_length_l179_17995

/-- 
Proves that an arithmetic sequence with first term 2, common difference 3, 
and last term 2014 has 671 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ),
    a = 2 → d = 3 → last = 2014 → 
    last = a + (n - 1) * d → n = 671 :=
by sorry

end arithmetic_sequence_length_l179_17995


namespace cottage_village_price_l179_17902

/-- The selling price of each house in a cottage village -/
def house_selling_price : ℕ := by sorry

/-- The number of houses in the village -/
def num_houses : ℕ := 15

/-- The total cost of construction for the entire village -/
def total_cost : ℕ := 150 + 105 + 225 + 45

/-- The markup percentage of the construction company -/
def markup_percentage : ℚ := 20 / 100

theorem cottage_village_price :
  (house_selling_price : ℚ) = (total_cost : ℚ) / num_houses * (1 + markup_percentage) ∧
  house_selling_price = 42 := by sorry

end cottage_village_price_l179_17902


namespace even_function_comparison_l179_17927

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is increasing on (-∞, 0) if f(x) < f(y) for all x < y < 0 -/
def IncreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → y < 0 → f x < f y

theorem even_function_comparison (f : ℝ → ℝ) (x₁ x₂ : ℝ)
    (heven : IsEven f)
    (hincr : IncreasingOnNegatives f)
    (hx₁ : x₁ < 0)
    (hsum : x₁ + x₂ > 0) :
    f x₁ > f x₂ := by
  sorry


end even_function_comparison_l179_17927


namespace fraction_to_decimal_l179_17934

theorem fraction_to_decimal : 58 / 200 = 1.16 := by sorry

end fraction_to_decimal_l179_17934


namespace erased_number_proof_l179_17936

theorem erased_number_proof (n : ℕ) (i : ℕ) :
  n > 0 ∧ i > 0 ∧ i ≤ n ∧
  (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17 →
  i = 7 :=
by sorry

end erased_number_proof_l179_17936


namespace prob_at_least_one_female_is_seven_tenths_l179_17962

/-- Represents the composition of a research team -/
structure ResearchTeam where
  total : Nat
  males : Nat
  females : Nat

/-- Calculates the probability of at least one female being selected
    when choosing two representatives from a research team -/
def probAtLeastOneFemale (team : ResearchTeam) : Rat :=
  sorry

/-- The main theorem stating the probability for the given team composition -/
theorem prob_at_least_one_female_is_seven_tenths :
  let team : ResearchTeam := ⟨5, 3, 2⟩
  probAtLeastOneFemale team = 7 / 10 := by
  sorry

end prob_at_least_one_female_is_seven_tenths_l179_17962


namespace geometric_progression_equality_l179_17935

theorem geometric_progression_equality 
  (a b q : ℝ) 
  (n p : ℕ) 
  (h_q : q ≠ 1) 
  (h_sum : a * (1 - q^(n*p)) / (1 - q) = b * (1 - q^(n*p)) / (1 - q^p)) :
  b = a * (1 - q^p) / (1 - q) :=
sorry

end geometric_progression_equality_l179_17935


namespace factor_expression_l179_17978

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := by
  sorry

end factor_expression_l179_17978


namespace magnitude_a_minus_2b_equals_sqrt_21_l179_17943

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- State the theorem
theorem magnitude_a_minus_2b_equals_sqrt_21 
  (h1 : ‖b‖ = 2 * ‖a‖) 
  (h2 : ‖b‖ = 2) 
  (h3 : inner a b = ‖a‖ * ‖b‖ * (-1/2)) : 
  ‖a - 2 • b‖ = Real.sqrt 21 := by
sorry


end magnitude_a_minus_2b_equals_sqrt_21_l179_17943


namespace simplify_sqrt_expression_l179_17916

theorem simplify_sqrt_expression : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l179_17916


namespace range_of_a_l179_17972

/-- The exponential function f(x) = (2a - 6)^x is monotonically decreasing on ℝ -/
def P (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 6)^x > (2*a - 6)^y

/-- Both real roots of the equation x^2 - 3ax + 2a^2 + 1 = 0 are greater than 3 -/
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 3*a*x + 2*a^2 + 1 = 0 → x > 3

theorem range_of_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 4) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ≥ 3.5 ∧ a < 5 := by sorry

end range_of_a_l179_17972


namespace car_speed_proof_l179_17920

/-- The speed of a car in km/h -/
def car_speed : ℝ := 48

/-- The reference speed in km/h -/
def reference_speed : ℝ := 60

/-- The additional time taken in seconds -/
def additional_time : ℝ := 15

/-- The distance traveled in km -/
def distance : ℝ := 1

theorem car_speed_proof :
  (distance / car_speed) * 3600 = (distance / reference_speed) * 3600 + additional_time :=
by sorry

end car_speed_proof_l179_17920


namespace wall_width_calculation_l179_17904

theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) :
  mirror_side = 21 →
  wall_length = 31.5 →
  (mirror_side * mirror_side) * 2 = wall_length * (882 / wall_length) := by
  sorry

#check wall_width_calculation

end wall_width_calculation_l179_17904


namespace arithmetic_sequence_common_difference_l179_17999

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a₁ : ℝ)
  (d : ℝ)
  (h1 : arithmeticSequence a₁ d 1 + arithmeticSequence a₁ d 3 + arithmeticSequence a₁ d 5 = 15)
  (h2 : arithmeticSequence a₁ d 4 = 3) :
  d = -2 := by
  sorry

end arithmetic_sequence_common_difference_l179_17999


namespace novel_pages_count_l179_17922

theorem novel_pages_count (x : ℝ) : 
  let day1_read := x / 6 + 10
  let day1_remaining := x - day1_read
  let day2_read := day1_remaining / 5 + 14
  let day2_remaining := day1_remaining - day2_read
  let day3_read := day2_remaining / 4 + 16
  let day3_remaining := day2_remaining - day3_read
  day3_remaining = 48 → x = 161 := by
sorry

end novel_pages_count_l179_17922


namespace arc_length_for_given_circle_l179_17921

theorem arc_length_for_given_circle (r : ℝ) (θ : ℝ) (arc_length : ℝ) : 
  r = 2 → θ = π / 7 → arc_length = r * θ → arc_length = 2 * π / 7 := by
  sorry

end arc_length_for_given_circle_l179_17921


namespace floor_plus_half_l179_17966

theorem floor_plus_half (x : ℝ) : 
  ⌊x + 0.5⌋ = ⌊x⌋ ∨ ⌊x + 0.5⌋ = ⌊x⌋ + 1 := by sorry

end floor_plus_half_l179_17966


namespace efficient_methods_l179_17941

-- Define the types of calculation methods
inductive CalculationMethod
  | Mental
  | Written
  | Calculator

-- Define a function to determine the most efficient method for a given calculation
def most_efficient_method (calculation : ℕ → ℕ → ℕ) : CalculationMethod :=
  sorry

-- Define the specific calculations
def calc1 : ℕ → ℕ → ℕ := λ x y ↦ (x - y) / 5
def calc2 : ℕ → ℕ → ℕ := λ x _ ↦ x * x

-- State the theorem
theorem efficient_methods :
  (most_efficient_method calc1 = CalculationMethod.Calculator) ∧
  (most_efficient_method calc2 = CalculationMethod.Mental) :=
sorry

end efficient_methods_l179_17941


namespace sum_of_fractions_l179_17905

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l179_17905


namespace soccer_team_girls_l179_17960

theorem soccer_team_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  present = (2 / 3 : ℚ) * boys + girls →
  girls = 18 := by
sorry

end soccer_team_girls_l179_17960


namespace f_properties_l179_17991

def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x > 2 ∧ y > x → f x < f y) ∧ 
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry

end f_properties_l179_17991


namespace probability_sum_5_is_one_ninth_l179_17923

/-- The number of possible outcomes when rolling two fair dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum of 5) when rolling two fair dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two fair dice -/
def probability_sum_5 : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of rolling a sum of 5 with two fair dice is 1/9 -/
theorem probability_sum_5_is_one_ninth : probability_sum_5 = 1 / 9 := by
  sorry

end probability_sum_5_is_one_ninth_l179_17923


namespace inscribed_circle_radius_bound_l179_17915

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the shorter leg -/
  a : ℝ
  /-- The length of the longer leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- Ensure a is the shorter leg -/
  h_a_le_b : a ≤ b
  /-- Pythagorean theorem -/
  h_pythagorean : a^2 + b^2 = c^2
  /-- Formula for the radius of the inscribed circle -/
  h_radius : r = (a + b - c) / 2
  /-- Positivity conditions -/
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_c_pos : c > 0
  h_r_pos : r > 0

/-- The main theorem: the radius of the inscribed circle is less than one-third of the longer leg -/
theorem inscribed_circle_radius_bound (t : RightTriangleWithInscribedCircle) : t.r < t.b / 3 := by
  sorry

end inscribed_circle_radius_bound_l179_17915


namespace ellipse_major_axis_length_l179_17982

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis of the ellipse is 8 --/
theorem ellipse_major_axis_length :
  let cylinder_radius : ℝ := 2
  let major_minor_ratio : ℝ := 2
  major_axis_length cylinder_radius major_minor_ratio = 8 := by
  sorry

#check ellipse_major_axis_length

end ellipse_major_axis_length_l179_17982


namespace product_of_distances_l179_17931

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the ellipse
def P : ℝ × ℝ := sorry

-- State that P is on the ellipse C
axiom P_on_C : C P.1 P.2

-- Define the dot product of vectors PF₁ and PF₂
def PF₁_dot_PF₂ : ℝ := sorry

-- State that the dot product of PF₁ and PF₂ is zero
axiom PF₁_perp_PF₂ : PF₁_dot_PF₂ = 0

-- Define the distances |PF₁| and |PF₂|
def dist_PF₁ : ℝ := sorry
def dist_PF₂ : ℝ := sorry

-- Theorem to prove
theorem product_of_distances : dist_PF₁ * dist_PF₂ = 2 := by sorry

end product_of_distances_l179_17931


namespace machines_working_time_l179_17975

theorem machines_working_time (x : ℝ) : 
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := by
  sorry

end machines_working_time_l179_17975


namespace fraction_value_l179_17961

theorem fraction_value (x y : ℝ) (h : (1 / x) + (1 / y) = 2) :
  (-2 * y + x * y - 2 * x) / (3 * x + x * y + 3 * y) = -3 / 7 := by
sorry

end fraction_value_l179_17961


namespace quadratic_inequality_roots_l179_17913

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b*x - 5 < 0 ↔ x < 1 ∨ x > 5) → b = 6 :=
by sorry

end quadratic_inequality_roots_l179_17913


namespace product_of_special_reals_l179_17985

/-- Given two real numbers a and b satisfying certain conditions, 
    their product is approximately 17.26 -/
theorem product_of_special_reals (a b : ℝ) 
  (sum_eq : a + b = 8)
  (fourth_power_sum : a^4 + b^4 = 272) :
  ∃ ε > 0, |a * b - 17.26| < ε :=
sorry

end product_of_special_reals_l179_17985


namespace negation_of_universal_quantification_l179_17987

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x^2 + 2*x ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x < 0) := by sorry

end negation_of_universal_quantification_l179_17987


namespace f_composition_l179_17965

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the domain of x
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 5

-- State the theorem
theorem f_composition (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (2 * x - 3) = 4 * x - 5 := by
  sorry

end f_composition_l179_17965


namespace f_of_5_eq_92_l179_17924

/-- Given a function f(x) = 2x^2 + y where f(2) = 50, prove that f(5) = 92 -/
theorem f_of_5_eq_92 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 50) :
  f 5 = 92 := by
  sorry

end f_of_5_eq_92_l179_17924


namespace distinct_permutations_with_repetitions_l179_17993

-- Define the number of elements
def n : ℕ := 5

-- Define the number of repetitions for the first digit
def r1 : ℕ := 3

-- Define the number of repetitions for the second digit
def r2 : ℕ := 2

-- State the theorem
theorem distinct_permutations_with_repetitions :
  (n.factorial) / (r1.factorial * r2.factorial) = 10 := by
  sorry

end distinct_permutations_with_repetitions_l179_17993


namespace min_turns_rook_path_l179_17930

/-- Represents a chessboard --/
structure Chessboard :=
  (files : Nat)
  (ranks : Nat)

/-- Represents a rook's path on a chessboard --/
structure RookPath :=
  (board : Chessboard)
  (turns : Nat)
  (visitsAllSquares : Bool)

/-- Defines a valid rook path that visits all squares exactly once --/
def isValidRookPath (path : RookPath) : Prop :=
  path.board.files = 8 ∧
  path.board.ranks = 8 ∧
  path.visitsAllSquares = true

/-- Theorem: The minimum number of turns for a rook to visit all squares on an 8x8 chessboard is 14 --/
theorem min_turns_rook_path :
  ∀ (path : RookPath), isValidRookPath path → path.turns ≥ 14 :=
by sorry

end min_turns_rook_path_l179_17930


namespace correct_sum_after_error_l179_17907

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a before multiplying by b and adding 35 results in 226,
    then the correct sum of ab + 35 is 54. -/
theorem correct_sum_after_error (a b : ℕ+) : 
  (a.val ≥ 10 ∧ a.val ≤ 99) →
  (((10 * (a.val % 10) + (a.val / 10)) * b.val) + 35 = 226) →
  (a.val * b.val + 35 = 54) :=
by sorry

end correct_sum_after_error_l179_17907


namespace divisible_by_27_l179_17984

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(5*n + 1) + 5^(n + 2) = 27 * k := by
  sorry

end divisible_by_27_l179_17984


namespace cos_sin_shift_l179_17955

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (x/2 - Real.pi/4) = Real.sin (x/2 + Real.pi/4) := by
  sorry

end cos_sin_shift_l179_17955


namespace exercise_books_count_l179_17933

/-- Given a shop with pencils, pens, exercise books, and erasers in the ratio 10 : 2 : 3 : 4,
    where there are 150 pencils, prove that there are 45 exercise books. -/
theorem exercise_books_count (pencils : ℕ) (pens : ℕ) (exercise_books : ℕ) (erasers : ℕ) :
  pencils = 150 →
  10 * pens = 2 * pencils →
  10 * exercise_books = 3 * pencils →
  10 * erasers = 4 * pencils →
  exercise_books = 45 := by
  sorry

end exercise_books_count_l179_17933


namespace binary_conversion_l179_17909

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, false, true]

-- Define the function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the function to convert decimal to base-7
def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 45 ∧
  decimal_to_base7 (binary_to_decimal binary_num) = [6, 3] := by
  sorry

end binary_conversion_l179_17909


namespace max_value_quadratic_l179_17947

theorem max_value_quadratic (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2*x + a^2 - 1 ≤ 16) ∧ 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a^2 - 1 = 16) → 
  a = 3 ∨ a = -3 := by
sorry

end max_value_quadratic_l179_17947


namespace arithmetic_seq_product_l179_17989

/-- An increasing arithmetic sequence of integers -/
def is_increasing_arithmetic_seq (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_seq_product (b : ℕ → ℤ) 
  (h_seq : is_increasing_arithmetic_seq b)
  (h_prod : b 4 * b 5 = 15) :
  b 3 * b 6 = 7 := by
  sorry

end arithmetic_seq_product_l179_17989


namespace problem_1_l179_17977

theorem problem_1 (x y : ℝ) : (x - 2*y)^2 - x*(x + 3*y) - 4*y^2 = -7*x*y := by
  sorry

end problem_1_l179_17977


namespace hyperbola_vertex_distance_l179_17925

def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 48 * x - y^2 + 6 * y + 50 = 0

def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_equation = 2 * Real.sqrt 85 / 3 := by
  sorry

end hyperbola_vertex_distance_l179_17925


namespace sand_pile_volume_l179_17938

/-- Theorem: Volume of a conical sand pile --/
theorem sand_pile_volume (diameter : Real) (height_ratio : Real) :
  diameter = 8 →
  height_ratio = 0.75 →
  let height := height_ratio * diameter
  let radius := diameter / 2
  let volume := (1 / 3) * Real.pi * radius^2 * height
  volume = 32 * Real.pi := by
  sorry

end sand_pile_volume_l179_17938


namespace divisibility_equivalence_l179_17940

theorem divisibility_equivalence (n : ℕ) :
  5 ∣ (1^n + 2^n + 3^n + 4^n) ↔ n % 4 ≠ 0 := by
  sorry

end divisibility_equivalence_l179_17940


namespace basketball_probability_l179_17900

/-- The probability of a basketball player scoring a basket -/
def p : ℚ := 2/3

/-- The number of attempts -/
def n : ℕ := 3

/-- The maximum number of successful baskets we're considering -/
def k : ℕ := 1

/-- The probability of scoring at most once in three attempts -/
def prob_at_most_one : ℚ := 7/27

theorem basketball_probability :
  (Finset.sum (Finset.range (k + 1)) (λ i => Nat.choose n i * p^i * (1 - p)^(n - i))) = prob_at_most_one :=
sorry

end basketball_probability_l179_17900


namespace oranges_in_bowl_l179_17926

def bowl_of_fruit (num_bananas : ℕ) (num_apples : ℕ) (num_oranges : ℕ) : Prop :=
  num_apples = 2 * num_bananas ∧
  num_bananas + num_apples + num_oranges = 12

theorem oranges_in_bowl :
  ∃ (num_oranges : ℕ), bowl_of_fruit 2 (2 * 2) num_oranges ∧ num_oranges = 6 :=
by
  sorry

end oranges_in_bowl_l179_17926


namespace two_digit_numbers_dividing_all_relatives_l179_17937

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_relative (ab n : ℕ) : Prop :=
  is_two_digit ab ∧
  n % 10 = ab % 10 ∧
  ∀ d ∈ (n / 10).digits 10, d ≠ 0 ∧
  digit_sum (n / 10) = ab / 10

def divides_all_relatives (ab : ℕ) : Prop :=
  is_two_digit ab ∧
  ∀ n : ℕ, is_relative ab n → ab ∣ n

theorem two_digit_numbers_dividing_all_relatives :
  {ab : ℕ | divides_all_relatives ab} =
  {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 45, 90} :=
sorry

end two_digit_numbers_dividing_all_relatives_l179_17937


namespace parabola_intersection_and_area_minimization_l179_17992

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the line passing through M and intersecting the parabola
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the dot product of vectors OA and OB
def dot_product (x1 x2 : ℝ) : ℝ := x1 * x2 + (x1^2) * (x2^2)

theorem parabola_intersection_and_area_minimization 
  (k m : ℝ) -- Parameters of the line
  (x1 x2 : ℝ) -- x-coordinates of intersection points A and B
  (h1 : parabola x1 = line k m x1) -- A is on both parabola and line
  (h2 : parabola x2 = line k m x2) -- B is on both parabola and line
  (h3 : dot_product x1 x2 = 2) -- Given condition
  (h4 : m = 2) -- Line passes through (0, 2)
  : 
  (∃ (x : ℝ), line k m x = 2) ∧ -- Line passes through (0, 2)
  (∃ (area : ℝ), area = 3 ∧ 
    ∀ (x : ℝ), x > 0 → x + 9/(4*x) ≥ area) -- Minimum area is 3
  := by sorry

end parabola_intersection_and_area_minimization_l179_17992


namespace sin_square_inequality_l179_17951

theorem sin_square_inequality (n : ℕ+) (x : ℝ) :
  n * (Real.sin x)^2 ≥ Real.sin x * Real.sin (n * x) := by
  sorry

end sin_square_inequality_l179_17951


namespace constant_term_expansion_l179_17957

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x + 1/x - 2)^5
  ∃ (p : ℝ → ℝ), expansion = p x ∧ p 0 = -252 :=
by sorry

end constant_term_expansion_l179_17957


namespace wyatt_remaining_money_l179_17970

/-- Calculates the remaining money after shopping, given the initial amount,
    costs of items, quantities, and discount rate. -/
def remaining_money (initial_amount : ℚ)
                    (bread_cost loaves orange_juice_cost cartons
                     cookie_cost boxes apple_cost pounds
                     chocolate_cost bars : ℚ)
                    (discount_rate : ℚ) : ℚ :=
  let total_cost := bread_cost * loaves +
                    orange_juice_cost * cartons +
                    cookie_cost * boxes +
                    apple_cost * pounds +
                    chocolate_cost * bars
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that Wyatt has $127.60 left after shopping -/
theorem wyatt_remaining_money :
  remaining_money 200
                  6.50 5 3.25 4 2.10 7 1.75 3 2.50 6
                  0.10 = 127.60 := by
  sorry

end wyatt_remaining_money_l179_17970


namespace certain_number_problem_l179_17944

theorem certain_number_problem (x : ℤ) : 17 * (x + 99) = 3111 ↔ x = 84 := by
  sorry

end certain_number_problem_l179_17944


namespace portias_school_students_l179_17945

theorem portias_school_students (portia_students lara_students : ℕ) 
  (h1 : portia_students = 4 * lara_students)
  (h2 : portia_students + lara_students = 2500) : 
  portia_students = 2000 := by
  sorry

end portias_school_students_l179_17945


namespace candle_ratio_problem_l179_17919

/-- Given the ratio of red candles to blue candles and the number of red candles,
    calculate the number of blue candles. -/
theorem candle_ratio_problem (red_candles : ℕ) (red_ratio blue_ratio : ℕ) 
    (h_red : red_candles = 45)
    (h_ratio : red_ratio = 5 ∧ blue_ratio = 3) :
    red_candles * blue_ratio = red_ratio * 27 :=
by sorry

end candle_ratio_problem_l179_17919


namespace nested_triple_op_result_l179_17908

def triple_op (a b c : ℚ) : ℚ := (2 * a + b) / c

def nested_triple_op (x y z : ℚ) : ℚ :=
  triple_op (triple_op 30 60 90) (triple_op 3 6 9) (triple_op 6 12 18)

theorem nested_triple_op_result : nested_triple_op 30 60 90 = 4 := by
  sorry

end nested_triple_op_result_l179_17908


namespace apples_in_first_group_l179_17958

-- Define the cost of an apple
def apple_cost : ℚ := 21/100

-- Define the equation for the first group
def first_group (x : ℚ) (orange_cost : ℚ) : Prop :=
  x * apple_cost + 3 * orange_cost = 177/100

-- Define the equation for the second group
def second_group (orange_cost : ℚ) : Prop :=
  2 * apple_cost + 5 * orange_cost = 127/100

-- Theorem stating that the number of apples in the first group is 6
theorem apples_in_first_group :
  ∃ (orange_cost : ℚ), first_group 6 orange_cost ∧ second_group orange_cost :=
sorry

end apples_in_first_group_l179_17958


namespace reflections_composition_is_translation_l179_17983

/-- Four distinct points on a circle -/
structure CirclePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2 ∧
    (D.1 - center.1)^2 + (D.2 - center.2)^2 = radius^2

/-- Reflection across a line defined by two points -/
def reflect (p q : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Translation of a point -/
def translate (v : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := (x.1 + v.1, x.2 + v.2)

/-- The main theorem stating that the composition of reflections is a translation -/
theorem reflections_composition_is_translation (points : CirclePoints) :
  ∃ (v : ℝ × ℝ), ∀ (x : ℝ × ℝ),
    reflect points.D points.A (reflect points.C points.D (reflect points.B points.C (reflect points.A points.B x))) = translate v x :=
sorry

end reflections_composition_is_translation_l179_17983


namespace binomial_coefficient_x_cubed_in_x_plus_one_to_sixth_l179_17996

theorem binomial_coefficient_x_cubed_in_x_plus_one_to_sixth : 
  (Finset.range 7).sum (fun k => Nat.choose 6 k * X^k) = 
    X^6 + 6*X^5 + 15*X^4 + 20*X^3 + 15*X^2 + 6*X + 1 :=
by sorry

end binomial_coefficient_x_cubed_in_x_plus_one_to_sixth_l179_17996


namespace three_hits_in_five_shots_l179_17981

/-- The probability of hitting the target exactly k times in n independent shots,
    where p is the probability of hitting the target in each shot. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of hitting the target exactly 3 times in 5 shots,
    where the probability of hitting the target in each shot is 0.6,
    is equal to 0.3456. -/
theorem three_hits_in_five_shots :
  binomial_probability 5 3 0.6 = 0.3456 := by
  sorry

end three_hits_in_five_shots_l179_17981


namespace floor_sqrt_80_l179_17974

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l179_17974


namespace common_divisor_problem_l179_17911

theorem common_divisor_problem (n : ℕ) (hn : n < 50) :
  (∃ d : ℕ, d > 1 ∧ d ∣ (3 * n + 5) ∧ d ∣ (5 * n + 4)) ↔ n ∈ ({7, 20, 33, 46} : Set ℕ) :=
by sorry

end common_divisor_problem_l179_17911


namespace greatest_divisor_with_remainder_l179_17914

theorem greatest_divisor_with_remainder (a b c : ℕ) (h : a = 263 ∧ b = 935 ∧ c = 1383) :
  (∃ (d : ℕ), d > 0 ∧ 
    (a % d = 7 ∧ b % d = 7 ∧ c % d = 7) ∧
    (∀ (k : ℕ), k > d → (a % k ≠ 7 ∨ b % k ≠ 7 ∨ c % k ≠ 7))) →
  (∃ (d : ℕ), d = 16 ∧
    (a % d = 7 ∧ b % d = 7 ∧ c % d = 7) ∧
    (∀ (k : ℕ), k > d → (a % k ≠ 7 ∨ b % k ≠ 7 ∨ c % k ≠ 7))) :=
by sorry

end greatest_divisor_with_remainder_l179_17914


namespace rearrangements_without_substring_l179_17979

def string_length : ℕ := 8
def h_count : ℕ := 2
def m_count : ℕ := 4
def t_count : ℕ := 2

def total_arrangements : ℕ := string_length.factorial / (h_count.factorial * m_count.factorial * t_count.factorial)

def substring_length : ℕ := 4
def remaining_string_length : ℕ := string_length - substring_length + 1

def arrangements_with_substring : ℕ := 
  (remaining_string_length.factorial / (h_count.pred.factorial * m_count.pred.pred.pred.factorial)) * 
  (substring_length.factorial / m_count.pred.factorial)

theorem rearrangements_without_substring : 
  total_arrangements - arrangements_with_substring + 1 = 361 := by sorry

end rearrangements_without_substring_l179_17979


namespace intersection_A_B_union_A_B_l179_17949

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = x^2 - 4*x + 3}
def B : Set ℝ := {y | ∃ x, y = -x^2 - 2*x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {y | -1 ≤ y ∧ y ≤ 1} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = Set.univ := by sorry

end intersection_A_B_union_A_B_l179_17949


namespace smallest_fraction_l179_17917

theorem smallest_fraction (S : Set ℚ) (h : S = {1/2, 2/3, 1/4, 5/6, 7/12}) :
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 1/4 :=
by
  sorry

end smallest_fraction_l179_17917


namespace problem_1_problem_2_l179_17964

theorem problem_1 : (-1/2)⁻¹ + (3 - Real.pi)^0 + (-3)^2 = 8 := by sorry

theorem problem_2 (a : ℝ) : a^2 * a^4 - (-2*a^2)^3 - 3*a^2 + a^2 = 9*a^6 - 2*a^2 := by sorry

end problem_1_problem_2_l179_17964


namespace rectangular_wall_area_l179_17976

theorem rectangular_wall_area : 
  ∀ (width length area : ℝ),
    width = 5.4 →
    length = 2.5 →
    area = width * length →
    area = 13.5 := by
  sorry

end rectangular_wall_area_l179_17976


namespace complement_of_A_l179_17956

-- Define the set A
def A : Set ℝ := {x : ℝ | x ≤ 1}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | x > 1} := by sorry

end complement_of_A_l179_17956


namespace oak_trees_after_planting_l179_17910

/-- The number of oak trees in a park after planting new trees -/
theorem oak_trees_after_planting (current : ℕ) (new : ℕ) : current = 5 → new = 4 → current + new = 9 := by
  sorry

end oak_trees_after_planting_l179_17910


namespace cubic_inequality_iff_open_interval_l179_17954

theorem cubic_inequality_iff_open_interval :
  ∀ x : ℝ, x * (x^2 - 9) < 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 3 := by sorry

end cubic_inequality_iff_open_interval_l179_17954


namespace reading_time_calculation_gwendolyn_reading_time_l179_17980

theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : ℕ :=
  let sentences_per_page := paragraphs_per_page * sentences_per_paragraph
  let total_sentences := sentences_per_page * total_pages
  total_sentences / reading_speed

theorem gwendolyn_reading_time : 
  reading_time_calculation 300 40 20 150 = 400 := by
  sorry

end reading_time_calculation_gwendolyn_reading_time_l179_17980


namespace handshake_count_l179_17906

theorem handshake_count (n : ℕ) (h : n = 6) : 
  n * 2 * (n * 2 - 2) / 2 = 60 := by
  sorry

#check handshake_count

end handshake_count_l179_17906


namespace girls_fraction_l179_17986

theorem girls_fraction (T G B : ℝ) (x : ℝ) 
  (h1 : x * G = (1 / 5) * T)  -- Some fraction of girls is 1/5 of total
  (h2 : B / G = 1.5)          -- Ratio of boys to girls is 1.5
  (h3 : T = B + G)            -- Total is sum of boys and girls
  : x = 1 / 2 := by 
  sorry

end girls_fraction_l179_17986


namespace product_of_numbers_l179_17959

theorem product_of_numbers (x y : ℝ) : x + y = 24 → x - y = 8 → x * y = 128 := by
  sorry

end product_of_numbers_l179_17959


namespace cubic_root_product_l179_17952

theorem cubic_root_product (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + a - 5 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 5 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 5 = 0) → 
  a * b * c = 5/3 := by
sorry

end cubic_root_product_l179_17952


namespace total_turnips_l179_17928

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end total_turnips_l179_17928


namespace sequence_sum_l179_17973

def a : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * a (n + 2) - 4 * (n + 3) * a (n + 1) + (4 * (n + 3) - 8) * a n

theorem sequence_sum (n : ℕ) : a n = n.factorial + 2^n := by
  sorry

end sequence_sum_l179_17973


namespace train_distance_l179_17939

/-- Proves that a train traveling at a speed derived from covering 2 miles in 2 minutes will travel 180 miles in 3 hours. -/
theorem train_distance (distance : ℝ) (time : ℝ) (hours : ℝ) : 
  distance = 2 → time = 2 → hours = 3 → (distance / time) * (hours * 60) = 180 := by
  sorry

end train_distance_l179_17939


namespace polynomial_simplification_l179_17912

theorem polynomial_simplification (x : ℝ) :
  (3*x^2 + 5*x + 8)*(x - 2) - (x - 2)*(x^2 + 6*x - 72) + (2*x - 15)*(x - 2)*(x + 4) =
  4*x^3 - 17*x^2 + 38*x - 40 := by
  sorry

end polynomial_simplification_l179_17912


namespace first_discount_percentage_l179_17990

/-- Proves that the first discount percentage is 20% given the conditions of the problem -/
theorem first_discount_percentage (original_price : ℝ) (second_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 400)
  (h2 : second_discount = 15)
  (h3 : final_price = 272)
  (h4 : final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100)) :
  first_discount = 20 := by
  sorry


end first_discount_percentage_l179_17990


namespace arrangeStudents_eq_288_l179_17953

/-- The number of ways to arrange 6 students in a 2x3 grid with constraints -/
def arrangeStudents : ℕ :=
  let totalStudents : ℕ := 6
  let rows : ℕ := 2
  let columns : ℕ := 3
  let positionsForA : ℕ := totalStudents
  let positionsForB : ℕ := 2  -- Not in same row or column as A
  let remainingStudents : ℕ := totalStudents - 2
  positionsForA * positionsForB * Nat.factorial remainingStudents

theorem arrangeStudents_eq_288 : arrangeStudents = 288 := by
  sorry

end arrangeStudents_eq_288_l179_17953


namespace exponential_equation_comparison_l179_17971

theorem exponential_equation_comparison
  (c d k n : ℝ)
  (hc : c > 0)
  (hk : k > 0)
  (hd : d ≠ 0)
  (hn : n ≠ 0) :
  (Real.log c / Real.log k < d / n) ↔
  ((1 / d) * Real.log (1 / c) < (1 / n) * Real.log (1 / k)) :=
sorry

end exponential_equation_comparison_l179_17971


namespace slope_of_line_slope_of_specific_line_l179_17929

/-- The slope of a line given by the equation y + ax - b = 0 is -a. -/
theorem slope_of_line (a b : ℝ) : 
  (fun x y : ℝ => y + a * x - b = 0) = (fun x y : ℝ => y = -a * x + b) := by
  sorry

/-- The slope of the line y + 3x - 1 = 0 is -3. -/
theorem slope_of_specific_line : 
  (fun x y : ℝ => y + 3 * x - 1 = 0) = (fun x y : ℝ => y = -3 * x + 1) := by
  sorry

end slope_of_line_slope_of_specific_line_l179_17929


namespace gardeners_mowing_time_l179_17932

theorem gardeners_mowing_time (rate_A rate_B : ℚ) (h1 : rate_A = 1 / 3) (h2 : rate_B = 1 / 5) :
  1 / (rate_A + rate_B) = 15 / 8 := by
  sorry

end gardeners_mowing_time_l179_17932


namespace sum_of_squares_l179_17942

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 10)
  (eq2 : b^2 + 5*c = -10)
  (eq3 : c^2 + 7*a = -21) :
  a^2 + b^2 + c^2 = 20.75 := by
sorry

end sum_of_squares_l179_17942


namespace bmw_sales_l179_17950

theorem bmw_sales (total : ℕ) (mercedes_percent : ℚ) (nissan_percent : ℚ) (ford_percent : ℚ) (chevrolet_percent : ℚ) 
  (h_total : total = 300)
  (h_mercedes : mercedes_percent = 20 / 100)
  (h_nissan : nissan_percent = 25 / 100)
  (h_ford : ford_percent = 10 / 100)
  (h_chevrolet : chevrolet_percent = 18 / 100) :
  ↑(total - (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent).num * total / (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent).den) = 81 := by
  sorry


end bmw_sales_l179_17950


namespace polynomial_evaluation_l179_17994

theorem polynomial_evaluation : (4 : ℝ)^4 + (4 : ℝ)^3 + (4 : ℝ)^2 + (4 : ℝ) + 1 = 341 := by sorry

end polynomial_evaluation_l179_17994


namespace hall_dark_tile_fraction_l179_17948

/-- Represents a tiling pattern on a floor -/
structure TilingPattern :=
  (size : Nat)
  (dark_tiles_in_section : Nat)
  (section_size : Nat)

/-- The fraction of dark tiles in a tiling pattern -/
def dark_tile_fraction (pattern : TilingPattern) : Rat :=
  pattern.dark_tiles_in_section / (pattern.section_size * pattern.section_size)

/-- Theorem stating that for the given tiling pattern, the fraction of dark tiles is 5/8 -/
theorem hall_dark_tile_fraction :
  ∀ (pattern : TilingPattern),
    pattern.size = 8 ∧
    pattern.section_size = 4 ∧
    pattern.dark_tiles_in_section = 10 →
    dark_tile_fraction pattern = 5 / 8 :=
by
  sorry

end hall_dark_tile_fraction_l179_17948


namespace leak_empty_time_l179_17903

/-- Given a pipe that can fill a tank in 6 hours, and with a leak it takes 8 hours to fill the tank,
    prove that the leak alone will empty the full tank in 24 hours. -/
theorem leak_empty_time (fill_rate : ℝ) (combined_rate : ℝ) (leak_rate : ℝ) : 
  fill_rate = 1 / 6 →
  combined_rate = 1 / 8 →
  combined_rate = fill_rate - leak_rate →
  1 / leak_rate = 24 := by
sorry

end leak_empty_time_l179_17903


namespace quadratic_roots_condition_l179_17969

theorem quadratic_roots_condition (a : ℝ) (h1 : a ≠ 0) (h2 : a < -1) :
  ∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 < 0 ∧ 
  (a * x1^2 + 2 * x1 + 1 = 0) ∧ 
  (a * x2^2 + 2 * x2 + 1 = 0) :=
sorry

end quadratic_roots_condition_l179_17969


namespace f_difference_at_five_l179_17946

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^4 + x^3 + 5*x

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 6550 := by
  sorry

end f_difference_at_five_l179_17946


namespace complement_intersection_theorem_l179_17918

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 3} := by sorry

end complement_intersection_theorem_l179_17918


namespace scientific_notation_826_million_l179_17997

theorem scientific_notation_826_million : 
  826000000 = 8.26 * (10 : ℝ)^8 := by sorry

end scientific_notation_826_million_l179_17997


namespace average_difference_l179_17988

def number_of_students : ℕ := 120
def number_of_teachers : ℕ := 4
def class_sizes : List ℕ := [60, 30, 20, 10]

def t : ℚ := (List.sum class_sizes) / number_of_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / number_of_students

theorem average_difference : t - s = -11663/1000 := by
  sorry

end average_difference_l179_17988


namespace intersection_constraint_l179_17967

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 3}

def N (m b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m*p.1 + b}

theorem intersection_constraint (b : ℝ) :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) → b ∈ Set.Icc (-Real.sqrt (3/2)) (Real.sqrt (3/2)) :=
sorry

end intersection_constraint_l179_17967


namespace no_m_exists_for_equality_necessary_but_not_sufficient_condition_l179_17963

-- Define set P
def P : Set ℝ := {x : ℝ | x^2 - 8*x - 20 ≤ 0}

-- Define set S as a function of m
def S (m : ℝ) : Set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for part I
theorem no_m_exists_for_equality :
  ¬ ∃ m : ℝ, P = S m :=
sorry

-- Theorem for part II
theorem necessary_but_not_sufficient_condition :
  ∀ m : ℝ, m ≤ 3 → (P ⊆ S m ∧ P ≠ S m) :=
sorry

end no_m_exists_for_equality_necessary_but_not_sufficient_condition_l179_17963


namespace quadratic_roots_property_l179_17998

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 + 2*α - 2005 = 0) → 
  (β^2 + 2*β - 2005 = 0) → 
  (α^2 + 3*α + β = 2003) := by
sorry

end quadratic_roots_property_l179_17998
