import Mathlib

namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l114_11478

/-- The difference in miles biked between Alberto and Bjorn after four hours -/
theorem alberto_bjorn_distance_difference :
  let alberto_distance : ℕ := 60
  let bjorn_distance : ℕ := 45
  alberto_distance - bjorn_distance = 15 := by
sorry

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l114_11478


namespace NUMINAMATH_CALUDE_line_moved_down_l114_11476

/-- Given a line with equation y = 2x + 3, prove that moving it down by 5 units
    results in the equation y = 2x - 2. -/
theorem line_moved_down (x y : ℝ) :
  (y = 2 * x + 3) → (y - 5 = 2 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_line_moved_down_l114_11476


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l114_11418

theorem quadratic_equation_solution :
  ∀ (a b : ℝ),
  (∀ x : ℝ, x^2 - 6*x + 18 = 28 ↔ (x = a ∨ x = b)) →
  a ≥ b →
  a + 3*b = 12 - 2*Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l114_11418


namespace NUMINAMATH_CALUDE_cone_volume_l114_11474

/-- Given a cone with lateral area 20π and angle between slant height and base arccos(4/5),
    prove that its volume is 16π. -/
theorem cone_volume (r l h : ℝ) (lateral_area : ℝ) (angle : ℝ) : 
  lateral_area = 20 * Real.pi →
  angle = Real.arccos (4/5) →
  r / l = 4 / 5 →
  lateral_area = Real.pi * r * l →
  h = Real.sqrt (l^2 - r^2) →
  (1/3) * Real.pi * r^2 * h = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l114_11474


namespace NUMINAMATH_CALUDE_negative_two_times_negative_three_l114_11404

theorem negative_two_times_negative_three : (-2) * (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_negative_three_l114_11404


namespace NUMINAMATH_CALUDE_boys_girls_relation_l114_11484

/-- Represents the number of girls a boy dances with based on his position -/
def girls_danced_with (n : ℕ) : ℕ := 2 * n + 1

/-- 
Theorem: In a class where boys dance with girls following a specific pattern,
the number of boys is related to the number of girls by b = (g - 1) / 2.
-/
theorem boys_girls_relation (b g : ℕ) (h1 : b > 0) (h2 : g > 0) 
  (h3 : ∀ n, n ∈ Finset.range b → girls_danced_with n ≤ g) 
  (h4 : girls_danced_with b = g) : 
  b = (g - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_boys_girls_relation_l114_11484


namespace NUMINAMATH_CALUDE_blanket_price_problem_l114_11475

theorem blanket_price_problem (price1 price2 avg_price : ℕ) 
  (count1 count2 count_unknown : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 150 →
  count1 = 3 →
  count2 = 3 →
  count_unknown = 2 →
  (count1 * price1 + count2 * price2 + count_unknown * 
    ((count1 + count2 + count_unknown) * avg_price - 
     count1 * price1 - count2 * price2) / count_unknown) / 
    (count1 + count2 + count_unknown) = avg_price →
  ((count1 + count2 + count_unknown) * avg_price - 
   count1 * price1 - count2 * price2) / count_unknown = 225 :=
by sorry

end NUMINAMATH_CALUDE_blanket_price_problem_l114_11475


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l114_11447

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 1) (h2 : x * y = 2 * x + y + 2) :
  x + y ≥ 7 ∧ ∃ x0 y0, x0 > 1 ∧ x0 * y0 = 2 * x0 + y0 + 2 ∧ x0 + y0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l114_11447


namespace NUMINAMATH_CALUDE_pascal_triangle_ratio_l114_11477

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The row number in Pascal's Triangle -/
def n : ℕ := 53

/-- The position of the first entry in the consecutive trio -/
def r : ℕ := 23

/-- Theorem stating that three consecutive entries in row 53 of Pascal's Triangle are in the ratio 4:5:6 -/
theorem pascal_triangle_ratio :
  ∃ (r : ℕ), r < n ∧ 
    (choose n r : ℚ) / (choose n (r + 1)) = 4 / 5 ∧
    (choose n (r + 1) : ℚ) / (choose n (r + 2)) = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_pascal_triangle_ratio_l114_11477


namespace NUMINAMATH_CALUDE_rhinestone_problem_l114_11409

theorem rhinestone_problem (total : ℕ) (bought_fraction : ℚ) (found_fraction : ℚ) : 
  total = 45 → 
  bought_fraction = 1/3 → 
  found_fraction = 1/5 → 
  total - (total * bought_fraction).floor - (total * found_fraction).floor = 21 := by
  sorry

end NUMINAMATH_CALUDE_rhinestone_problem_l114_11409


namespace NUMINAMATH_CALUDE_discount_problem_l114_11414

/-- Proves that if a 25% discount on a purchase is $40, then the total amount paid after the discount is $120. -/
theorem discount_problem (original_price : ℝ) (discount_amount : ℝ) (discount_percentage : ℝ) 
  (h1 : discount_amount = 40)
  (h2 : discount_percentage = 0.25)
  (h3 : discount_amount = discount_percentage * original_price) :
  original_price - discount_amount = 120 := by
  sorry

#check discount_problem

end NUMINAMATH_CALUDE_discount_problem_l114_11414


namespace NUMINAMATH_CALUDE_mean_weight_of_participants_l114_11411

/-- Represents a stem and leaf plot entry -/
structure StemLeafEntry :=
  (stem : ℕ)
  (leaves : List ℕ)

/-- Calculates the sum of weights from a stem and leaf entry -/
def sumWeights (entry : StemLeafEntry) : ℕ :=
  entry.leaves.sum + entry.stem * 100 * entry.leaves.length

/-- Calculates the number of participants from a stem and leaf entry -/
def countParticipants (entry : StemLeafEntry) : ℕ :=
  entry.leaves.length

theorem mean_weight_of_participants (data : List StemLeafEntry) 
  (h1 : data = [
    ⟨12, [3, 5]⟩, 
    ⟨13, [0, 2, 3, 5, 7, 8]⟩, 
    ⟨14, [1, 5, 5, 9, 9]⟩, 
    ⟨15, [0, 2, 3, 5, 8]⟩, 
    ⟨16, [4, 7, 7, 9]⟩
  ]) : 
  (data.map sumWeights).sum / (data.map countParticipants).sum = 3217 / 22 := by
  sorry

end NUMINAMATH_CALUDE_mean_weight_of_participants_l114_11411


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l114_11495

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5)
  (h_mean2 : (a 3 + a 7) / 2 = 7) :
  ∃ (b c : ℝ), ∀ n : ℕ, a n = b * n + c ∧ b = 2 ∧ c = -3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l114_11495


namespace NUMINAMATH_CALUDE_product_equals_two_thirds_l114_11412

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 1 + (a n - 1)^2

-- Define the infinite product of a_n
def infiniteProduct : ℚ := sorry

-- Theorem statement
theorem product_equals_two_thirds : infiniteProduct = 2/3 := by sorry

end NUMINAMATH_CALUDE_product_equals_two_thirds_l114_11412


namespace NUMINAMATH_CALUDE_probability_all_female_finalists_l114_11406

def total_contestants : ℕ := 7
def female_contestants : ℕ := 4
def male_contestants : ℕ := 3
def finalists : ℕ := 3

theorem probability_all_female_finalists :
  (Nat.choose female_contestants finalists : ℚ) / (Nat.choose total_contestants finalists : ℚ) = 4 / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_all_female_finalists_l114_11406


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l114_11479

theorem square_area_equal_perimeter_triangle (a b c : ℝ) (square_side : ℝ) : 
  a = 5.8 ∧ b = 7.5 ∧ c = 10.7 →
  4 * square_side = a + b + c →
  square_side ^ 2 = 36 := by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l114_11479


namespace NUMINAMATH_CALUDE_max_value_expression_l114_11453

theorem max_value_expression (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (k * x + y)^2 / (x^2 + y^2) ≤ k^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l114_11453


namespace NUMINAMATH_CALUDE_composite_rectangle_area_l114_11415

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A larger rectangle composed of three identical smaller rectangles -/
structure CompositeRectangle where
  smallRectangle : Rectangle
  count : ℕ

/-- The area of the composite rectangle -/
def CompositeRectangle.area (cr : CompositeRectangle) : ℝ :=
  cr.smallRectangle.area * cr.count

theorem composite_rectangle_area :
  ∀ (r : Rectangle),
    r.width = 8 →
    (CompositeRectangle.area { smallRectangle := r, count := 3 }) = 384 :=
by
  sorry

end NUMINAMATH_CALUDE_composite_rectangle_area_l114_11415


namespace NUMINAMATH_CALUDE_h_perimeter_is_26_l114_11428

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Calculates the perimeter of an H-shaped figure formed by three rectangles -/
def hPerimeter (r : Rectangle) : ℝ :=
  2 * r.length + 4 * r.width + 2 * r.length

/-- Theorem: The perimeter of an H-shaped figure formed by three 3x5 inch rectangles is 26 inches -/
theorem h_perimeter_is_26 :
  let r : Rectangle := { length := 5, width := 3 }
  hPerimeter r = 26 := by
  sorry

end NUMINAMATH_CALUDE_h_perimeter_is_26_l114_11428


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l114_11466

/-- Calculates the percentage of yellow tint in an updated mixture -/
theorem yellow_tint_percentage 
  (original_volume : ℝ) 
  (original_yellow_percentage : ℝ) 
  (added_yellow : ℝ) : 
  original_volume = 20 →
  original_yellow_percentage = 0.5 →
  added_yellow = 6 →
  let original_yellow := original_volume * original_yellow_percentage
  let total_yellow := original_yellow + added_yellow
  let new_volume := original_volume + added_yellow
  (total_yellow / new_volume) * 100 = 61.5 := by
sorry

end NUMINAMATH_CALUDE_yellow_tint_percentage_l114_11466


namespace NUMINAMATH_CALUDE_abs_x_minus_one_leq_two_solution_set_l114_11459

theorem abs_x_minus_one_leq_two_solution_set :
  {x : ℝ | |x - 1| ≤ 2} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_leq_two_solution_set_l114_11459


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_l114_11496

theorem sum_real_imag_parts (z : ℂ) : z = 1 + I → (z.re + z.im = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_l114_11496


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_mixture_l114_11439

/-- Given a mixture of water and an alcohol solution, calculate the percentage of alcohol in the new mixture. -/
theorem alcohol_percentage_in_mixture 
  (water_volume : ℝ) 
  (solution_volume : ℝ) 
  (original_alcohol_percentage : ℝ) : 
  water_volume = 16 → 
  solution_volume = 24 → 
  original_alcohol_percentage = 90 → 
  (original_alcohol_percentage / 100 * solution_volume) / (water_volume + solution_volume) * 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_mixture_l114_11439


namespace NUMINAMATH_CALUDE_three_digit_primes_exist_l114_11446

theorem three_digit_primes_exist : 
  ∃ (S : Finset Nat), 
    (1 ≤ S.card ∧ S.card ≤ 10) ∧ 
    (∀ p ∈ S, 100 ≤ p ∧ p ≤ 999 ∧ Nat.Prime p) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_primes_exist_l114_11446


namespace NUMINAMATH_CALUDE_distance_AB_l114_11448

/-- The distance between points A(2,1) and B(5,-1) is √13. -/
theorem distance_AB : Real.sqrt 13 = Real.sqrt ((5 - 2)^2 + (-1 - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_l114_11448


namespace NUMINAMATH_CALUDE_range_of_m_l114_11422

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = (1/2)^x}

-- Define the set N
def N (m : ℝ) : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = (1/(m-1) + 1)*(x-1) + (|m|-1)*(x-2)}

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l114_11422


namespace NUMINAMATH_CALUDE_problem_statement_l114_11461

theorem problem_statement (A B : ℝ) :
  (∀ x : ℝ, x ≠ 5 → A / (x - 5) + B * (x + 1) = (-2 * x^2 + 16 * x + 18) / (x - 5)) →
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l114_11461


namespace NUMINAMATH_CALUDE_triangle_altitude_proof_l114_11426

theorem triangle_altitude_proof (a b c h : ℝ) : 
  a = 13 ∧ b = 15 ∧ c = 22 →
  a + b > c ∧ a + c > b ∧ b + c > a →
  h = (30 * Real.sqrt 10) / 11 →
  (1 / 2) * c * h = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_proof_l114_11426


namespace NUMINAMATH_CALUDE_ellipse_and_quadratic_conditions_l114_11485

/-- Represents an ellipse equation with parameter a -/
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*a) + y^2 / (3*a - 6) = 1

/-- Checks if the ellipse has foci on the x-axis -/
def has_foci_on_x_axis (a : ℝ) : Prop :=
  2*a < 3*a - 6

/-- Represents the quadratic inequality with parameter a -/
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 + (a + 4)*x + 16 > 0

/-- Checks if the solution set of the quadratic inequality is ℝ -/
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_inequality a x

/-- The main theorem stating the conditions for a -/
theorem ellipse_and_quadratic_conditions (a : ℝ) :
  (is_ellipse a ∧ has_foci_on_x_axis a ∧ solution_set_is_reals a) ↔ (2 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_quadratic_conditions_l114_11485


namespace NUMINAMATH_CALUDE_four_tangent_lines_l114_11499

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if two circles are on the same side of a line -/
def sameSideOfLine (A B : Circle) (m : Line) : Prop := sorry

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Function to reflect a line over another line -/
def reflectLine (l : Line) (m : Line) : Line := sorry

/-- The main theorem -/
theorem four_tangent_lines (A B : Circle) (m : Line) 
  (h : sameSideOfLine A B m) : 
  ∃ (l₁ l₂ l₃ l₄ : Line), 
    (isTangent l₁ A ∧ isTangent (reflectLine l₁ m) B) ∧
    (isTangent l₂ A ∧ isTangent (reflectLine l₂ m) B) ∧
    (isTangent l₃ A ∧ isTangent (reflectLine l₃ m) B) ∧
    (isTangent l₄ A ∧ isTangent (reflectLine l₄ m) B) ∧
    (l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₁ ≠ l₄ ∧ l₂ ≠ l₃ ∧ l₂ ≠ l₄ ∧ l₃ ≠ l₄) :=
by sorry

end NUMINAMATH_CALUDE_four_tangent_lines_l114_11499


namespace NUMINAMATH_CALUDE_certain_multiple_proof_l114_11427

theorem certain_multiple_proof (n : ℝ) (m : ℝ) (h1 : n = 5) (h2 : 7 * n - 15 = m * n + 10) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_multiple_proof_l114_11427


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l114_11434

/-- Given two quadratic equations with coefficients a, b, c, d where the roots of
    a²x² + bx + c = 0 are 2011 times the roots of cx² + dx + a = 0,
    prove that b² = d² -/
theorem quadratic_root_relation (a b c d : ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), c * x₁^2 + d * x₁ + a = 0 ∧ c * x₂^2 + d * x₂ + a = 0 → 
       a^2 * (2011 * x₁)^2 + b * (2011 * x₁) + c = 0 ∧ 
       a^2 * (2011 * x₂)^2 + b * (2011 * x₂) + c = 0) : 
  b^2 = d^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l114_11434


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l114_11457

/-- Represents a trapezoid ABCD with given side lengths and perimeter -/
structure Trapezoid where
  ab : ℝ
  ad : ℝ
  bc : ℝ
  perimeter : ℝ

/-- Calculates the length of CD in the trapezoid -/
def calculate_cd (t : Trapezoid) : ℝ :=
  t.perimeter - (t.ab + t.ad + t.bc)

/-- Theorem stating that for a trapezoid with given measurements, CD = 16 -/
theorem trapezoid_cd_length (t : Trapezoid) 
  (h1 : t.ab = 12)
  (h2 : t.ad = 5)
  (h3 : t.bc = 7)
  (h4 : t.perimeter = 40) : 
  calculate_cd t = 16 := by
  sorry

#eval calculate_cd { ab := 12, ad := 5, bc := 7, perimeter := 40 }

end NUMINAMATH_CALUDE_trapezoid_cd_length_l114_11457


namespace NUMINAMATH_CALUDE_max_concert_tickets_l114_11423

theorem max_concert_tickets (ticket_price : ℕ) (budget : ℕ) : 
  ticket_price = 15 → budget = 120 → 
  ∃ (max_tickets : ℕ), max_tickets = 8 ∧ 
    (∀ n : ℕ, n * ticket_price ≤ budget → n ≤ max_tickets) :=
by sorry

end NUMINAMATH_CALUDE_max_concert_tickets_l114_11423


namespace NUMINAMATH_CALUDE_dan_placed_16_pencils_l114_11467

/-- The number of pencils Dan placed on the desk -/
def pencils_dan_placed (drawer : ℕ) (desk_initial : ℕ) (total_after : ℕ) : ℕ :=
  total_after - (drawer + desk_initial)

/-- Theorem stating that Dan placed 16 pencils on the desk -/
theorem dan_placed_16_pencils : 
  pencils_dan_placed 43 19 78 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dan_placed_16_pencils_l114_11467


namespace NUMINAMATH_CALUDE_zoo_new_species_l114_11489

theorem zoo_new_species (initial_types : ℕ) (time_per_type : ℕ) (total_time_after : ℕ) : 
  initial_types = 5 → 
  time_per_type = 6 → 
  total_time_after = 54 → 
  (initial_types + (total_time_after / time_per_type - initial_types)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_zoo_new_species_l114_11489


namespace NUMINAMATH_CALUDE_sin_600_degrees_l114_11401

theorem sin_600_degrees : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l114_11401


namespace NUMINAMATH_CALUDE_reflection_composition_l114_11410

/-- Two lines in the xy-plane that intersect at the origin -/
structure IntersectingLines where
  ℓ₁ : Set (ℝ × ℝ)
  ℓ₂ : Set (ℝ × ℝ)
  intersect_origin : (0, 0) ∈ ℓ₁ ∩ ℓ₂

/-- A point in the xy-plane -/
def Point := ℝ × ℝ

/-- Reflection of a point over a line -/
def reflect (p : Point) (ℓ : Set Point) : Point := sorry

theorem reflection_composition 
  (lines : IntersectingLines)
  (Q : Point)
  (h₁ : Q = (-2, 3))
  (h₂ : lines.ℓ₁ = {(x, y) | 3 * x - y = 0})
  (h₃ : reflect (reflect Q lines.ℓ₁) lines.ℓ₂ = (5, -2)) :
  lines.ℓ₂ = {(x, y) | x + 4 * y = 0} := by
  sorry

end NUMINAMATH_CALUDE_reflection_composition_l114_11410


namespace NUMINAMATH_CALUDE_complex_multiplication_complex_division_l114_11483

-- Define complex numbers
def i : ℂ := Complex.I

-- Part 1
theorem complex_multiplication :
  (1 - 2*i) * (3 + 4*i) * (-2 + i) = -20 + 15*i := by sorry

-- Part 2
theorem complex_division (x a : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : x > a) :
  (Complex.ofReal x) / (Complex.ofReal (x - a)) = -1/5 + 2/5*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_complex_division_l114_11483


namespace NUMINAMATH_CALUDE_two_sqrt_three_in_set_l114_11480

theorem two_sqrt_three_in_set : 2 * Real.sqrt 3 ∈ {x : ℝ | x < 4} := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_three_in_set_l114_11480


namespace NUMINAMATH_CALUDE_triangle_area_zero_l114_11498

theorem triangle_area_zero (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + b*c + a*c = 11 →
  a*b*c = 6 →
  ∃ (s : ℝ), s*(s - a)*(s - b)*(s - c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_zero_l114_11498


namespace NUMINAMATH_CALUDE_minimize_distance_l114_11491

/-- Given points P and Q in the xy-plane, and R on the line y = 2x - 4,
    prove that the value of n that minimizes PR + RQ is 0 -/
theorem minimize_distance (P Q R : ℝ × ℝ) : 
  P = (-1, -3) →
  Q = (5, 3) →
  R.1 = 2 →
  R.2 = 2 * R.1 - 4 →
  (∀ S : ℝ × ℝ, S.1 = 2 ∧ S.2 = 2 * S.1 - 4 → 
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ≤
    Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) + Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2)) →
  R.2 = 0 := by
sorry

end NUMINAMATH_CALUDE_minimize_distance_l114_11491


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l114_11441

/-- Given the initial conditions of a pricing problem, prove that the profit-maximizing price is 95 yuan. -/
theorem profit_maximizing_price 
  (initial_cost : ℝ)
  (initial_price : ℝ)
  (initial_units : ℝ)
  (price_increase : ℝ)
  (units_decrease : ℝ)
  (h1 : initial_cost = 80)
  (h2 : initial_price = 90)
  (h3 : initial_units = 400)
  (h4 : price_increase = 1)
  (h5 : units_decrease = 20)
  : ∃ (max_price : ℝ), max_price = 95 ∧ 
    ∀ (x : ℝ), 
      (initial_price + x) * (initial_units - units_decrease * x) - initial_cost * (initial_units - units_decrease * x) ≤ 
      (initial_price + (max_price - initial_price)) * (initial_units - units_decrease * (max_price - initial_price)) - 
      initial_cost * (initial_units - units_decrease * (max_price - initial_price)) :=
by sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l114_11441


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l114_11405

/-- Calculates the average speed for a round trip given uphill and downhill times and distances -/
theorem round_trip_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (h1 : uphill_distance = 2)
  (h2 : uphill_time = 45 / 60)
  (h3 : downhill_distance = 2)
  (h4 : downhill_time = 15 / 60)
  : (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l114_11405


namespace NUMINAMATH_CALUDE_min_number_for_triangle_l114_11407

/-- A function that checks if three numbers can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The property that any 17 numbers chosen from 1 to 2005 always contain a triangle -/
def always_contains_triangle (n : ℕ) : Prop :=
  ∀ (s : Finset ℕ), s.card = n → (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2005) →
    ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ can_form_triangle a b c

/-- The theorem stating that 17 is the minimum number for which the property holds -/
theorem min_number_for_triangle :
  always_contains_triangle 17 ∧ ¬(always_contains_triangle 16) :=
sorry

end NUMINAMATH_CALUDE_min_number_for_triangle_l114_11407


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l114_11402

-- Define the sets A and B
def A (a : ℝ) := { x : ℝ | |x - (a+1)^2/2| ≤ (a-1)^2/2 }
def B (a : ℝ) := { x : ℝ | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0 }

-- Define the subset relation
def is_subset (S T : Set ℝ) := ∀ x, x ∈ S → x ∈ T

-- State the theorem
theorem range_of_a_for_subset : 
  { a : ℝ | is_subset (A a) (B a) } = Set.union (Set.Icc 1 3) {-1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l114_11402


namespace NUMINAMATH_CALUDE_prob_more_ones_than_sixes_l114_11471

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling numDice dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of outcomes where the same number of 1's and 6's are rolled -/
def sameOnesSixes : ℕ := 2424

/-- The probability of rolling more 1's than 6's when rolling numDice fair numSides-sided dice -/
def probMoreOnesThanSixes : ℚ := 2676 / 7776

theorem prob_more_ones_than_sixes :
  probMoreOnesThanSixes = 1 / 2 * (1 - sameOnesSixes / totalOutcomes) :=
sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_sixes_l114_11471


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l114_11451

theorem polynomial_divisibility : ∀ (x : ℂ),
  (x^5 + x^4 + x^3 + x^2 + x + 1 = 0) →
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l114_11451


namespace NUMINAMATH_CALUDE_student_grouping_l114_11425

/-- Calculates the minimum number of groups needed to split students -/
def minGroups (totalStudents : ℕ) (maxGroupSize : ℕ) : ℕ :=
  (totalStudents + maxGroupSize - 1) / maxGroupSize

theorem student_grouping (totalStudents : ℕ) (maxGroupSize : ℕ) 
  (h1 : totalStudents = 30) (h2 : maxGroupSize = 12) :
  minGroups totalStudents maxGroupSize = 3 := by
  sorry

#eval minGroups 30 12  -- Should output 3

end NUMINAMATH_CALUDE_student_grouping_l114_11425


namespace NUMINAMATH_CALUDE_product_remainder_eleven_l114_11481

theorem product_remainder_eleven : (1010 * 1011 * 1012 * 1013 * 1014) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_eleven_l114_11481


namespace NUMINAMATH_CALUDE_vivi_fabric_purchase_l114_11420

/-- The total yards of fabric Vivi bought -/
def total_yards (checkered_cost plain_cost cost_per_yard : ℚ) : ℚ :=
  checkered_cost / cost_per_yard + plain_cost / cost_per_yard

/-- Proof that Vivi bought 16 yards of fabric -/
theorem vivi_fabric_purchase :
  total_yards 75 45 (7.5 : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_vivi_fabric_purchase_l114_11420


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l114_11417

open Matrix

theorem matrix_N_satisfies_conditions :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![1, -2, 0; 4, 6, 1; -3, 5, 2]
  let i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
  let j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
  let k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]
  N * i = !![1; 4; -3] ∧
  N * j = !![-2; 6; 5] ∧
  N * k = !![0; 1; 2] ∧
  det N ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l114_11417


namespace NUMINAMATH_CALUDE_fourth_member_income_l114_11462

/-- Given a family of 4 members with an average income of 10000,
    where 3 members earn 8000, 15000, and 6000 respectively,
    prove that the income of the fourth member is 11000. -/
theorem fourth_member_income
  (num_members : Nat)
  (avg_income : Nat)
  (income1 income2 income3 : Nat)
  (h1 : num_members = 4)
  (h2 : avg_income = 10000)
  (h3 : income1 = 8000)
  (h4 : income2 = 15000)
  (h5 : income3 = 6000) :
  num_members * avg_income - (income1 + income2 + income3) = 11000 :=
by sorry

end NUMINAMATH_CALUDE_fourth_member_income_l114_11462


namespace NUMINAMATH_CALUDE_quadratic_polynomial_divisibility_l114_11403

theorem quadratic_polynomial_divisibility (p : ℕ) (a b c : ℕ) (h_prime : Nat.Prime p)
  (h_a : 0 < a ∧ a ≤ p) (h_b : 0 < b ∧ b ≤ p) (h_c : 0 < c ∧ c ≤ p)
  (h_divisible : ∀ x : ℕ, x > 0 → p ∣ (a * x^2 + b * x + c)) :
  (p = 2 ∧ a + b + c = 4) ∨ (p > 2 ∧ a + b + c = 3 * p) := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_divisibility_l114_11403


namespace NUMINAMATH_CALUDE_annulus_area_l114_11435

theorem annulus_area (r₁ r₂ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) :
  π * r₂^2 - π * r₁^2 = 3 * π := by sorry

end NUMINAMATH_CALUDE_annulus_area_l114_11435


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l114_11497

theorem square_minus_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 6) 
  (product_eq : a * b = 3) : 
  a^2 - a*b + b^2 = 27 := by sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l114_11497


namespace NUMINAMATH_CALUDE_number_circle_exists_l114_11463

/-- A type representing a three-digit number with no zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  tens_nonzero : tens ≠ 0
  ones_nonzero : ones ≠ 0
  hundreds_lt_ten : hundreds < 10
  tens_lt_ten : tens < 10
  ones_lt_ten : ones < 10

/-- A type representing a circle of six three-digit numbers -/
structure NumberCircle where
  numbers : Fin 6 → ThreeDigitNumber
  all_different : ∀ i j, i ≠ j → numbers i ≠ numbers j
  circular_property : ∀ i, 
    (numbers i).tens = (numbers ((i + 1) % 6)).hundreds ∧
    (numbers i).ones = (numbers ((i + 1) % 6)).tens

/-- Function to check if a number is divisible by n -/
def isDivisibleBy (num : ThreeDigitNumber) (n : Nat) : Prop :=
  (100 * num.hundreds + 10 * num.tens + num.ones) % n = 0

/-- The main theorem -/
theorem number_circle_exists (n : Nat) : 
  (∃ circle : NumberCircle, ∀ i, isDivisibleBy (circle.numbers i) n) ↔ n = 3 ∨ n = 7 :=
sorry

end NUMINAMATH_CALUDE_number_circle_exists_l114_11463


namespace NUMINAMATH_CALUDE_children_boarding_bus_l114_11432

theorem children_boarding_bus (initial_children final_children : ℕ) 
  (h1 : initial_children = 18)
  (h2 : final_children = 25) :
  final_children - initial_children = 7 := by
  sorry

end NUMINAMATH_CALUDE_children_boarding_bus_l114_11432


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l114_11458

theorem solve_exponential_equation :
  ∃ x : ℝ, (1000 : ℝ)^2 = 10^x ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l114_11458


namespace NUMINAMATH_CALUDE_total_cost_calculation_l114_11455

theorem total_cost_calculation : 
  let sandwich_price : ℚ := 349/100
  let soda_price : ℚ := 87/100
  let sandwich_quantity : ℕ := 2
  let soda_quantity : ℕ := 4
  let total_cost : ℚ := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  total_cost = 1046/100 := by
sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l114_11455


namespace NUMINAMATH_CALUDE_maria_alice_ages_sum_l114_11421

/-- Maria and Alice's ages problem -/
theorem maria_alice_ages_sum : 
  ∀ (maria alice : ℕ), 
    maria = alice + 8 →  -- Maria is eight years older than Alice
    maria + 10 = 3 * (alice - 6) →  -- Ten years from now, Maria will be three times as old as Alice was six years ago
    maria + alice = 44  -- The sum of their current ages is 44
    := by sorry

end NUMINAMATH_CALUDE_maria_alice_ages_sum_l114_11421


namespace NUMINAMATH_CALUDE_similar_triangles_AB_length_l114_11452

/-- Two similar triangles with given side lengths and angles -/
structure SimilarTriangles where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  DE : ℝ
  EF : ℝ
  DF : ℝ
  angleBAC : ℝ
  angleEDF : ℝ

/-- Theorem stating that for the given similar triangles, AB = 75/17 -/
theorem similar_triangles_AB_length (t : SimilarTriangles)
  (h1 : t.AB = 5)
  (h2 : t.BC = 17)
  (h3 : t.AC = 12)
  (h4 : t.DE = 9)
  (h5 : t.EF = 15)
  (h6 : t.DF = 12)
  (h7 : t.angleBAC = 120)
  (h8 : t.angleEDF = 120) :
  t.AB = 75 / 17 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_AB_length_l114_11452


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l114_11470

/-- Profit calculation for Mary and Harry's partnership --/
theorem partnership_profit_calculation
  (mary_investment harry_investment : ℚ)
  (effort_share investment_share : ℚ)
  (mary_extra : ℚ)
  (h1 : mary_investment = 700)
  (h2 : harry_investment = 300)
  (h3 : effort_share = 1/3)
  (h4 : investment_share = 2/3)
  (h5 : mary_extra = 800) :
  ∃ (P : ℚ),
    P = 3000 ∧
    (P/6 + (mary_investment / (mary_investment + harry_investment)) * (investment_share * P)) -
    (P/6 + (harry_investment / (mary_investment + harry_investment)) * (investment_share * P)) = mary_extra :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l114_11470


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l114_11469

theorem lucky_larry_coincidence :
  let a : ℤ := 2
  let b : ℤ := 3
  let c : ℤ := 4
  let d : ℤ := 5
  ∃ f : ℤ, (a + b - c + d - f = a + (b - (c + (d - f)))) ∧ f = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l114_11469


namespace NUMINAMATH_CALUDE_smallest_sum_l114_11490

/-- Given positive integers A, B, C, and D satisfying certain conditions,
    the smallest possible sum A + B + C + D is 43. -/
theorem smallest_sum (A B C D : ℕ+) : 
  (∃ r : ℚ, B.val - A.val = r ∧ C.val - B.val = r) →  -- arithmetic sequence condition
  (∃ q : ℚ, C.val / B.val = q ∧ D.val / C.val = q) →  -- geometric sequence condition
  C.val / B.val = 4 / 3 →                             -- given ratio
  A.val + B.val + C.val + D.val ≥ 43 :=               -- smallest possible sum
by sorry

end NUMINAMATH_CALUDE_smallest_sum_l114_11490


namespace NUMINAMATH_CALUDE_team_formation_count_l114_11456

def num_boys : ℕ := 10
def num_girls : ℕ := 12
def boys_to_select : ℕ := 5
def girls_to_select : ℕ := 3

theorem team_formation_count : 
  (Nat.choose num_boys boys_to_select) * (Nat.choose num_girls girls_to_select) = 55440 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_count_l114_11456


namespace NUMINAMATH_CALUDE_min_cost_stationery_l114_11443

/-- Represents the cost and quantity of stationery items --/
structure Stationery where
  costA : ℕ  -- Cost of item A
  costB : ℕ  -- Cost of item B
  totalItems : ℕ  -- Total number of items to purchase
  minCost : ℕ  -- Minimum total cost
  maxCost : ℕ  -- Maximum total cost

/-- Theorem stating the minimum cost for the stationery purchase --/
theorem min_cost_stationery (s : Stationery) 
  (h1 : 2 * s.costA + s.costB = 35)
  (h2 : s.costA + 3 * s.costB = 30)
  (h3 : s.totalItems = 120)
  (h4 : s.minCost = 955)
  (h5 : s.maxCost = 1000) :
  ∃ (x : ℕ), x ≥ 36 ∧ 
             10 * x + 600 = 960 ∧ 
             ∀ (y : ℕ), y ≥ 36 → 10 * y + 600 ≥ 960 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_stationery_l114_11443


namespace NUMINAMATH_CALUDE_soy_sauce_bottle_ounces_l114_11419

/-- Represents the number of ounces in one cup -/
def ounces_per_cup : ℕ := 8

/-- Represents the number of cups of soy sauce required for the first recipe -/
def recipe1_cups : ℕ := 2

/-- Represents the number of cups of soy sauce required for the second recipe -/
def recipe2_cups : ℕ := 1

/-- Represents the number of cups of soy sauce required for the third recipe -/
def recipe3_cups : ℕ := 3

/-- Represents the number of bottles Stephanie needs to buy -/
def bottles_needed : ℕ := 3

/-- Theorem stating that one bottle of soy sauce contains 16 ounces -/
theorem soy_sauce_bottle_ounces : 
  (recipe1_cups + recipe2_cups + recipe3_cups) * ounces_per_cup / bottles_needed = 16 := by
  sorry

end NUMINAMATH_CALUDE_soy_sauce_bottle_ounces_l114_11419


namespace NUMINAMATH_CALUDE_triangle_probability_theorem_l114_11465

/-- The number of points in the plane -/
def num_points : ℕ := 10

/-- The total number of possible segments -/
def total_segments : ℕ := (num_points * (num_points - 1)) / 2

/-- The number of segments chosen -/
def chosen_segments : ℕ := 4

/-- The probability of choosing 4 segments that form a triangle -/
def triangle_probability : ℚ := 1680 / 49665

theorem triangle_probability_theorem :
  triangle_probability = (num_points.choose 3 * (total_segments - 3)) / total_segments.choose chosen_segments :=
by sorry

end NUMINAMATH_CALUDE_triangle_probability_theorem_l114_11465


namespace NUMINAMATH_CALUDE_product_digit_sum_l114_11445

def digit_repeat (d₁ d₂ d₃ : ℕ) (n : ℕ) : ℕ :=
  (d₁ * 10^2 + d₂ * 10 + d₃) * (10^(3*n) - 1) / 999

def a : ℕ := digit_repeat 3 0 3 33
def b : ℕ := digit_repeat 5 0 5 33

theorem product_digit_sum :
  (a * b % 10) + ((a * b / 1000) % 10) = 8 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l114_11445


namespace NUMINAMATH_CALUDE_min_value_theorem_l114_11416

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x + 2*y) = Real.log x + Real.log y) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log (a + 2*b) = Real.log a + Real.log b → 2*a + b ≥ 2*x + y) ∧ 
  (2*x + y = 9) ∧ (x = 3) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l114_11416


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_four_ninths_l114_11473

theorem smallest_fraction_greater_than_four_ninths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (4 : ℚ) / 9 < (a : ℚ) / b →
    (41 : ℚ) / 92 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_four_ninths_l114_11473


namespace NUMINAMATH_CALUDE_train_passing_time_l114_11442

/-- The time for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 110 →
  train_speed = 65 * (5/18) →
  man_speed = 7 * (5/18) →
  (train_length / (train_speed + man_speed)) = 5.5 := by sorry

end NUMINAMATH_CALUDE_train_passing_time_l114_11442


namespace NUMINAMATH_CALUDE_sum_of_two_with_prime_bound_l114_11433

theorem sum_of_two_with_prime_bound (n : ℕ) (h : n ≥ 50) :
  ∃ x y : ℕ, n = x + y ∧
    ∀ p : ℕ, p.Prime → (p ∣ x ∨ p ∣ y) → (n : ℝ).sqrt ≥ p :=
  sorry

end NUMINAMATH_CALUDE_sum_of_two_with_prime_bound_l114_11433


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l114_11464

theorem soccer_league_female_fraction :
  -- Last year's male participants
  ∀ (last_year_males : ℕ),
  last_year_males = 30 →
  -- Total participation increase
  ∀ (total_increase_rate : ℚ),
  total_increase_rate = 108/100 →
  -- Male participation increase
  ∀ (male_increase_rate : ℚ),
  male_increase_rate = 110/100 →
  -- Female participation increase
  ∀ (female_increase_rate : ℚ),
  female_increase_rate = 115/100 →
  -- The fraction of female participants this year
  ∃ (female_fraction : ℚ),
  female_fraction = 10/43 ∧
  (∃ (last_year_females : ℕ),
    -- Total participants this year
    total_increase_rate * (last_year_males + last_year_females : ℚ) =
    -- Males this year + Females this year
    male_increase_rate * last_year_males + female_increase_rate * last_year_females ∧
    -- Female fraction calculation
    female_fraction = (female_increase_rate * last_year_females) /
      (male_increase_rate * last_year_males + female_increase_rate * last_year_females)) :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l114_11464


namespace NUMINAMATH_CALUDE_surface_area_is_14_l114_11493

/-- The surface area of a rectangular prism formed by joining three 1x1x1 cubes side by side -/
def surface_area_of_prism : ℕ :=
  let length : ℕ := 3
  let width : ℕ := 1
  let height : ℕ := 1
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of the prism is 14 -/
theorem surface_area_is_14 : surface_area_of_prism = 14 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_is_14_l114_11493


namespace NUMINAMATH_CALUDE_expand_product_l114_11468

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l114_11468


namespace NUMINAMATH_CALUDE_nikolai_wins_l114_11454

/-- Represents a mountain goat with its jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- Calculates the number of jumps needed to cover a given distance -/
def jumps_needed (goat : Goat) (distance : ℕ) : ℕ :=
  (distance + goat.jump_distance - 1) / goat.jump_distance

/-- Represents the race between two goats -/
structure Race where
  goat1 : Goat
  goat2 : Goat
  distance : ℕ

/-- Determines if the first goat is faster than the second goat -/
def is_faster (race : Race) : Prop :=
  jumps_needed race.goat1 race.distance < jumps_needed race.goat2 race.distance

theorem nikolai_wins (gennady nikolai : Goat) (h1 : gennady.jump_distance = 6)
    (h2 : nikolai.jump_distance = 4) : is_faster { goat1 := nikolai, goat2 := gennady, distance := 2000 } := by
  sorry

#check nikolai_wins

end NUMINAMATH_CALUDE_nikolai_wins_l114_11454


namespace NUMINAMATH_CALUDE_original_price_correct_l114_11430

/-- The original price of an article before discounts -/
def original_price : ℝ := 81.30

/-- The final sale price after all discounts -/
def final_price : ℝ := 36

/-- The list of discount rates -/
def discount_rates : List ℝ := [0.15, 0.25, 0.20, 0.18]

/-- Calculate the price after applying all discounts -/
def price_after_discounts (price : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) price

theorem original_price_correct : 
  ∃ ε > 0, abs (price_after_discounts original_price discount_rates - final_price) < ε :=
by
  sorry

#eval price_after_discounts original_price discount_rates

end NUMINAMATH_CALUDE_original_price_correct_l114_11430


namespace NUMINAMATH_CALUDE_prime_divides_binomial_coefficient_l114_11472

theorem prime_divides_binomial_coefficient (p k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_binomial_coefficient_l114_11472


namespace NUMINAMATH_CALUDE_fifth_month_sale_l114_11431

/-- Given sales data for 6 months, prove the sale amount for the fifth month --/
theorem fifth_month_sale 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (average : ℚ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h6 : sale6 = 7391)
  (h_avg : average = 6900)
  (h_avg_def : average = (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6) :
  sale5 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l114_11431


namespace NUMINAMATH_CALUDE_work_completion_time_l114_11413

theorem work_completion_time (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (5 : ℝ) / 12 + 4 / b + 3 / c = 1 →
  1 / ((1 / b) + (1 / c)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l114_11413


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l114_11408

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ a ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l114_11408


namespace NUMINAMATH_CALUDE_chord_equation_l114_11482

def Circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}

def P : ℝ × ℝ := (1, 1)

def is_midpoint (m : ℝ × ℝ) (p : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  p.1 = (m.1 + n.1) / 2 ∧ p.2 = (m.2 + n.2) / 2

theorem chord_equation (M N : ℝ × ℝ) (h1 : M ∈ Circle) (h2 : N ∈ Circle)
  (h3 : is_midpoint M P N) :
  ∃ (a b c : ℝ), a * P.1 + b * P.2 + c = 0 ∧
                 ∀ (x y : ℝ), (x, y) ∈ Circle → 
                 ((x, y) = M ∨ (x, y) = N) → 
                 a * x + b * y + c = 0 ∧
                 (a, b, c) = (2, -1, -1) := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l114_11482


namespace NUMINAMATH_CALUDE_expression_evaluation_l114_11424

theorem expression_evaluation : 7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + 12 = 5765542 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l114_11424


namespace NUMINAMATH_CALUDE_final_number_independent_of_operations_l114_11429

/-- Represents the state of the blackboard with counts of 0, 1, and 2 --/
structure BoardState where
  count0 : Nat
  count1 : Nat
  count2 : Nat

/-- Represents a single operation of replacing two numbers with the third --/
inductive Operation
  | replace01with2
  | replace02with1
  | replace12with0

/-- Applies an operation to a board state --/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.replace01with2 => { count0 := state.count0 - 1, count1 := state.count1 - 1, count2 := state.count2 + 1 }
  | Operation.replace02with1 => { count0 := state.count0 - 1, count1 := state.count1 + 1, count2 := state.count2 - 1 }
  | Operation.replace12with0 => { count0 := state.count0 + 1, count1 := state.count1 - 1, count2 := state.count2 - 1 }

/-- Checks if the board state has only one number remaining --/
def isFinalState (state : BoardState) : Bool :=
  (state.count0 > 0 && state.count1 = 0 && state.count2 = 0) ||
  (state.count0 = 0 && state.count1 > 0 && state.count2 = 0) ||
  (state.count0 = 0 && state.count1 = 0 && state.count2 > 0)

/-- Gets the final number on the board --/
def getFinalNumber (state : BoardState) : Nat :=
  if state.count0 > 0 then 0
  else if state.count1 > 0 then 1
  else 2

/-- Theorem: The final number is determined by initial counts and their parities --/
theorem final_number_independent_of_operations (initialState : BoardState) 
  (ops1 ops2 : List Operation) 
  (h1 : isFinalState (ops1.foldl applyOperation initialState))
  (h2 : isFinalState (ops2.foldl applyOperation initialState)) :
  getFinalNumber (ops1.foldl applyOperation initialState) = 
  getFinalNumber (ops2.foldl applyOperation initialState) := by
  sorry

#check final_number_independent_of_operations

end NUMINAMATH_CALUDE_final_number_independent_of_operations_l114_11429


namespace NUMINAMATH_CALUDE_harper_gift_cost_l114_11438

/-- Harper's gift-buying problem -/
theorem harper_gift_cost (son_teachers daughter_teachers total_spent : ℕ) 
  (h1 : son_teachers = 3)
  (h2 : daughter_teachers = 4)
  (h3 : total_spent = 70) :
  total_spent / (son_teachers + daughter_teachers) = 10 := by
  sorry

#check harper_gift_cost

end NUMINAMATH_CALUDE_harper_gift_cost_l114_11438


namespace NUMINAMATH_CALUDE_complex_equation_solution_l114_11486

/-- Given that (1-√3i)z = √3+i, prove that z = i -/
theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I * Real.sqrt 3) * z = Real.sqrt 3 + Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l114_11486


namespace NUMINAMATH_CALUDE_range_of_a_eq_l114_11494

/-- Proposition p: The solution set of the inequality x^2 + (a-1)x + a^2 < 0 is empty. -/
def prop_p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + a^2 ≥ 0

/-- Quadratic function f(x) = x^2 - mx + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- Proposition q: f(3/2 + x) = f(3/2 - x), and max(f(x)) = 2 for x ∈ [0, a] -/
def prop_q (a : ℝ) : Prop :=
  ∃ m, (∀ x, f m ((3:ℝ)/2 + x) = f m ((3:ℝ)/2 - x)) ∧
       (∀ x, x ∈ Set.Icc 0 a → f m x ≤ 2) ∧
       (∃ x, x ∈ Set.Icc 0 a ∧ f m x = 2)

/-- The range of a given the conditions -/
def range_of_a : Set ℝ :=
  {a | (¬(prop_p a ∧ prop_q a)) ∧ (prop_p a ∨ prop_q a)}

theorem range_of_a_eq :
  range_of_a = Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_eq_l114_11494


namespace NUMINAMATH_CALUDE_off_road_vehicle_cost_l114_11492

theorem off_road_vehicle_cost 
  (dirt_bike_cost : ℕ) 
  (dirt_bike_count : ℕ) 
  (off_road_count : ℕ) 
  (registration_fee : ℕ) 
  (total_cost : ℕ) 
  (h1 : dirt_bike_cost = 150)
  (h2 : dirt_bike_count = 3)
  (h3 : off_road_count = 4)
  (h4 : registration_fee = 25)
  (h5 : total_cost = 1825)
  (h6 : total_cost = dirt_bike_cost * dirt_bike_count + 
                     off_road_count * x + 
                     registration_fee * (dirt_bike_count + off_road_count)) :
  x = 300 := by
  sorry


end NUMINAMATH_CALUDE_off_road_vehicle_cost_l114_11492


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l114_11444

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, y) in that quadrant such that y = mx + b -/
def passes_through_quadrant (m b : ℝ) (quad : Nat) : Prop :=
  match quad with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ y = m * x + b
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ y = m * x + b
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ y = m * x + b
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ y = m * x + b
  | _ => False

/-- The graph of y = -5x + 5 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant (-5) 5 1 ∧
  passes_through_quadrant (-5) 5 2 ∧
  passes_through_quadrant (-5) 5 4 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l114_11444


namespace NUMINAMATH_CALUDE_log_2_bounds_l114_11460

theorem log_2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
                     (h3 : 2^11 = 2048) (h4 : 2^14 = 16384) :
  3/11 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 2/7 := by
  sorry

end NUMINAMATH_CALUDE_log_2_bounds_l114_11460


namespace NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l114_11449

/-- A cubic sequence is a sequence of integers given by a_n = n^3 + bn^2 + cn + d,
    where b, c, and d are integer constants and n ranges over all integers. -/
def CubicSequence (b c d : ℤ) : ℤ → ℤ := fun n ↦ n^3 + b*n^2 + c*n + d

/-- A number is a perfect square if there exists an integer whose square equals the number. -/
def IsPerfectSquare (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

theorem cubic_sequence_with_two_squares_exists : ∃ b c d : ℤ,
  let a := CubicSequence b c d
  IsPerfectSquare (a 2015) ∧
  IsPerfectSquare (a 2016) ∧
  (∀ n : ℤ, n ≠ 2015 ∧ n ≠ 2016 → ¬ IsPerfectSquare (a n)) ∧
  a 2015 * a 2016 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l114_11449


namespace NUMINAMATH_CALUDE_sequence_problem_l114_11437

theorem sequence_problem (a b : ℝ) 
  (h1 : 0 < 2 ∧ 0 < a ∧ 0 < b ∧ 0 < 9)
  (h2 : a - 2 = b - a)  -- arithmetic sequence condition
  (h3 : a / 2 = b / a ∧ b / a = 9 / b)  -- geometric sequence condition
  : a = 4 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l114_11437


namespace NUMINAMATH_CALUDE_quadratic_function_range_l114_11450

/-- A quadratic function with specific properties -/
def f (a b x : ℝ) : ℝ := -x^2 + a*x + b^2 - b + 1

/-- The theorem statement -/
theorem quadratic_function_range (a b : ℝ) :
  (∀ x, f a b (1 - x) = f a b (1 + x)) →
  (∀ x ∈ Set.Icc (-1) 1, f a b x > 0) →
  b < -1 ∨ b > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l114_11450


namespace NUMINAMATH_CALUDE_mary_fruit_purchase_cost_l114_11487

/-- Represents the cost of each fruit type -/
structure FruitCosts where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

/-- Represents the quantity of each fruit type bought -/
structure FruitQuantities where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

/-- Calculates the total cost before discounts -/
def totalCostBeforeDiscounts (costs : FruitCosts) (quantities : FruitQuantities) : ℕ :=
  costs.apple * quantities.apple +
  costs.orange * quantities.orange +
  costs.banana * quantities.banana +
  costs.peach * quantities.peach +
  costs.grape * quantities.grape

/-- Calculates the discount for every 5 fruits bought -/
def fiveForOneDiscount (totalFruits : ℕ) : ℕ :=
  totalFruits / 5

/-- Calculates the discount for peaches and grapes bought together -/
def peachGrapeDiscount (peaches : ℕ) (grapes : ℕ) : ℕ :=
  (min (peaches / 3) (grapes / 2)) * 3

/-- Calculates the final cost after applying discounts -/
def finalCost (costs : FruitCosts) (quantities : FruitQuantities) : ℕ :=
  let totalFruits := quantities.apple + quantities.orange + quantities.banana + quantities.peach + quantities.grape
  let costBeforeDiscounts := totalCostBeforeDiscounts costs quantities
  let fiveForOneDiscountAmount := fiveForOneDiscount totalFruits
  let peachGrapeDiscountAmount := peachGrapeDiscount quantities.peach quantities.grape
  costBeforeDiscounts - fiveForOneDiscountAmount - peachGrapeDiscountAmount

/-- Theorem: Mary will pay $51 for her fruit purchase -/
theorem mary_fruit_purchase_cost :
  let costs : FruitCosts := { apple := 1, orange := 2, banana := 3, peach := 4, grape := 5 }
  let quantities : FruitQuantities := { apple := 5, orange := 3, banana := 2, peach := 6, grape := 4 }
  finalCost costs quantities = 51 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_purchase_cost_l114_11487


namespace NUMINAMATH_CALUDE_bug_meeting_point_l114_11488

/-- Represents a triangle with given side lengths -/
structure Triangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ

/-- Represents a bug moving along the perimeter of a triangle -/
structure Bug where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the meeting point of two bugs on a triangle's perimeter -/
def meetingPoint (t : Triangle) (b1 b2 : Bug) : ℝ :=
  sorry

theorem bug_meeting_point (t : Triangle) (b1 b2 : Bug) :
  t.pq = 8 ∧ t.qr = 10 ∧ t.pr = 12 ∧
  b1.speed = 2 ∧ b2.speed = 3 ∧
  b1.direction ≠ b2.direction →
  meetingPoint t b1 b2 = 3 :=
sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l114_11488


namespace NUMINAMATH_CALUDE_ball_distance_theorem_l114_11440

def initial_height : ℚ := 120
def rebound_fraction : ℚ := 1/3
def num_bounces : ℕ := 4

def descent_distance (n : ℕ) : ℚ :=
  initial_height * (rebound_fraction ^ n)

def total_distance : ℚ :=
  2 * (initial_height * (1 - rebound_fraction^(num_bounces + 1)) / (1 - rebound_fraction)) - initial_height

theorem ball_distance_theorem :
  total_distance = 5000 / 27 := by sorry

end NUMINAMATH_CALUDE_ball_distance_theorem_l114_11440


namespace NUMINAMATH_CALUDE_noodles_given_correct_daniel_noodles_l114_11400

/-- The number of noodles Daniel gave to William -/
def noodles_given (initial current : ℕ) : ℕ := initial - current

theorem noodles_given_correct (initial current : ℕ) (h : current ≤ initial) :
  noodles_given initial current = initial - current :=
by
  sorry

/-- The specific problem instance -/
theorem daniel_noodles :
  noodles_given 66 54 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_noodles_given_correct_daniel_noodles_l114_11400


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l114_11436

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime :
  (first_seven_primes.sum) % eighth_prime = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l114_11436
