import Mathlib

namespace stratified_sampling_example_l545_54597

/-- Given a total number of positions, male doctors, and female doctors,
    calculate the number of male doctors to be selected through stratified sampling. -/
def stratified_sampling (total_positions : ℕ) (male_doctors : ℕ) (female_doctors : ℕ) : ℕ :=
  (total_positions * male_doctors) / (male_doctors + female_doctors)

theorem stratified_sampling_example :
  stratified_sampling 15 120 180 = 6 := by
  sorry

end stratified_sampling_example_l545_54597


namespace felicity_gas_usage_l545_54551

/-- Proves that Felicity used 23 gallons of gas given the problem conditions -/
theorem felicity_gas_usage (adhira : ℝ) : 
  (adhira + (4 * adhira - 5) = 30) → (4 * adhira - 5 = 23) :=
by
  sorry

end felicity_gas_usage_l545_54551


namespace dog_age_ratio_l545_54549

/-- Given information about five dogs' ages, prove the ratio of the 4th to 3rd fastest dog's age --/
theorem dog_age_ratio :
  ∀ (age1 age2 age3 age4 age5 : ℕ),
  -- Average age of 1st and 5th fastest dogs is 18 years
  (age1 + age5) / 2 = 18 →
  -- 1st fastest dog is 10 years old
  age1 = 10 →
  -- 2nd fastest dog is 2 years younger than the 1st fastest dog
  age2 = age1 - 2 →
  -- 3rd fastest dog is 4 years older than the 2nd fastest dog
  age3 = age2 + 4 →
  -- 4th fastest dog is half the age of the 3rd fastest dog
  2 * age4 = age3 →
  -- 5th fastest dog is 20 years older than the 4th fastest dog
  age5 = age4 + 20 →
  -- Ratio of 4th fastest dog's age to 3rd fastest dog's age is 1:2
  2 * age4 = age3 := by
  sorry

end dog_age_ratio_l545_54549


namespace diamond_op_four_three_l545_54553

def diamond_op (m n : ℕ) : ℕ := n ^ 2 - m

theorem diamond_op_four_three : diamond_op 4 3 = 5 := by sorry

end diamond_op_four_three_l545_54553


namespace jerichos_money_l545_54515

theorem jerichos_money (x : ℕ) : 
  x - (14 + 7) = 9 → 2 * x = 60 := by
  sorry

end jerichos_money_l545_54515


namespace simplification_to_x_plus_one_l545_54572

theorem simplification_to_x_plus_one (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end simplification_to_x_plus_one_l545_54572


namespace water_bottle_cost_l545_54501

/-- Proves that the cost of a water bottle is $2 given the conditions of Adam's shopping trip. -/
theorem water_bottle_cost (num_sandwiches : ℕ) (sandwich_price total_cost : ℚ) : 
  num_sandwiches = 3 →
  sandwich_price = 3 →
  total_cost = 11 →
  total_cost - (num_sandwiches : ℚ) * sandwich_price = 2 :=
by sorry

end water_bottle_cost_l545_54501


namespace inequality_range_l545_54535

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end inequality_range_l545_54535


namespace jie_is_tallest_l545_54505

-- Define a type for the people
inductive Person : Type
  | Igor : Person
  | Jie : Person
  | Faye : Person
  | Goa : Person
  | Han : Person

-- Define a relation for "taller than"
def taller_than : Person → Person → Prop := sorry

-- Define the conditions
axiom igor_shorter_jie : taller_than Person.Jie Person.Igor
axiom faye_taller_goa : taller_than Person.Faye Person.Goa
axiom jie_taller_faye : taller_than Person.Jie Person.Faye
axiom han_shorter_goa : taller_than Person.Goa Person.Han

-- Define what it means to be the tallest
def is_tallest (p : Person) : Prop :=
  ∀ q : Person, p ≠ q → taller_than p q

-- State the theorem
theorem jie_is_tallest : is_tallest Person.Jie := by
  sorry

end jie_is_tallest_l545_54505


namespace job_completion_time_l545_54538

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  4 * (1/x + 1/30) = 0.4 → 
  x = 15 := by
sorry

end job_completion_time_l545_54538


namespace consecutive_numbers_square_sum_l545_54523

theorem consecutive_numbers_square_sum (a b c : ℕ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b + c = 27) → a^2 + b^2 + c^2 = 245 := by
  sorry

end consecutive_numbers_square_sum_l545_54523


namespace min_value_theorem_l545_54510

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a / (4 * b) + 1 / a ≥ 2 ∧
  (a / (4 * b) + 1 / a = 2 ↔ a = 2/3 ∧ b = 1/3) :=
sorry

end min_value_theorem_l545_54510


namespace parallel_resistances_solutions_l545_54519

theorem parallel_resistances_solutions : 
  ∀ x y z : ℕ+, 
    (1 : ℚ) / z = 1 / x + 1 / y → 
    ((x = 3 ∧ y = 6 ∧ z = 2) ∨ 
     (x = 4 ∧ y = 4 ∧ z = 2) ∨ 
     (x = 4 ∧ y = 12 ∧ z = 3) ∨ 
     (x = 6 ∧ y = 6 ∧ z = 3)) :=
by sorry

end parallel_resistances_solutions_l545_54519


namespace DL_length_l545_54581

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

-- Define the circles ω3 and ω4
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point L
def L : ℝ × ℝ := sorry

-- Define the given triangle
def givenTriangle : Triangle :=
  { DE := 6
  , EF := 10
  , FD := 8 }

-- Define circle ω3
def ω3 : Circle := sorry

-- Define circle ω4
def ω4 : Circle := sorry

-- State the theorem
theorem DL_length (t : Triangle) (ω3 ω4 : Circle) :
  t = givenTriangle →
  (ω3.center.1 - L.1)^2 + (ω3.center.2 - L.2)^2 = ω3.radius^2 →
  (ω4.center.1 - L.1)^2 + (ω4.center.2 - L.2)^2 = ω4.radius^2 →
  (0 - L.1)^2 + (0 - L.2)^2 = 4^2 := by
  sorry

end DL_length_l545_54581


namespace range_of_a_l545_54599

open Set

/-- The range of a for which ¬p is a necessary but not sufficient condition for ¬q -/
theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → 
    (x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0)) →
  (∃ x : ℝ, (x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧ 
    ¬(x^2 - 4*a*x + 3*a^2 < 0)) →
  (a ≤ -4 ∨ -2/3 ≤ a) :=
by sorry


end range_of_a_l545_54599


namespace circle_equation_correct_circle_properties_l545_54562

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def lies_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The specific circle we're considering -/
def our_circle : Circle :=
  { center := (0, 3)
    radius := 1 }

theorem circle_equation_correct :
  ∀ x y : ℝ, x^2 + (y - 3)^2 = 1 ↔ lies_on_circle our_circle (x, y) :=
sorry

theorem circle_properties :
  our_circle.center.1 = 0 ∧
  our_circle.radius = 1 ∧
  lies_on_circle our_circle (1, 3) :=
sorry

end circle_equation_correct_circle_properties_l545_54562


namespace complex_number_equality_l545_54576

theorem complex_number_equality : Complex.abs ((1 - Complex.I) / (1 + Complex.I)) + 2 * Complex.I = 1 + 2 * Complex.I := by
  sorry

end complex_number_equality_l545_54576


namespace ratio_of_x_to_y_l545_54532

theorem ratio_of_x_to_y (x y : ℝ) (h1 : 5 * x = 6 * y) (h2 : x * y ≠ 0) :
  (1/3 * x) / (1/5 * y) = 2 := by
  sorry

end ratio_of_x_to_y_l545_54532


namespace trapezoid_area_theorem_l545_54543

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- Represents the result of the area calculation -/
structure AreaResult :=
  (r1 : ℚ)
  (n1 : ℕ)
  (r2 : ℚ)
  (n2 : ℕ)
  (r3 : ℚ)

/-- Function to calculate the area of the trapezoid -/
def calculateArea (t : Trapezoid) : AreaResult :=
  sorry

/-- Theorem stating the properties of the calculated area -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.side1 = 4)
  (h2 : t.side2 = 6)
  (h3 : t.side3 = 8)
  (h4 : t.side4 = 10) :
  let result := calculateArea t
  Int.floor (result.r1 + result.r2 + result.r3 + result.n1 + result.n2) = 274 ∧
  ¬∃ (p : ℕ), Prime p ∧ (p^2 ∣ result.n1 ∨ p^2 ∣ result.n2) :=
by sorry

end trapezoid_area_theorem_l545_54543


namespace partial_fraction_decomposition_l545_54563

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = 5/2 ∧ Q = 0 ∧ R = -5) ∧
    ∀ (x : ℚ), x ≠ 4 ∧ x ≠ 2 →
      5*x / ((x - 4) * (x - 2)^3) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^3 :=
by sorry

end partial_fraction_decomposition_l545_54563


namespace atomic_weight_Ba_value_l545_54507

/-- The atomic weight of Fluorine (F) -/
def atomic_weight_F : ℝ := 19

/-- The molecular weight of the compound BaF₂ -/
def molecular_weight_BaF2 : ℝ := 175

/-- The number of F atoms in the compound -/
def num_F_atoms : ℕ := 2

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := molecular_weight_BaF2 - num_F_atoms * atomic_weight_F

theorem atomic_weight_Ba_value : atomic_weight_Ba = 137 := by sorry

end atomic_weight_Ba_value_l545_54507


namespace goods_train_speed_l545_54574

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 60) 
  (h2 : goods_train_length = 0.3) -- 300 m converted to km
  (h3 : passing_time = 1/300) -- 12 seconds converted to hours
  : ∃ (goods_train_speed : ℝ), goods_train_speed = 30 :=
by sorry

end goods_train_speed_l545_54574


namespace cubic_plus_linear_increasing_l545_54598

/-- The function f(x) = x^3 + x is strictly increasing on all real numbers. -/
theorem cubic_plus_linear_increasing : 
  ∀ x y : ℝ, x < y → (x^3 + x) < (y^3 + y) := by
sorry

end cubic_plus_linear_increasing_l545_54598


namespace expression_decrease_l545_54552

theorem expression_decrease (k x y : ℝ) (hk : k ≠ 0) :
  let x' := 0.75 * x
  let y' := 0.65 * y
  k * x' * y'^2 = (507/1600) * (k * x * y^2) := by
sorry

end expression_decrease_l545_54552


namespace triangle_angle_measure_l545_54567

theorem triangle_angle_measure (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_condition : Real.sin B ^ 2 - Real.sin C ^ 2 - Real.sin A ^ 2 = Real.sqrt 3 * Real.sin A * Real.sin C) : 
  B = 5 * Real.pi / 6 := by
sorry

end triangle_angle_measure_l545_54567


namespace a_range_l545_54500

theorem a_range (P : ∀ x > 0, x + 4 / x ≥ a) 
                (q : ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0) : 
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end a_range_l545_54500


namespace dot_product_of_vectors_l545_54524

/-- Given two vectors a and b in ℝ², prove that their dot product is -12
    when a + b = (1, 3) and a - b = (3, 7). -/
theorem dot_product_of_vectors (a b : ℝ × ℝ) 
    (h1 : a + b = (1, 3)) 
    (h2 : a - b = (3, 7)) : 
  a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end dot_product_of_vectors_l545_54524


namespace equal_segments_after_rearrangement_l545_54584

-- Define a line in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a right-angled triangle
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)

-- Define a function to check if a line is parallel to another line
def isParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if a line intersects triangles in equal segments
def intersectsInEqualSegments (l : Line) (t1 t2 t3 : RightTriangle) : Prop :=
  sorry -- Definition omitted for brevity

-- Main theorem
theorem equal_segments_after_rearrangement
  (l : Line)
  (t1 t2 t3 : RightTriangle)
  (h1 : ∃ (l' : Line), isParallel l l' ∧ intersectsInEqualSegments l' t1 t2 t3) :
  ∃ (l'' : Line), isParallel l l'' ∧ intersectsInEqualSegments l'' t1 t2 t3 :=
by sorry

end equal_segments_after_rearrangement_l545_54584


namespace bill_sunday_miles_bill_sunday_miles_proof_l545_54522

theorem bill_sunday_miles : ℕ → ℕ → ℕ → Prop :=
  fun bill_saturday bill_sunday julia_sunday =>
    (bill_sunday = bill_saturday + 4) →
    (julia_sunday = 2 * bill_sunday) →
    (bill_saturday + bill_sunday + julia_sunday = 28) →
    bill_sunday = 8

-- The proof would go here, but we'll skip it as requested
theorem bill_sunday_miles_proof : ∃ (bill_saturday bill_sunday julia_sunday : ℕ),
  bill_sunday_miles bill_saturday bill_sunday julia_sunday :=
sorry

end bill_sunday_miles_bill_sunday_miles_proof_l545_54522


namespace gingerbread_theorem_l545_54534

def gingerbread_problem (red_hats blue_boots both : ℕ) : Prop :=
  let total := red_hats + blue_boots - both
  (red_hats : ℚ) / total * 100 = 50

theorem gingerbread_theorem :
  gingerbread_problem 6 9 3 := by
  sorry

end gingerbread_theorem_l545_54534


namespace cone_volume_divided_by_pi_l545_54589

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20 units, divided by π, is equal to 1125√7. -/
theorem cone_volume_divided_by_pi : 
  ∀ (r h : ℝ) (V : ℝ),
  -- Conditions
  (2 * π * r = 30 * π) →  -- Arc length becomes circumference of cone's base
  (20^2 = r^2 + h^2) →    -- Pythagorean theorem relating slant height to radius and height
  (V = (1/3) * π * r^2 * h) →  -- Volume formula for a cone
  -- Conclusion
  (V / π = 1125 * Real.sqrt 7) :=
by
  sorry

end cone_volume_divided_by_pi_l545_54589


namespace abc_inequality_abc_inequality_tight_l545_54596

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ 1/8 :=
sorry

theorem abc_inequality_tight :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) = 1/8 :=
sorry

end abc_inequality_abc_inequality_tight_l545_54596


namespace parabola_roots_l545_54548

/-- Given a parabola y = ax^2 - 2ax + c where a ≠ 0 that passes through the point (3, 0),
    prove that the solutions to ax^2 - 2ax + c = 0 are x₁ = -1 and x₂ = 3. -/
theorem parabola_roots (a c : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 - 2*a*x + c = 0 ↔ x = -1 ∨ x = 3) ↔
  a * 3^2 - 2*a*3 + c = 0 :=
by sorry

end parabola_roots_l545_54548


namespace min_gennadys_required_l545_54573

/-- Represents the number of people with a given name -/
structure Attendees :=
  (alexanders : Nat)
  (borises : Nat)
  (vasilys : Nat)
  (gennadys : Nat)

/-- Checks if the arrangement is valid (no two people with the same name are adjacent) -/
def isValidArrangement (a : Attendees) : Prop :=
  a.borises - 1 ≤ a.alexanders + a.vasilys + a.gennadys

/-- The given festival attendance -/
def festivalAttendance : Attendees :=
  { alexanders := 45
  , borises := 122
  , vasilys := 27
  , gennadys := 49 }

/-- Theorem stating that 49 is the minimum number of Gennadys required -/
theorem min_gennadys_required :
  isValidArrangement festivalAttendance ∧
  ∀ g : Nat, g < festivalAttendance.gennadys →
    ¬isValidArrangement { alexanders := festivalAttendance.alexanders
                        , borises := festivalAttendance.borises
                        , vasilys := festivalAttendance.vasilys
                        , gennadys := g } :=
by
  sorry

end min_gennadys_required_l545_54573


namespace exists_n_with_digit_sum_property_l545_54569

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Predicate to check if a number is composed of only 0 and 1 digits -/
def isComposedOf01 (n : ℕ) : Prop :=
  sorry

/-- Main theorem -/
theorem exists_n_with_digit_sum_property (m : ℕ) :
  ∃ n : ℕ, isComposedOf01 n ∧ sumOfDigits n = m ∧ sumOfDigits (n^2) = m^2 :=
sorry

end exists_n_with_digit_sum_property_l545_54569


namespace circle_radius_theorem_l545_54593

/-- The radius of a circle concentric with and outside a regular octagon -/
def circle_radius (octagon_side_length : ℝ) (probability_four_sides : ℝ) : ℝ :=
  sorry

/-- The theorem stating the relationship between the circle radius, octagon side length, and probability of seeing four sides -/
theorem circle_radius_theorem (octagon_side_length : ℝ) (probability_four_sides : ℝ) :
  circle_radius octagon_side_length probability_four_sides = 6 * Real.sqrt 2 - Real.sqrt 3 :=
by
  sorry

end circle_radius_theorem_l545_54593


namespace ratio_is_pure_imaginary_l545_54559

theorem ratio_is_pure_imaginary (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  ∃ (y : ℝ), z₁ / z₂ = Complex.I * y := by
  sorry

end ratio_is_pure_imaginary_l545_54559


namespace planting_area_is_2x_l545_54509

/-- Represents the area of the planting region in a rectangular garden with an internal path. -/
def planting_area (x : ℝ) : ℝ :=
  let garden_length : ℝ := x + 2
  let garden_width : ℝ := 4
  let path_width : ℝ := 1
  let planting_length : ℝ := garden_length - 2 * path_width
  let planting_width : ℝ := garden_width - 2 * path_width
  planting_length * planting_width

/-- Theorem stating that the planting area is equal to 2x square meters. -/
theorem planting_area_is_2x (x : ℝ) : planting_area x = 2 * x := by
  sorry


end planting_area_is_2x_l545_54509


namespace smallest_m_is_ten_l545_54530

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℤ
  first_term : a 1 = -19
  difference : a 7 - a 4 = 6
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℤ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The theorem to be proved -/
theorem smallest_m_is_ten (seq : ArithmeticSequence) :
  ∃ m : ℕ+, (∀ n : ℕ+, sum_n seq n ≥ sum_n seq m) ∧ 
    (∀ k : ℕ+, k < m → ∃ n : ℕ+, sum_n seq n < sum_n seq k) ∧
    m = 10 := by
  sorry

end smallest_m_is_ten_l545_54530


namespace max_tan_A_in_triangle_l545_54533

open Real

theorem max_tan_A_in_triangle (a b c A B C : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  -- Given conditions
  a = 2 →
  b * cos C - c * cos B = 4 →
  π/4 ≤ C ∧ C ≤ π/3 →
  -- Conclusion
  (∃ (max_tan_A : ℝ), max_tan_A = 1/2 ∧ ∀ (tan_A : ℝ), tan_A = tan A → tan_A ≤ max_tan_A) :=
by sorry

end max_tan_A_in_triangle_l545_54533


namespace intersection_of_A_and_B_l545_54583

def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

theorem intersection_of_A_and_B : A ∩ B = {8, 10} := by
  sorry

end intersection_of_A_and_B_l545_54583


namespace degrees_to_radians_1920_l545_54587

theorem degrees_to_radians_1920 : 
  (1920 : ℝ) * (π / 180) = (32 * π) / 3 := by sorry

end degrees_to_radians_1920_l545_54587


namespace model_evaluation_criteria_l545_54506

-- Define the concept of a model
def Model : Type := ℝ → ℝ

-- Define the concept of residuals
def Residuals (m : Model) (data : Set (ℝ × ℝ)) : Set ℝ := sorry

-- Define the concept of residual plot distribution
def EvenlyDistributedInHorizontalBand (r : Set ℝ) : Prop := sorry

-- Define the sum of squared residuals
def SumSquaredResiduals (r : Set ℝ) : ℝ := sorry

-- Define the concept of model appropriateness
def ModelAppropriate (m : Model) (data : Set (ℝ × ℝ)) : Prop := 
  EvenlyDistributedInHorizontalBand (Residuals m data)

-- Define the concept of better fitting model
def BetterFittingModel (m1 m2 : Model) (data : Set (ℝ × ℝ)) : Prop :=
  SumSquaredResiduals (Residuals m1 data) < SumSquaredResiduals (Residuals m2 data)

-- Theorem statement
theorem model_evaluation_criteria 
  (m : Model) (data : Set (ℝ × ℝ)) (m1 m2 : Model) :
  (ModelAppropriate m data ↔ 
    EvenlyDistributedInHorizontalBand (Residuals m data)) ∧
  (BetterFittingModel m1 m2 data ↔ 
    SumSquaredResiduals (Residuals m1 data) < SumSquaredResiduals (Residuals m2 data)) :=
by sorry

end model_evaluation_criteria_l545_54506


namespace distance_between_points_l545_54529

/-- The distance between points (1, -3) and (-4, 7) is 5√5. -/
theorem distance_between_points : Real.sqrt ((1 - (-4))^2 + (-3 - 7)^2) = 5 * Real.sqrt 5 := by
  sorry

end distance_between_points_l545_54529


namespace quadratic_inequality_range_l545_54528

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end quadratic_inequality_range_l545_54528


namespace ned_trays_theorem_l545_54544

/-- The number of trays Ned can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Ned made -/
def num_trips : ℕ := 4

/-- The number of trays Ned picked up from the second table -/
def trays_from_second_table : ℕ := 5

/-- The number of trays Ned picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * num_trips - trays_from_second_table

theorem ned_trays_theorem : trays_from_first_table = 27 := by
  sorry

end ned_trays_theorem_l545_54544


namespace competition_probability_l545_54556

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The probability of incorrectly answering a single question -/
def p_incorrect : ℝ := 1 - p_correct

/-- The number of preset questions in the competition -/
def num_questions : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := p_correct * p_incorrect * p_correct * p_correct

theorem competition_probability :
  prob_four_questions = 0.128 :=
sorry

end competition_probability_l545_54556


namespace dot_product_max_value_l545_54540

theorem dot_product_max_value (x y z : ℝ) :
  let a : Fin 3 → ℝ := ![1, 1, -2]
  let b : Fin 3 → ℝ := ![x, y, z]
  x^2 + y^2 + z^2 = 16 →
  (∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 16 → 
    (a 0) * x' + (a 1) * y' + (a 2) * z' ≤ (a 0) * x + (a 1) * y + (a 2) * z) →
  (a 0) * x + (a 1) * y + (a 2) * z = 4 * Real.sqrt 6 :=
by sorry

end dot_product_max_value_l545_54540


namespace fraction_simplification_l545_54539

theorem fraction_simplification (d : ℝ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end fraction_simplification_l545_54539


namespace f_shift_three_l545_54565

/-- Given a function f(x) = x(x-1)/2, prove that f(x+3) = f(x) + 3x + 3 for all real x. -/
theorem f_shift_three (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (x - 1) / 2
  f (x + 3) = f x + 3 * x + 3 := by
  sorry

end f_shift_three_l545_54565


namespace no_double_application_function_l545_54571

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end no_double_application_function_l545_54571


namespace equation_solution_l545_54542

theorem equation_solution : ∃ x : ℝ, (Real.sqrt (72 / 25) = (x / 25) ^ (1/4)) ∧ x = 207.36 := by
  sorry

end equation_solution_l545_54542


namespace parallel_iff_slope_eq_l545_54558

/-- Two lines in the plane -/
structure Line where
  k : ℝ
  b : ℝ

/-- Define when two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.k = l2.k

/-- The main theorem: k1 = k2 iff l1 ∥ l2 -/
theorem parallel_iff_slope_eq (l1 l2 : Line) :
  l1.k = l2.k ↔ parallel l1 l2 :=
by sorry

end parallel_iff_slope_eq_l545_54558


namespace correct_lineup_count_l545_54568

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def all_stars : ℕ := 3
def required_players : ℕ := 2

def possible_lineups : ℕ := Nat.choose (team_size - all_stars) (lineup_size - required_players)

theorem correct_lineup_count : possible_lineups = 220 := by sorry

end correct_lineup_count_l545_54568


namespace ratio_of_sum_to_difference_l545_54594

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 8 * (x - y)) : x / y = 9 / 7 := by
  sorry

end ratio_of_sum_to_difference_l545_54594


namespace line_equation_from_intercept_and_slope_l545_54518

/-- A line with x-intercept a and slope m -/
structure Line where
  a : ℝ  -- x-intercept
  m : ℝ  -- slope

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with x-intercept 2 and slope 1, its equation is x - y - 2 = 0 -/
theorem line_equation_from_intercept_and_slope :
  ∀ (L : Line), L.a = 2 ∧ L.m = 1 →
  ∃ (eq : LineEquation), eq.a = 1 ∧ eq.b = -1 ∧ eq.c = -2 :=
by sorry

end line_equation_from_intercept_and_slope_l545_54518


namespace base_conversion_314_to_1242_l545_54517

/-- Converts a natural number from base 10 to base 6 --/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number in base 10 --/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_314_to_1242 :
  toBase6 314 = [1, 2, 4, 2] ∧ fromBase6 [1, 2, 4, 2] = 314 := by
  sorry

end base_conversion_314_to_1242_l545_54517


namespace N_subset_M_l545_54503

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x - 2 = 0}

theorem N_subset_M : N ⊆ M := by
  sorry

end N_subset_M_l545_54503


namespace fraction_integrality_l545_54557

theorem fraction_integrality (a b c : ℤ) 
  (h : ∃ (n : ℤ), (a * b / c + a * c / b + b * c / a) = n) : 
  (∃ (n1 : ℤ), a * b / c = n1) ∧ 
  (∃ (n2 : ℤ), a * c / b = n2) ∧ 
  (∃ (n3 : ℤ), b * c / a = n3) := by
sorry

end fraction_integrality_l545_54557


namespace solution_set_of_quadratic_inequality_l545_54504

theorem solution_set_of_quadratic_inequality :
  ∀ x : ℝ, 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2/3 := by
  sorry

end solution_set_of_quadratic_inequality_l545_54504


namespace shaded_area_percentage_l545_54536

/-- Two congruent squares with side length 20 overlap to form a 20 by 40 rectangle.
    The shaded area is the overlap of the two squares. -/
theorem shaded_area_percentage (square_side : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side = 20 →
  rectangle_width = 20 →
  rectangle_length = 40 →
  (square_side * square_side) / (rectangle_width * rectangle_length) = 1 / 2 := by
  sorry

end shaded_area_percentage_l545_54536


namespace smallest_n_square_and_cube_l545_54564

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 5 * n = a^2) ∧ 
  (∃ (b : ℕ), 3 * n = b^3) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 5 * m = x^2) → 
    (∃ (y : ℕ), 3 * m = y^3) → 
    m ≥ n) ∧
  n = 1125 := by
sorry

end smallest_n_square_and_cube_l545_54564


namespace a_squared_plus_reciprocal_squared_is_integer_l545_54531

theorem a_squared_plus_reciprocal_squared_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1 / a = k) :
  ∃ m : ℤ, a^2 + 1 / a^2 = m := by
sorry

end a_squared_plus_reciprocal_squared_is_integer_l545_54531


namespace work_completion_time_l545_54582

/-- The time taken for Ganesh, Ram, and Sohan to complete a work together, given their individual work rates. -/
theorem work_completion_time 
  (ganesh_ram_rate : ℚ) -- Combined work rate of Ganesh and Ram
  (sohan_rate : ℚ)       -- Work rate of Sohan
  (h1 : ganesh_ram_rate = 1 / 24) -- Ganesh and Ram can complete the work in 24 days
  (h2 : sohan_rate = 1 / 48)      -- Sohan can complete the work in 48 days
  : (1 : ℚ) / (ganesh_ram_rate + sohan_rate) = 16 := by
  sorry

end work_completion_time_l545_54582


namespace derivative_f_at_zero_l545_54579

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (2^(Real.tan x) - 2^(Real.sin x)) / x^2 else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = Real.log (Real.sqrt 2) := by sorry

end derivative_f_at_zero_l545_54579


namespace spherical_to_rectangular_conversion_l545_54537

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, 2 * Real.sqrt 3, 2) := by sorry

end spherical_to_rectangular_conversion_l545_54537


namespace unique_sums_count_l545_54546

def set_A : Finset ℕ := {2, 3, 5, 8}
def set_B : Finset ℕ := {1, 4, 6, 7}

theorem unique_sums_count : 
  Finset.card ((set_A.product set_B).image (fun p => p.1 + p.2)) = 11 := by
  sorry

end unique_sums_count_l545_54546


namespace multiples_properties_l545_54555

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 12 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ (∃ p : ℤ, a - b = 4 * p) :=
by sorry

end multiples_properties_l545_54555


namespace bankers_gain_specific_case_l545_54570

/-- Calculates the banker's gain given the banker's discount, interest rate, and time period. -/
def bankers_gain (bankers_discount : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  (bankers_discount * interest_rate * time) / (100 + (interest_rate * time))

/-- Theorem stating that given the specific conditions, the banker's gain is 90. -/
theorem bankers_gain_specific_case :
  bankers_gain 340 12 3 = 90 := by
  sorry

end bankers_gain_specific_case_l545_54570


namespace decimal_period_equals_number_period_l545_54511

/-- The length of the repeating period in the decimal representation of a fraction -/
def decimal_period_length (n p : ℕ) : ℕ := sorry

/-- The length of the period of a number in decimal representation -/
def number_period_length (p : ℕ) : ℕ := sorry

/-- Theorem stating that for a natural number n and a prime number p, 
    where n ≤ p - 1, the length of the repeating period in the decimal 
    representation of n/p is equal to the length of the period of p -/
theorem decimal_period_equals_number_period (n p : ℕ) 
  (h_prime : Nat.Prime p) (h_n_le_p_minus_one : n ≤ p - 1) : 
  decimal_period_length n p = number_period_length p := by
  sorry

end decimal_period_equals_number_period_l545_54511


namespace at_most_one_integer_point_on_circle_l545_54527

theorem at_most_one_integer_point_on_circle :
  ∀ (x y u v : ℤ),
  (x - Real.sqrt 2)^2 + (y - Real.sqrt 3)^2 = (u - Real.sqrt 2)^2 + (v - Real.sqrt 3)^2 →
  x = u ∧ y = v :=
by sorry

end at_most_one_integer_point_on_circle_l545_54527


namespace sqrt_equation_solution_l545_54508

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- The unique solution is z = -11
  use -11
  sorry

end sqrt_equation_solution_l545_54508


namespace eighteenth_replacement_is_march_l545_54591

def months : List String := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def replacement_interval : Nat := 5

def first_replacement_month : String := "February"

def nth_replacement (n : Nat) : Nat :=
  replacement_interval * (n - 1)

theorem eighteenth_replacement_is_march :
  let months_after_february := nth_replacement 18
  let month_index := months_after_february % 12
  let replacement_month := months[(months.indexOf first_replacement_month + month_index) % 12]
  replacement_month = "March" := by
  sorry

end eighteenth_replacement_is_march_l545_54591


namespace binary_1101011_equals_base5_412_l545_54513

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem binary_1101011_equals_base5_412 : 
  decimal_to_base5 (binary_to_decimal [true, true, false, true, false, true, true]) = [4, 1, 2] := by
  sorry

end binary_1101011_equals_base5_412_l545_54513


namespace parabola_intersection_l545_54566

theorem parabola_intersection (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b*x₁ + c = 0 ∧ a * x₂^2 + 2*b*x₂ + c = 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b * x₁^2 + 2*c*x₁ + a = 0 ∧ b * x₂^2 + 2*c*x₂ + a = 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ c * x₁^2 + 2*a*x₁ + b = 0 ∧ c * x₂^2 + 2*a*x₂ + b = 0) :=
sorry

end parabola_intersection_l545_54566


namespace cone_volume_from_cylinder_volume_l545_54516

/-- Given a cylinder with volume 72π cm³, prove that a cone with the same height and radius has a volume of 24π cm³ -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 72 * π → (1/3) * π * r^2 * h = 24 * π := by
  sorry

#check cone_volume_from_cylinder_volume

end cone_volume_from_cylinder_volume_l545_54516


namespace graduating_class_boys_count_l545_54526

theorem graduating_class_boys_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 466 → difference = 212 → boys + (boys + difference) = total → boys = 127 := by
  sorry

end graduating_class_boys_count_l545_54526


namespace multiplication_division_equality_l545_54547

theorem multiplication_division_equality : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end multiplication_division_equality_l545_54547


namespace district3_to_district1_ratio_l545_54586

/-- The number of voters in District 1 -/
def district1_voters : ℕ := 322

/-- The difference in voters between District 3 and District 2 -/
def district3_2_diff : ℕ := 19

/-- The total number of voters in all three districts -/
def total_voters : ℕ := 1591

/-- The ratio of voters in District 3 to District 1 -/
def voter_ratio : ℚ := 2

theorem district3_to_district1_ratio :
  ∃ (district2_voters district3_voters : ℕ),
    district2_voters = district3_voters - district3_2_diff ∧
    district1_voters + district2_voters + district3_voters = total_voters ∧
    district3_voters = (voter_ratio : ℚ) * district1_voters := by
  sorry

end district3_to_district1_ratio_l545_54586


namespace geometry_propositions_l545_54585

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Axioms for the properties of parallel and perpendicular
axiom parallel_transitive {a b c : Plane} : parallel a b → parallel a c → parallel b c
axiom perpendicular_from_line {l : Line} {a b : Plane} : 
  line_perpendicular_plane l a → line_parallel_plane l b → perpendicular a b

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem geometry_propositions :
  -- Proposition ①
  (∀ a b c : Plane, parallel a b → parallel a c → parallel b c) ∧
  -- Proposition ③
  (∀ l : Line, ∀ a b : Plane, line_perpendicular_plane l a → line_parallel_plane l b → perpendicular a b) ∧
  -- Negation of Proposition ②
  ¬(∀ l : Line, ∀ a b : Plane, perpendicular a b → line_parallel_plane l a → line_perpendicular_plane l b) ∧
  -- Negation of Proposition ④
  ¬(∀ l1 l2 : Line, ∀ a : Plane, line_parallel l1 l2 → line_in_plane l2 a → line_parallel_plane l1 a) :=
by sorry

end geometry_propositions_l545_54585


namespace wheat_cost_per_acre_l545_54512

theorem wheat_cost_per_acre 
  (total_land : ℕ)
  (wheat_land : ℕ)
  (corn_cost_per_acre : ℕ)
  (total_capital : ℕ)
  (h1 : total_land = 4500)
  (h2 : wheat_land = 3400)
  (h3 : corn_cost_per_acre = 42)
  (h4 : total_capital = 165200) :
  ∃ (wheat_cost_per_acre : ℕ),
    wheat_cost_per_acre * wheat_land + 
    corn_cost_per_acre * (total_land - wheat_land) = 
    total_capital ∧ 
    wheat_cost_per_acre = 35 := by
  sorry

end wheat_cost_per_acre_l545_54512


namespace min_value_expression_l545_54580

theorem min_value_expression (x y k : ℝ) : (x*y - k)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end min_value_expression_l545_54580


namespace decimal_93_to_binary_binary_to_decimal_93_l545_54592

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_93_to_binary :
  to_binary 93 = [true, false, true, true, true, false, true] :=
sorry

theorem binary_to_decimal_93 :
  from_binary [true, false, true, true, true, false, true] = 93 :=
sorry

end decimal_93_to_binary_binary_to_decimal_93_l545_54592


namespace sandwich_combinations_l545_54550

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) : 
  n_meat = 12 → n_cheese = 11 → (n_meat.choose 1) * (n_cheese.choose 3) = 1980 := by
  sorry

end sandwich_combinations_l545_54550


namespace choir_arrangement_l545_54514

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n % 14 = 0) ↔ n ≥ 6930 ∧ ∀ m : ℕ, m < n → (m % 9 ≠ 0 ∨ m % 10 ≠ 0 ∨ m % 11 ≠ 0 ∨ m % 14 ≠ 0) :=
by sorry

end choir_arrangement_l545_54514


namespace satellite_upgraded_fraction_l545_54560

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (upgraded_total : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.upgraded_total / (s.units * s.non_upgraded_per_unit + s.upgraded_total)

theorem satellite_upgraded_fraction
  (s : Satellite)
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.upgraded_total / 6) :
  upgraded_fraction s = 1/5 := by
  sorry

end satellite_upgraded_fraction_l545_54560


namespace smallest_term_at_six_l545_54577

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 3 * n^2 - 38 * n + 12

/-- The index of the smallest term in the sequence -/
def smallest_term_index : ℕ := 6

/-- Theorem stating that the smallest term in the sequence occurs at index 6 -/
theorem smallest_term_at_six :
  ∀ (n : ℕ), n ≠ smallest_term_index → a n > a smallest_term_index :=
sorry

end smallest_term_at_six_l545_54577


namespace arithmetic_sequence_special_case_l545_54595

/-- An arithmetic sequence with the given properties has the general term formula a_n = 2n -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence
  d ≠ 0 →  -- non-zero common difference
  a 1 = 2 →  -- a_1 = 2
  (a 2 * a 8 = (a 4)^2) →  -- (a_2, a_4, a_8) forms a geometric sequence
  (∀ n, a n = 2 * n) :=  -- general term formula
by sorry

end arithmetic_sequence_special_case_l545_54595


namespace probability_sum_five_l545_54521

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 4

theorem probability_sum_five (dice_outcomes : ℕ) (favorable_outcomes : ℕ) :
  dice_outcomes = 36 →
  favorable_outcomes = 4 →
  (favorable_outcomes : ℚ) / dice_outcomes = 1/9 := by
  sorry

end probability_sum_five_l545_54521


namespace consecutive_negative_integers_sum_l545_54502

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 1224 → n + (n + 1) = -69 := by
  sorry

end consecutive_negative_integers_sum_l545_54502


namespace remainder_problem_l545_54590

theorem remainder_problem (n : ℤ) : n % 5 = 3 → (4 * n - 5) % 5 = 2 := by
  sorry

end remainder_problem_l545_54590


namespace max_length_AB_l545_54554

/-- The function representing the length of AB -/
def f (t : ℝ) : ℝ := -2 * t^2 + 3 * t + 9

/-- The theorem stating the maximum value of f(t) for t in [0, 3] -/
theorem max_length_AB : 
  ∃ (t : ℝ), t ∈ Set.Icc 0 3 ∧ f t = 81/8 ∧ ∀ x ∈ Set.Icc 0 3, f x ≤ 81/8 :=
sorry

end max_length_AB_l545_54554


namespace texas_migration_l545_54520

/-- The number of people moving to Texas in four days -/
def people_moving : ℕ := 3600

/-- The number of days -/
def num_days : ℕ := 4

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def avg_people_per_hour : ℚ :=
  people_moving / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem texas_migration :
  round_to_nearest avg_people_per_hour = 38 := by
  sorry

end texas_migration_l545_54520


namespace horner_method_correctness_horner_poly_at_5_l545_54525

def horner_poly (x : ℝ) : ℝ := (((((3*x - 4)*x + 6)*x - 2)*x - 5)*x - 2)

def original_poly (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem horner_method_correctness :
  ∀ x : ℝ, horner_poly x = original_poly x :=
sorry

theorem horner_poly_at_5 : horner_poly 5 = 7548 :=
sorry

end horner_method_correctness_horner_poly_at_5_l545_54525


namespace marble_remainder_l545_54561

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 6 → (r + p) % 8 = 3 := by
  sorry

end marble_remainder_l545_54561


namespace no_integer_solution_l545_54541

theorem no_integer_solution : ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 - 2*x^2*y^2 - 2*y^2*z^2 - 2*z^2*x^2 = 2000 := by
  sorry

end no_integer_solution_l545_54541


namespace puppy_weight_l545_54575

theorem puppy_weight (puppy smaller_cat larger_cat : ℝ) 
  (total_weight : puppy + smaller_cat + larger_cat = 24)
  (puppy_larger_cat : puppy + larger_cat = 2 * smaller_cat)
  (puppy_smaller_cat : puppy + smaller_cat = larger_cat) :
  puppy = 4 := by
  sorry

end puppy_weight_l545_54575


namespace no_triangle_with_special_side_ratios_l545_54545

theorem no_triangle_with_special_side_ratios :
  ¬ ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a + b > c ∧ b + c > a ∧ a + c > b) ∧
    ((a = b / 2 ∧ a = c / 3) ∨ 
     (b = a / 2 ∧ b = c / 3) ∨ 
     (c = a / 2 ∧ c = b / 3)) :=
by sorry

end no_triangle_with_special_side_ratios_l545_54545


namespace beef_weight_before_processing_l545_54578

theorem beef_weight_before_processing (weight_after : ℝ) (percent_lost : ℝ) : 
  weight_after = 240 ∧ percent_lost = 40 → 
  weight_after / (1 - percent_lost / 100) = 400 := by
  sorry

end beef_weight_before_processing_l545_54578


namespace digit_puzzle_solution_l545_54588

theorem digit_puzzle_solution :
  ∃! (A B C D E F G H J : ℕ),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ J < 10) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
     E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
     F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
     G ≠ H ∧ G ≠ J ∧
     H ≠ J) ∧
    (100 * A + 10 * B + C + 100 * D + 10 * E + F + 10 * G + E = 100 * G + 10 * E + F) ∧
    (100 * G + 10 * E + F + 10 * D + E = 100 * H + 10 * F + J) ∧
    A = 2 ∧ B = 3 ∧ C = 0 ∧ D = 1 ∧ E = 7 ∧ F = 8 ∧ G = 4 ∧ H = 5 ∧ J = 6 :=
by sorry

end digit_puzzle_solution_l545_54588
