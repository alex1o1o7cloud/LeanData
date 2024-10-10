import Mathlib

namespace smallest_two_digit_multiple_plus_one_l346_34659

theorem smallest_two_digit_multiple_plus_one : ∃ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ (k : ℕ), n = 2 * k + 1) ∧
  (∃ (k : ℕ), n = 3 * k + 1) ∧
  (∃ (k : ℕ), n = 4 * k + 1) ∧
  (∃ (k : ℕ), n = 5 * k + 1) ∧
  (∃ (k : ℕ), n = 6 * k + 1) ∧
  (∀ (m : ℕ), m < n → 
    (m < 10 ∨ m ≥ 100 ∨
    (∀ (k : ℕ), m ≠ 2 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 3 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 4 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 5 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 6 * k + 1))) :=
by sorry

end smallest_two_digit_multiple_plus_one_l346_34659


namespace original_price_correct_l346_34677

/-- The original price of a dish, given specific discount and tip conditions --/
def original_price : ℝ := 24

/-- John's total payment for the dish --/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's total payment for the dish --/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions --/
theorem original_price_correct :
  john_payment original_price - jane_payment original_price = 0.36 :=
by sorry

end original_price_correct_l346_34677


namespace red_snapper_cost_l346_34629

/-- The cost of a Red snapper given the fisherman's daily catch and earnings -/
theorem red_snapper_cost (red_snappers : ℕ) (tunas : ℕ) (tuna_cost : ℚ) (daily_earnings : ℚ) : 
  red_snappers = 8 → tunas = 14 → tuna_cost = 2 → daily_earnings = 52 → 
  (daily_earnings - (tunas * tuna_cost)) / red_snappers = 3 := by
sorry

end red_snapper_cost_l346_34629


namespace element_order_l346_34617

-- Define the elements as a custom type
inductive Element : Type
  | A | B | C | D | E

-- Define the properties
def in_same_period (e₁ e₂ : Element) : Prop := sorry

def forms_basic_oxide (e : Element) : Prop := sorry

def basicity (e : Element) : ℝ := sorry

def hydride_stability (e : Element) : ℝ := sorry

def ionic_radius (e : Element) : ℝ := sorry

def atomic_number (e : Element) : ℕ := sorry

-- State the theorem
theorem element_order :
  (∀ e₁ e₂ : Element, in_same_period e₁ e₂) →
  forms_basic_oxide Element.A →
  forms_basic_oxide Element.B →
  basicity Element.B > basicity Element.A →
  hydride_stability Element.C > hydride_stability Element.D →
  (∀ e : Element, ionic_radius Element.E ≤ ionic_radius e) →
  (atomic_number Element.B < atomic_number Element.A ∧
   atomic_number Element.A < atomic_number Element.E ∧
   atomic_number Element.E < atomic_number Element.D ∧
   atomic_number Element.D < atomic_number Element.C) :=
by sorry


end element_order_l346_34617


namespace trig_identity_proof_l346_34628

/-- Proves that cos(70°)sin(80°) + cos(20°)sin(10°) = 1/2 -/
theorem trig_identity_proof : 
  Real.cos (70 * π / 180) * Real.sin (80 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end trig_identity_proof_l346_34628


namespace regular_polygon_sides_l346_34625

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end regular_polygon_sides_l346_34625


namespace bruce_bags_l346_34624

/-- Calculates the number of bags Bruce can buy with the change after purchasing crayons, books, and calculators. -/
def bags_bought (crayons_packs : ℕ) (crayon_price : ℕ) (books : ℕ) (book_price : ℕ) 
                (calculators : ℕ) (calculator_price : ℕ) (initial_money : ℕ) (bag_price : ℕ) : ℕ :=
  let total_spent := crayons_packs * crayon_price + books * book_price + calculators * calculator_price
  let change := initial_money - total_spent
  change / bag_price

/-- Theorem stating that Bruce can buy 11 bags with the change. -/
theorem bruce_bags : 
  bags_bought 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end bruce_bags_l346_34624


namespace arithmetic_sequence_ninth_term_l346_34638

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem states that for an arithmetic sequence satisfying given conditions, the 9th term equals 7. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 2)
  (h_fifth : a 5 = 3) :
  a 9 = 7 := by
  sorry

end arithmetic_sequence_ninth_term_l346_34638


namespace shoe_size_increase_l346_34675

/-- Represents the increase in length (in inches) for each unit increase in shoe size -/
def length_increase : ℝ := 0.2

/-- The smallest shoe size -/
def min_size : ℕ := 8

/-- The largest shoe size -/
def max_size : ℕ := 17

/-- The length of the size 15 shoe (in inches) -/
def size_15_length : ℝ := 10.4

/-- The ratio of the largest size length to the smallest size length -/
def size_ratio : ℝ := 1.2

theorem shoe_size_increase :
  (min_size : ℝ) + (max_size - min_size) * length_increase = (min_size : ℝ) * size_ratio ∧
  (min_size : ℝ) + (15 - min_size) * length_increase = size_15_length ∧
  length_increase = 0.2 := by sorry

end shoe_size_increase_l346_34675


namespace count_three_digit_Q_equal_l346_34663

def Q (n : ℕ) : ℕ := 
  n % 3 + n % 5 + n % 7 + n % 11

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_Q_equal : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_three_digit n ∧ Q n = Q (n + 1)) ∧ 
    S.card = 9 ∧
    (∀ n, is_three_digit n → Q n = Q (n + 1) → n ∈ S) :=
sorry

end count_three_digit_Q_equal_l346_34663


namespace rational_square_plus_one_positive_l346_34644

theorem rational_square_plus_one_positive (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end rational_square_plus_one_positive_l346_34644


namespace divisibility_by_hundred_l346_34632

theorem divisibility_by_hundred (n : ℕ) : 
  ∃ (k : ℕ), 100 ∣ (5^n + 12*n^2 + 12*n + 3) ↔ n = 5*k + 2 := by
  sorry

end divisibility_by_hundred_l346_34632


namespace range_of_b_and_m_l346_34641

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x - 1

-- Define the set of b values
def B : Set ℝ := {b | b < 0 ∨ b > 4}

-- Define the function F
def F (x m : ℝ) : ℝ := x^2 - m*(x - 1) + 1 - m - m^2

-- Define the set of m values
def M : Set ℝ := {m | -Real.sqrt (4/5) ≤ m ∧ m ≤ Real.sqrt (4/5) ∨ m ≥ 2}

theorem range_of_b_and_m :
  (∀ b : ℝ, (∃ x : ℝ, f x < b * g x) ↔ b ∈ B) ∧
  (∀ m : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → |F x m| < |F y m|) → m ∈ M) :=
sorry

end range_of_b_and_m_l346_34641


namespace unique_decreasing_term_l346_34670

def a (n : ℕ+) : ℚ := 4 / (11 - 2 * n)

theorem unique_decreasing_term :
  ∃! (n : ℕ+), a (n + 1) < a n :=
by
  sorry

end unique_decreasing_term_l346_34670


namespace ellipse_equation_l346_34609

/-- The equation of an ellipse with given parameters -/
theorem ellipse_equation (ε x₀ y₀ α : ℝ) (ε_pos : 0 < ε) (ε_lt_one : ε < 1) :
  let c : ℝ := (y₀ - x₀ * Real.tan α) / Real.tan α
  let a : ℝ := c / ε
  let b : ℝ := Real.sqrt (a^2 - c^2)
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔
    (x^2 / (c^2 / ε^2) + y^2 / ((c^2 / ε^2) - c^2) = 1) :=
by sorry

end ellipse_equation_l346_34609


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l346_34651

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let x₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*(-14))) / (2*1)
  let x₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*(-14))) / (2*1)
  x₁ + x₂ = 7 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l346_34651


namespace sum_coordinates_of_B_l346_34681

/-- Given two points A and B in the plane, where A is at the origin and B is on the line y = 5,
    and the slope of the line AB is 3/4, prove that the sum of the x- and y-coordinates of B is 35/3. -/
theorem sum_coordinates_of_B (A B : ℝ × ℝ) : 
  A = (0, 0) → 
  B.2 = 5 → 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 → 
  B.1 + B.2 = 35 / 3 := by
  sorry

end sum_coordinates_of_B_l346_34681


namespace binomial_coefficients_600_l346_34623

theorem binomial_coefficients_600 (n : ℕ) (h : n = 600) : 
  Nat.choose n n = 1 ∧ Nat.choose n 0 = 1 ∧ Nat.choose n 1 = n := by
  sorry

end binomial_coefficients_600_l346_34623


namespace base_equality_l346_34626

theorem base_equality : ∃ (n k : ℕ), n > 1 ∧ k > 1 ∧ n^2 + 1 = k^4 + k^3 + k + 1 := by
  sorry

end base_equality_l346_34626


namespace profit_sharing_ratio_l346_34664

/-- Represents an investment in a business. -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total investment value considering the amount and duration. -/
def investmentValue (i : Investment) : ℕ := i.amount * i.duration

/-- Represents the ratio of two numbers as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating that the profit sharing ratio is 2:3 given the investments of A and B. -/
theorem profit_sharing_ratio 
  (a : Investment) 
  (b : Investment) 
  (h1 : a.amount = 3500) 
  (h2 : a.duration = 12) 
  (h3 : b.amount = 9000) 
  (h4 : b.duration = 7) : 
  ∃ (r : Ratio), r.numerator = 2 ∧ r.denominator = 3 ∧ 
  investmentValue a * r.denominator = investmentValue b * r.numerator := by
  sorry


end profit_sharing_ratio_l346_34664


namespace book_price_increase_l346_34666

theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) (new_price : ℝ) :
  original_price = 300 →
  increase_percentage = 50 →
  new_price = original_price + (increase_percentage / 100) * original_price →
  new_price = 450 := by
sorry

end book_price_increase_l346_34666


namespace wire_service_reporters_l346_34601

theorem wire_service_reporters (total : ℝ) (x y both other_politics : ℝ) :
  x = 0.3 * total →
  y = 0.1 * total →
  both = 0.1 * total →
  other_politics = 0.25 * (x + y - both + other_politics) →
  total - (x + y - both + other_politics) = 0.45 * total :=
by sorry

end wire_service_reporters_l346_34601


namespace expression_equality_l346_34694

theorem expression_equality : (2004 - (2011 - 196)) + (2011 - (196 - 2004)) = 4008 := by
  sorry

end expression_equality_l346_34694


namespace parallelogram_area_l346_34603

/-- The area of a parallelogram with given side lengths and included angle -/
theorem parallelogram_area (a b : ℝ) (θ : Real) (ha : a = 32) (hb : b = 18) (hθ : θ = 75 * π / 180) :
  abs (a * b * Real.sin θ - 556.36) < 0.01 := by
  sorry

end parallelogram_area_l346_34603


namespace complex_equation_solution_l346_34676

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + Complex.I * Real.sqrt 3) * z = Complex.I * Real.sqrt 3 →
    z = (3 / 4 : ℂ) + Complex.I * (Real.sqrt 3 / 4) :=
by
  sorry

end complex_equation_solution_l346_34676


namespace fraction_product_simplification_l346_34620

theorem fraction_product_simplification (a b c : ℝ) 
  (ha : a ≠ 4) (hb : b ≠ 5) (hc : c ≠ 6) : 
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 := by
  sorry

end fraction_product_simplification_l346_34620


namespace eliminated_avg_is_four_l346_34683

/-- Represents an archery competition with the given conditions -/
structure ArcheryCompetition where
  n : ℕ  -- Half the number of participants
  max_score : ℕ
  advancing_avg : ℝ
  overall_avg_diff : ℝ

/-- The average score of eliminated contestants in the archery competition -/
def eliminated_avg (comp : ArcheryCompetition) : ℝ :=
  2 * comp.overall_avg_diff

/-- Theorem stating the average score of eliminated contestants is 4 points -/
theorem eliminated_avg_is_four (comp : ArcheryCompetition)
  (h1 : comp.max_score = 10)
  (h2 : comp.advancing_avg = 8)
  (h3 : comp.overall_avg_diff = 2) :
  eliminated_avg comp = 4 := by
  sorry

end eliminated_avg_is_four_l346_34683


namespace douglas_vote_percentage_l346_34690

theorem douglas_vote_percentage (total_percentage : ℝ) (ratio_x_to_y : ℝ) (y_percentage : ℝ) :
  total_percentage = 54 →
  ratio_x_to_y = 2 →
  y_percentage = 38.000000000000014 →
  ∃ x_percentage : ℝ,
    x_percentage = 62 ∧
    (x_percentage * (ratio_x_to_y / (ratio_x_to_y + 1)) + y_percentage * (1 / (ratio_x_to_y + 1))) = total_percentage :=
by sorry

end douglas_vote_percentage_l346_34690


namespace right_triangles_with_perimeter_equal_area_l346_34615

/-- A right triangle with integer side lengths. -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2

/-- The perimeter of a right triangle. -/
def perimeter (t : RightTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The area of a right triangle. -/
def area (t : RightTriangle) : ℕ :=
  t.a * t.b / 2

/-- The property that the perimeter equals the area. -/
def perimeter_equals_area (t : RightTriangle) : Prop :=
  perimeter t = area t

theorem right_triangles_with_perimeter_equal_area :
  {t : RightTriangle | perimeter_equals_area t} =
  {⟨5, 12, 13, by sorry⟩, ⟨6, 8, 10, by sorry⟩} :=
by sorry

end right_triangles_with_perimeter_equal_area_l346_34615


namespace sixth_root_of_24414062515625_l346_34689

theorem sixth_root_of_24414062515625 : (24414062515625 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end sixth_root_of_24414062515625_l346_34689


namespace gcd_51457_37958_is_1_l346_34699

theorem gcd_51457_37958_is_1 : Nat.gcd 51457 37958 = 1 := by
  sorry

end gcd_51457_37958_is_1_l346_34699


namespace parallelogram_product_l346_34627

/-- Given a parallelogram EFGH with side lengths as specified, 
    prove that the product of x and y is 57√2 -/
theorem parallelogram_product (x y : ℝ) : 
  58 = 3 * x + 1 →   -- EF = GH
  2 * y^2 = 36 →     -- FG = HE
  x * y = 57 * Real.sqrt 2 := by
sorry

end parallelogram_product_l346_34627


namespace positive_root_equation_l346_34671

theorem positive_root_equation : ∃ x : ℝ, x > 0 ∧ x^3 - 3*x^2 - x - Real.sqrt 2 = 0 :=
by
  use 2 + Real.sqrt 2
  sorry

end positive_root_equation_l346_34671


namespace intersection_segment_length_l346_34660

/-- Line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := x + y = 0

/-- Curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The length of segment AB formed by the intersection of line l and curve C -/
theorem intersection_segment_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end intersection_segment_length_l346_34660


namespace system_solution_l346_34679

theorem system_solution (x y : ℝ) : 
  ((x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = (Real.sqrt 17 - 1) / 2 ∧ y = (9 - Real.sqrt 17) / 2)) →
  (((x^2 * y^4)^(-Real.log x) = y^(Real.log (y / x^7))) ∧
   (y^2 - x*y - 2*x^2 + 8*x - 4*y = 0)) :=
by sorry

end system_solution_l346_34679


namespace ratio_inequality_l346_34692

theorem ratio_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b + b / c + c / a ≤ a^2 / b^2 + b^2 / c^2 + c^2 / a^2 := by
  sorry

end ratio_inequality_l346_34692


namespace arrangement_count_l346_34642

-- Define the number of children
def n : ℕ := 6

-- Define the number of odd positions available for the specific child
def odd_positions : ℕ := 3

-- Define the function to calculate the number of arrangements
def arrangements (n : ℕ) (odd_positions : ℕ) : ℕ :=
  odd_positions * Nat.factorial (n - 1)

-- Theorem statement
theorem arrangement_count :
  arrangements n odd_positions = 360 := by
  sorry

end arrangement_count_l346_34642


namespace triangle_area_l346_34604

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S = (2 * Real.sqrt 3) / 3) :=
by sorry

#check triangle_area

end triangle_area_l346_34604


namespace rectangle_perimeter_l346_34602

/-- Given a rectangle with sides a and b (in decimeters), prove that its perimeter is 20 decimeters
    if the sum of two sides is 10 and the sum of three sides is 14. -/
theorem rectangle_perimeter (a b : ℝ) : 
  a + b = 10 → a + a + b = 14 → 2 * (a + b) = 20 := by
  sorry

end rectangle_perimeter_l346_34602


namespace geometric_sequence_condition_l346_34650

-- Define what it means for three real numbers to form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

-- Theorem statement
theorem geometric_sequence_condition (a b c : ℝ) :
  (is_geometric_sequence a b c → a * c = b^2) ∧
  ∃ a b c : ℝ, a * c = b^2 ∧ ¬is_geometric_sequence a b c :=
by sorry

end geometric_sequence_condition_l346_34650


namespace sum_of_roots_is_3pi_l346_34662

-- Define the equation
def tanEquation (x : ℝ) : Prop := Real.tan x ^ 2 - 12 * Real.tan x + 4 = 0

-- Define the interval
def inInterval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem sum_of_roots_is_3pi :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, tanEquation x ∧ inInterval x) ∧
    (∀ x, tanEquation x ∧ inInterval x → x ∈ roots) ∧
    (Finset.sum roots id = 3 * Real.pi) :=
sorry

end sum_of_roots_is_3pi_l346_34662


namespace total_cost_packages_A_and_B_l346_34630

/-- Represents a subscription package with monthly cost, duration, and discount rate -/
structure Package where
  monthlyCost : ℝ
  duration : ℕ
  discountRate : ℝ

/-- Calculates the discounted cost of a package -/
def discountedCost (p : Package) : ℝ :=
  p.monthlyCost * p.duration * (1 - p.discountRate)

/-- The newspaper subscription packages -/
def packageA : Package := { monthlyCost := 10, duration := 6, discountRate := 0.1 }
def packageB : Package := { monthlyCost := 12, duration := 9, discountRate := 0.15 }

/-- Theorem stating the total cost of subscribing to Package A followed by Package B -/
theorem total_cost_packages_A_and_B :
  discountedCost packageA + discountedCost packageB = 145.80 := by
  sorry

#eval discountedCost packageA + discountedCost packageB

end total_cost_packages_A_and_B_l346_34630


namespace rosie_pies_l346_34605

/-- Given that Rosie can make 3 pies out of 12 apples, 
    prove that she can make 9 pies out of 36 apples. -/
theorem rosie_pies (apples_per_batch : ℕ) (pies_per_batch : ℕ) 
  (h1 : apples_per_batch = 12) 
  (h2 : pies_per_batch = 3) 
  (h3 : 36 = 3 * apples_per_batch) : 
  (36 / (apples_per_batch / pies_per_batch) : ℕ) = 9 := by
  sorry

end rosie_pies_l346_34605


namespace no_integer_solution_l346_34606

theorem no_integer_solution : ¬ ∃ (m n : ℤ), m^2 + 1954 = n^2 := by sorry

end no_integer_solution_l346_34606


namespace equal_division_possible_l346_34647

/-- Represents the state of the three vessels -/
structure VesselState :=
  (v1 v2 v3 : ℕ)

/-- Represents a pouring action between two vessels -/
inductive PourAction
  | from1to2 | from1to3 | from2to1 | from2to3 | from3to1 | from3to2

/-- Applies a pouring action to a vessel state -/
def applyPour (state : VesselState) (action : PourAction) : VesselState :=
  sorry

/-- Checks if a vessel state is valid (respects capacities) -/
def isValidState (state : VesselState) : Prop :=
  state.v1 ≤ 3 ∧ state.v2 ≤ 5 ∧ state.v3 ≤ 8

/-- Checks if a sequence of pours is valid -/
def isValidPourSequence (initialState : VesselState) (pours : List PourAction) : Prop :=
  sorry

/-- The theorem stating that it's possible to divide the liquid equally -/
theorem equal_division_possible : ∃ (pours : List PourAction),
  isValidPourSequence ⟨0, 0, 8⟩ pours ∧
  let finalState := pours.foldl applyPour ⟨0, 0, 8⟩
  finalState.v2 = 4 ∧ finalState.v3 = 4 :=
  sorry

end equal_division_possible_l346_34647


namespace dodecagon_diagonals_and_polygon_vertices_l346_34657

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals_and_polygon_vertices : 
  (num_diagonals 12 = 54) ∧ 
  (∃ n : ℕ, num_diagonals n = 135 ∧ n = 18) := by
  sorry

end dodecagon_diagonals_and_polygon_vertices_l346_34657


namespace closer_to_origin_l346_34637

theorem closer_to_origin : abs (-2 : ℝ) < abs (3 : ℝ) := by
  sorry

end closer_to_origin_l346_34637


namespace distinct_arrangements_count_l346_34667

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The symmetry group of a regular six-pointed star -/
def starSymmetryGroup : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    considering reflections and rotations as equivalent -/
def distinctArrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / starSymmetryGroup

theorem distinct_arrangements_count (star : SixPointedStar) :
  distinctArrangements star = 39916800 := by
  sorry

end distinct_arrangements_count_l346_34667


namespace max_page_number_proof_l346_34619

/-- Counts the number of '5' digits used in numbering pages from 1 to n --/
def count_fives (n : ℕ) : ℕ := sorry

/-- The highest page number that can be labeled with 16 '5' digits --/
def max_page_number : ℕ := 75

theorem max_page_number_proof :
  count_fives max_page_number ≤ 16 ∧
  ∀ m : ℕ, m > max_page_number → count_fives m > 16 :=
sorry

end max_page_number_proof_l346_34619


namespace factor_implies_b_equals_one_l346_34616

theorem factor_implies_b_equals_one (a b : ℤ) :
  (∃ c d : ℤ, ∀ x, (x^2 + x - 2) * (c*x + d) = a*x^3 - b*x^2 + x + 2) →
  b = 1 := by
sorry

end factor_implies_b_equals_one_l346_34616


namespace problem_solution_l346_34665

def p (a x : ℝ) : Prop := x^2 - (2*a - 3)*x - 6*a ≤ 0

def q (x : ℝ) : Prop := x - Real.sqrt x - 2 < 0

theorem problem_solution :
  (∀ x, (p 1 x ∧ q x) ↔ (0 ≤ x ∧ x ≤ 2)) ∧
  (∀ a, (∀ x, q x → p a x) ↔ a ≥ 2) := by sorry

end problem_solution_l346_34665


namespace algebraic_simplification_l346_34655

theorem algebraic_simplification (y : ℝ) (h : y ≠ 0) :
  (20 * y^3) * (8 * y^2) * (1 / (4*y)^3) = (5/2) * y^2 := by
  sorry

end algebraic_simplification_l346_34655


namespace min_value_problem_l346_34611

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) ≥ 3 ∧
  ((x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) = 3 →
    z / (16 * y) + x / 9 ≥ 2) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    (x₀^2 * y₀ * z₀) / 324 + (144 * y₀) / (x₀ * z₀) + 9 / (4 * x₀ * y₀^2) = 3 ∧
    z₀ / (16 * y₀) + x₀ / 9 = 2 ∧
    x₀ = 9 ∧ y₀ = (1/2) ∧ z₀ = 16 := by
  sorry

end min_value_problem_l346_34611


namespace simplify_polynomial_simplify_expression_l346_34686

-- Problem 1
theorem simplify_polynomial (x : ℝ) :
  2*x^3 - 4*x^2 - 3*x - 2*x^2 - x^3 + 5*x - 7 = x^3 - 6*x^2 + 2*x - 7 := by
  sorry

-- Problem 2
theorem simplify_expression (m n : ℝ) :
  let A := 2*m^2 - m*n
  let B := m^2 + 2*m*n - 5
  4*A - 2*B = 6*m^2 - 8*m*n + 10 := by
  sorry

end simplify_polynomial_simplify_expression_l346_34686


namespace max_gcd_of_sum_1111_l346_34610

theorem max_gcd_of_sum_1111 :
  ∃ (a b : ℕ+), a + b = 1111 ∧ 
  ∀ (c d : ℕ+), c + d = 1111 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 101 :=
sorry

end max_gcd_of_sum_1111_l346_34610


namespace find_a_value_l346_34613

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem find_a_value : ∃ a : ℝ, (B a ⊆ A a) ∧ (a = -1 ∨ a = 2) := by
  sorry

end find_a_value_l346_34613


namespace sin_period_omega_l346_34673

/-- 
Given a function y = sin(ωx - π/3) with ω > 0 and a minimum positive period of π,
prove that ω = 2.
-/
theorem sin_period_omega (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, ∃ y, y = Real.sin (ω * x - π / 3)) 
  (h3 : ∀ T > 0, (∀ x, Real.sin (ω * (x + T) - π / 3) = Real.sin (ω * x - π / 3)) → T ≥ π) 
  (h4 : ∀ x, Real.sin (ω * (x + π) - π / 3) = Real.sin (ω * x - π / 3)) : ω = 2 := by
  sorry

end sin_period_omega_l346_34673


namespace range_of_a_l346_34631

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x < 3 → (a - 1) * x < a + 3) ↔ (1 ≤ a ∧ a < 3) := by
  sorry

end range_of_a_l346_34631


namespace min_digit_sum_of_sum_l346_34672

/-- Two-digit number type -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- Function to sum the digits of a natural number -/
def digitSum (n : ℕ) : ℕ := (digits n).sum

/-- Predicate to check if two two-digit numbers have exactly one common digit -/
def hasOneCommonDigit (a b : TwoDigitNumber) : Prop := sorry

/-- Theorem: The smallest possible digit sum of S, where S is the sum of two two-digit numbers
    with exactly one common digit, and S is a three-digit number, is 2. -/
theorem min_digit_sum_of_sum (a b : TwoDigitNumber) 
  (h1 : hasOneCommonDigit a b) 
  (h2 : a.val + b.val ≥ 100 ∧ a.val + b.val ≤ 999) : 
  ∃ (S : ℕ), S = a.val + b.val ∧ digitSum S = 2 ∧ 
  ∀ (T : ℕ), T = a.val + b.val → digitSum T ≥ 2 :=
sorry

end min_digit_sum_of_sum_l346_34672


namespace average_value_of_z_squared_l346_34636

theorem average_value_of_z_squared (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = (46 * z^2) / 5 := by
  sorry

end average_value_of_z_squared_l346_34636


namespace complex_solution_l346_34684

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_solution (z : ℂ) (h : determinant z 1 z (2 * Complex.I) = 3 + 2 * Complex.I) :
  z = (1 / 5 : ℂ) - (8 / 5 : ℂ) * Complex.I :=
by sorry

end complex_solution_l346_34684


namespace sqrt_equation_solution_l346_34697

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 * x + 11) = 9 → x = 17.5 := by
  sorry

end sqrt_equation_solution_l346_34697


namespace pizza_slices_per_adult_l346_34622

theorem pizza_slices_per_adult (num_adults num_children num_pizzas slices_per_pizza slices_per_child : ℕ) :
  num_adults = 2 →
  num_children = 6 →
  num_pizzas = 3 →
  slices_per_pizza = 4 →
  slices_per_child = 1 →
  (num_pizzas * slices_per_pizza - num_children * slices_per_child) / num_adults = 3 :=
by
  sorry

end pizza_slices_per_adult_l346_34622


namespace students_on_right_side_l346_34633

theorem students_on_right_side (total : ℕ) (left : ℕ) (right : ℕ) : 
  total = 63 → left = 36 → right = total - left → right = 27 := by
  sorry

end students_on_right_side_l346_34633


namespace integer_solution_cyclic_equation_l346_34612

theorem integer_solution_cyclic_equation :
  ∀ x y z : ℤ, (x + y + z)^5 = 80*x*y*z*(x^2 + y^2 + z^2) →
  (∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨
            (x = a ∧ y = 0 ∧ z = -a) ∨
            (x = 0 ∧ y = a ∧ z = -a) ∨
            (x = -a ∧ y = a ∧ z = 0) ∨
            (x = -a ∧ y = 0 ∧ z = a) ∨
            (x = 0 ∧ y = -a ∧ z = a)) :=
by sorry

end integer_solution_cyclic_equation_l346_34612


namespace expression_with_eight_factors_l346_34695

theorem expression_with_eight_factors
  (x y : ℕ)
  (hx_prime : Nat.Prime x)
  (hy_prime : Nat.Prime y)
  (hx_odd : Odd x)
  (hy_odd : Odd y)
  (hxy_lt : x < y) :
  (Finset.filter (fun d => (x^3 * y) % d = 0) (Finset.range (x^3 * y + 1))).card = 8 :=
sorry

end expression_with_eight_factors_l346_34695


namespace function_satisfying_conditions_l346_34696

theorem function_satisfying_conditions (f : ℝ → ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) →
  f 1 = 1 →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y)) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y) →
  (∀ x : ℝ, x ≠ 0 → f x = 1 / x) :=
by sorry

end function_satisfying_conditions_l346_34696


namespace distinct_paths_count_l346_34668

/-- The number of floors in the building -/
def num_floors : ℕ := 5

/-- The number of staircases between each consecutive floor -/
def staircases_per_floor : ℕ := 2

/-- The number of floors to descend -/
def floors_to_descend : ℕ := num_floors - 1

/-- The number of distinct paths from the top floor to the bottom floor -/
def num_paths : ℕ := staircases_per_floor ^ floors_to_descend

theorem distinct_paths_count :
  num_paths = 16 := by sorry

end distinct_paths_count_l346_34668


namespace translated_parabola_vertex_l346_34639

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation
def translation_left : ℝ := 3
def translation_down : ℝ := 2

-- Theorem stating the vertex of the translated parabola
theorem translated_parabola_vertex :
  let vertex_x : ℝ := 2 - translation_left
  let vertex_y : ℝ := original_parabola 2 - translation_down
  (vertex_x, vertex_y) = (-1, -4) := by sorry

end translated_parabola_vertex_l346_34639


namespace equality_statements_l346_34608

theorem equality_statements :
  (∀ a b : ℝ, a - 3 = b - 3 → a = b) ∧
  (∀ a b m : ℝ, m ≠ 0 → a / m = b / m → a = b) := by sorry

end equality_statements_l346_34608


namespace right_triangle_sin_complement_l346_34614

theorem right_triangle_sin_complement (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  B = π / 2 →
  Real.sin A = 3 / 5 →
  Real.sin C = 4 / 5 :=
by sorry

end right_triangle_sin_complement_l346_34614


namespace range_of_x_when_a_is_one_range_of_a_when_p_necessary_not_sufficient_l346_34656

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ x ∈ Set.Ioo 2 3 :=
sorry

-- Part 2
theorem range_of_a_when_p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x 1) ∧ 
  (∃ x : ℝ, p x 1 ∧ ¬q x) ↔
  1 ∈ Set.Ioo 1 2 :=
sorry

end range_of_x_when_a_is_one_range_of_a_when_p_necessary_not_sufficient_l346_34656


namespace sum_of_solutions_equals_sqrt_five_l346_34649

theorem sum_of_solutions_equals_sqrt_five (x₀ y₀ : ℝ) 
  (h1 : y₀ = 1 / x₀) 
  (h2 : y₀ = |x₀| + 1) : 
  x₀ + y₀ = Real.sqrt 5 := by
sorry

end sum_of_solutions_equals_sqrt_five_l346_34649


namespace number_control_l346_34645

def increase_number (n : ℕ) : ℕ := n + 102

def can_rearrange_to_three_digits (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ n = a * 100 + b * 10 + c

theorem number_control (start : ℕ) (h_start : start = 123) :
  ∀ (t : ℕ), ∃ (n : ℕ), 
    n ≤ increase_number^[t] start ∧
    can_rearrange_to_three_digits n :=
by sorry

end number_control_l346_34645


namespace eighteen_horses_walking_legs_l346_34685

/-- Calculates the number of legs walking on the ground given the number of horses --/
def legsWalking (numHorses : ℕ) : ℕ :=
  let numMen := numHorses
  let numWalkingMen := numMen / 2
  let numWalkingHorses := numWalkingMen
  2 * numWalkingMen + 4 * numWalkingHorses

theorem eighteen_horses_walking_legs :
  legsWalking 18 = 54 := by
  sorry

end eighteen_horses_walking_legs_l346_34685


namespace parabola_normals_intersection_l346_34661

/-- The condition for three distinct points on a parabola to have intersecting normals -/
theorem parabola_normals_intersection
  (a b c : ℝ)
  (h_distinct : (a - b) * (b - c) * (c - a) ≠ 0)
  (h_parabola : ∀ (x : ℝ), (x = a ∨ x = b ∨ x = c) → ∃ (y : ℝ), y = x^2) :
  (∃ (p : ℝ × ℝ),
    (∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c) →
      (y - x^2) = -(1 / (2*x)) * (p.1 - x) ∧ p.2 = y)) ↔
  a + b + c = 0 :=
sorry

end parabola_normals_intersection_l346_34661


namespace arithmetic_sequence_exponents_l346_34687

theorem arithmetic_sequence_exponents (a b : ℝ) (m : ℝ) : 
  a > 0 → b > 0 → 
  2^a = m → 3^b = m → 
  2 * a * b = a + b → 
  m = Real.sqrt 6 := by
  sorry

end arithmetic_sequence_exponents_l346_34687


namespace quadratic_inequality_range_l346_34640

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) ↔ a ≥ 1 :=
by sorry

end quadratic_inequality_range_l346_34640


namespace min_value_reciprocal_sum_l346_34635

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 9 := by
  sorry

end min_value_reciprocal_sum_l346_34635


namespace specific_number_probability_l346_34688

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of favorable outcomes for a specific type of number -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a specific type of number when tossing two dice -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem specific_number_probability :
  probability = 5 / 12 := by sorry

end specific_number_probability_l346_34688


namespace solve_for_n_l346_34648

/-- The number of balls labeled '2' -/
def n : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := n + 2

/-- The probability of drawing a ball labeled '2' -/
def prob_2 : ℚ := n / total_balls

theorem solve_for_n : 
  (prob_2 = 1/3) → n = 1 :=
by sorry

end solve_for_n_l346_34648


namespace cos_two_theta_collinear_vectors_l346_34682

/-- Given two vectors AB and BC in 2D space, and that points A, B, and C are collinear,
    prove that cos(2θ) = 7/9 where θ is the angle in the definition of BC. -/
theorem cos_two_theta_collinear_vectors 
  (AB : ℝ × ℝ) 
  (BC : ℝ → ℝ × ℝ) 
  (h_AB : AB = (-1, -3))
  (h_BC : ∀ θ, BC θ = (2 * Real.sin θ, 2))
  (h_collinear : ∀ θ, ∃ k : ℝ, AB = k • BC θ) :
  ∃ θ, Real.cos (2 * θ) = 7/9 := by
  sorry

end cos_two_theta_collinear_vectors_l346_34682


namespace roots_sum_product_l346_34691

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 1 = 0) → 
  (b^4 - 6*b - 1 = 0) → 
  (a ≠ b) →
  (a*b + a + b = 1) := by
sorry

end roots_sum_product_l346_34691


namespace ngo_wage_problem_l346_34658

/-- Calculates the initial daily average wage of illiterate employees in an NGO -/
def initial_illiterate_wage (num_illiterate : ℕ) (num_literate : ℕ) (new_illiterate_wage : ℕ) (overall_decrease : ℕ) : ℕ :=
  let total_employees := num_illiterate + num_literate
  let total_wage_decrease := total_employees * overall_decrease
  (total_wage_decrease + num_illiterate * new_illiterate_wage) / num_illiterate

theorem ngo_wage_problem :
  initial_illiterate_wage 20 10 10 10 = 25 := by
  sorry

end ngo_wage_problem_l346_34658


namespace right_triangle_inscribed_circle_area_l346_34643

theorem right_triangle_inscribed_circle_area
  (r : ℝ) (c : ℝ) (h_r : r = 5) (h_c : c = 34) :
  let s := (c + 2 * r + (c - 2 * r)) / 2
  r * s = 195 := by
sorry

end right_triangle_inscribed_circle_area_l346_34643


namespace fried_chicken_cost_l346_34669

/-- Calculates the cost of fried chicken given the total spent and other expenses at a club. -/
theorem fried_chicken_cost
  (entry_fee : ℚ)
  (drink_cost : ℚ)
  (friends : ℕ)
  (rounds : ℕ)
  (james_drinks : ℕ)
  (tip_rate : ℚ)
  (total_spent : ℚ)
  (h_entry_fee : entry_fee = 20)
  (h_drink_cost : drink_cost = 6)
  (h_friends : friends = 5)
  (h_rounds : rounds = 2)
  (h_james_drinks : james_drinks = 6)
  (h_tip_rate : tip_rate = 0.3)
  (h_total_spent : total_spent = 163)
  : ∃ (chicken_cost : ℚ),
    chicken_cost = 14 ∧
    total_spent = entry_fee +
                  (friends * rounds + james_drinks) * drink_cost +
                  chicken_cost +
                  ((friends * rounds + james_drinks) * drink_cost + chicken_cost) * tip_rate :=
by sorry


end fried_chicken_cost_l346_34669


namespace arithmetic_series_sum_plus_100_l346_34674

theorem arithmetic_series_sum_plus_100 : 
  let a₁ : ℕ := 10
  let aₙ : ℕ := 100
  let d : ℕ := 1
  let n : ℕ := (aₙ - a₁) / d + 1
  let series_sum : ℕ := n * (a₁ + aₙ) / 2
  series_sum + 100 = 5105 := by
sorry

end arithmetic_series_sum_plus_100_l346_34674


namespace vector_decomposition_l346_34654

def x : Fin 3 → ℝ := ![(-9 : ℝ), 5, 5]
def p : Fin 3 → ℝ := ![(4 : ℝ), 1, 1]
def q : Fin 3 → ℝ := ![(2 : ℝ), 0, -3]
def r : Fin 3 → ℝ := ![(-1 : ℝ), 2, 1]

theorem vector_decomposition :
  x = (-1 : ℝ) • p + (-1 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end vector_decomposition_l346_34654


namespace equation_system_solution_l346_34698

theorem equation_system_solution :
  ∀ (x y z : ℝ),
    z ≠ 0 →
    3 * x - 5 * y - z = 0 →
    2 * x + 4 * y - 16 * z = 0 →
    (x^2 + 4*x*y) / (2*y^2 + z^2) = 4.35 := by
  sorry

end equation_system_solution_l346_34698


namespace park_visitors_difference_l346_34600

theorem park_visitors_difference (saturday_visitors : ℕ) (total_visitors : ℕ) 
    (h1 : saturday_visitors = 200)
    (h2 : total_visitors = 440) : 
  total_visitors - 2 * saturday_visitors = 40 := by
  sorry

end park_visitors_difference_l346_34600


namespace intersection_point_l346_34693

-- Define the two lines
def line1 (x y : ℚ) : Prop := y = -3 * x + 1
def line2 (x y : ℚ) : Prop := y + 5 = 15 * x - 2

-- Theorem statement
theorem intersection_point :
  ∃ (x y : ℚ), line1 x y ∧ line2 x y ∧ x = 1/3 ∧ y = 0 := by
  sorry

end intersection_point_l346_34693


namespace remaining_fabric_is_294_l346_34621

/-- Represents the flag-making scenario with given dimensions and quantities --/
structure FlagScenario where
  total_fabric : ℕ
  square_side : ℕ
  wide_length : ℕ
  wide_width : ℕ
  tall_length : ℕ
  tall_width : ℕ
  square_count : ℕ
  wide_count : ℕ
  tall_count : ℕ

/-- Calculates the remaining fabric after making flags --/
def remaining_fabric (scenario : FlagScenario) : ℕ :=
  scenario.total_fabric -
  (scenario.square_count * scenario.square_side * scenario.square_side +
   scenario.wide_count * scenario.wide_length * scenario.wide_width +
   scenario.tall_count * scenario.tall_length * scenario.tall_width)

/-- Theorem stating that the remaining fabric in the given scenario is 294 square feet --/
theorem remaining_fabric_is_294 (scenario : FlagScenario)
  (h1 : scenario.total_fabric = 1000)
  (h2 : scenario.square_side = 4)
  (h3 : scenario.wide_length = 5)
  (h4 : scenario.wide_width = 3)
  (h5 : scenario.tall_length = 3)
  (h6 : scenario.tall_width = 5)
  (h7 : scenario.square_count = 16)
  (h8 : scenario.wide_count = 20)
  (h9 : scenario.tall_count = 10) :
  remaining_fabric scenario = 294 := by
  sorry


end remaining_fabric_is_294_l346_34621


namespace pure_imaginary_fraction_l346_34646

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) / (2 + Complex.I) = Complex.I * b) → a = 1/2 := by
  sorry

end pure_imaginary_fraction_l346_34646


namespace line_slope_l346_34618

/-- The slope of the line given by the equation 4y + 5x = 20 is -5/4 -/
theorem line_slope (x y : ℝ) : 4 * y + 5 * x = 20 → (y - 5) / (-5 / 4) = x := by
  sorry

end line_slope_l346_34618


namespace complement_A_intersect_B_when_m_3_A_union_B_equals_A_iff_m_in_range_l346_34678

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem complement_A_intersect_B_when_m_3 :
  (Set.univ \ A) ∩ B 3 = {5} := by sorry

theorem A_union_B_equals_A_iff_m_in_range (m : ℝ) :
  A ∪ B m = A ↔ m < 3 := by sorry

end complement_A_intersect_B_when_m_3_A_union_B_equals_A_iff_m_in_range_l346_34678


namespace sphere_cylinder_volume_difference_l346_34652

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 5) (h_cylinder : r_cylinder = 3) :
  ∃ (h_cylinder : ℝ),
    (4 / 3 * π * r_sphere ^ 3) - (π * r_cylinder ^ 2 * h_cylinder) = (284 / 3 : ℝ) * π := by
  sorry

end sphere_cylinder_volume_difference_l346_34652


namespace biased_dice_expected_value_l346_34680

-- Define the probabilities and payoffs
def prob_odd : ℚ := 1/3
def prob_2 : ℚ := 1/9
def prob_4 : ℚ := 1/18
def prob_6 : ℚ := 1/9
def payoff_odd : ℚ := 4
def payoff_even : ℚ := -6

-- Define the expected value function
def expected_value (p_odd p_2 p_4 p_6 pay_odd pay_even : ℚ) : ℚ :=
  3 * p_odd * pay_odd + p_2 * pay_even + p_4 * pay_even + p_6 * pay_even

-- Theorem statement
theorem biased_dice_expected_value :
  expected_value prob_odd prob_2 prob_4 prob_6 payoff_odd payoff_even = 7/3 := by
  sorry

end biased_dice_expected_value_l346_34680


namespace unbroken_seashells_l346_34607

theorem unbroken_seashells (total_seashells broken_seashells : ℕ) 
  (h1 : total_seashells = 6)
  (h2 : broken_seashells = 4) :
  total_seashells - broken_seashells = 2 :=
by sorry

end unbroken_seashells_l346_34607


namespace geometric_sequence_tan_property_l346_34653

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_tan_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 6 + 2 * (a 4)^2 = Real.pi) :
  Real.tan (a 3 * a 5) = Real.sqrt 3 := by
  sorry

end geometric_sequence_tan_property_l346_34653


namespace scaled_job_workforce_l346_34634

/-- Calculates the number of men needed for a scaled job given the original workforce and timelines. -/
def men_needed_for_scaled_job (original_men : ℕ) (original_days : ℕ) (scale_factor : ℕ) (new_days : ℕ) : ℕ :=
  (original_men * original_days * scale_factor) / new_days

/-- Proves that 600 men are needed for a job 3 times the original size, given the original conditions. -/
theorem scaled_job_workforce :
  men_needed_for_scaled_job 250 16 3 20 = 600 := by
  sorry

#eval men_needed_for_scaled_job 250 16 3 20

end scaled_job_workforce_l346_34634
