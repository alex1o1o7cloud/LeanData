import Mathlib

namespace NUMINAMATH_CALUDE_det_max_value_l3334_333437

open Real Matrix

theorem det_max_value (θ : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1 + tan θ, 1, 1; 1, 1, 1 + cos θ]
  ∀ φ : ℝ, det A ≤ det (!![1, 1, 1; 1 + tan φ, 1, 1; 1, 1, 1 + cos φ]) :=
by sorry

end NUMINAMATH_CALUDE_det_max_value_l3334_333437


namespace NUMINAMATH_CALUDE_sin_equality_proof_l3334_333462

theorem sin_equality_proof (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (-474 * π / 180) → n = 66 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l3334_333462


namespace NUMINAMATH_CALUDE_function_and_extrema_l3334_333448

noncomputable def f (a b c x : ℝ) : ℝ := a * x - b / x + c

theorem function_and_extrema :
  ∀ a b c : ℝ,
  (f a b c 1 = 0) →
  (∀ x : ℝ, x ≠ 0 → HasDerivAt (f a b c) (-x + 3) 2) →
  (∀ x : ℝ, x ≠ 0 → f a b c x = -3 * x - 8 / x + 11) ∧
  (∃ x : ℝ, f a b c x = 11 + 4 * Real.sqrt 6 ∧ IsLocalMin (f a b c) x) ∧
  (∃ x : ℝ, f a b c x = 11 - 4 * Real.sqrt 6 ∧ IsLocalMax (f a b c) x) :=
by sorry

end NUMINAMATH_CALUDE_function_and_extrema_l3334_333448


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3334_333401

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3334_333401


namespace NUMINAMATH_CALUDE_remainder_of_2614303940317_div_13_l3334_333454

theorem remainder_of_2614303940317_div_13 : 
  2614303940317 % 13 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2614303940317_div_13_l3334_333454


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3334_333483

theorem algebraic_expression_value (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 3 / (a + 2)) / ((a^2 - 1) / (a + 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3334_333483


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3334_333482

theorem vector_equation_solution :
  let a : ℚ := 23/7
  let b : ℚ := -1/7
  let v1 : Fin 2 → ℚ := ![1, 4]
  let v2 : Fin 2 → ℚ := ![3, -2]
  let result : Fin 2 → ℚ := ![2, 10]
  (a • v1 + b • v2 = result) := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3334_333482


namespace NUMINAMATH_CALUDE_ellipse_k_value_l3334_333463

/-- The equation of an ellipse with a parameter k -/
def ellipse_equation (x y k : ℝ) : Prop := x^2 + (k*y^2)/5 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (0, 2)

/-- Theorem: For an ellipse with the given equation and focus, k equals 1 -/
theorem ellipse_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, ellipse_equation x y k) ∧ 
  (focus.1 = 0 ∧ focus.2 = 2) → k = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l3334_333463


namespace NUMINAMATH_CALUDE_min_value_theorem_l3334_333411

/-- Given that the solution set of (x+2)/(x+1) < 0 is {x | a < x < b},
    and point A(a,b) lies on the line mx + ny + 1 = 0 where mn > 0,
    prove that the minimum value of 2/m + 1/n is 9. -/
theorem min_value_theorem (a b m n : ℝ) : 
  (∀ x, (x + 2) / (x + 1) < 0 ↔ a < x ∧ x < b) →
  m * a + n * b + 1 = 0 →
  m * n > 0 →
  (∀ m' n', m' * n' > 0 → 2 / m' + 1 / n' ≥ 2 / m + 1 / n) →
  2 / m + 1 / n = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3334_333411


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l3334_333428

/-- The number of ways to distribute n distinct objects into m distinct containers,
    where each container can hold multiple objects. -/
def distribute (n m : ℕ) : ℕ := m^n

/-- The number of distinct arrangements for wearing 5 different rings
    on the 5 fingers of the right hand. -/
def ringArrangements : ℕ := distribute 5 5

theorem ring_arrangements_count :
  ringArrangements = 5^5 := by sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l3334_333428


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l3334_333441

/-- Represents the number of atoms of a particular element in a compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given its atom counts and atomic weights -/
def molecularWeight (atoms : AtomCount) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  atoms.carbon * carbonWeight + atoms.hydrogen * hydrogenWeight + atoms.oxygen * oxygenWeight

/-- Theorem stating that a compound with formula C3H6 and molecular weight 58 g/mol contains 1 oxygen atom -/
theorem compound_oxygen_count : 
  ∀ (atoms : AtomCount),
    atoms.carbon = 3 →
    atoms.hydrogen = 6 →
    molecularWeight atoms 12.01 1.008 16.00 = 58 →
    atoms.oxygen = 1 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_count_l3334_333441


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3334_333450

theorem negative_fraction_comparison : -3/4 > -6/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3334_333450


namespace NUMINAMATH_CALUDE_books_from_first_shop_l3334_333416

theorem books_from_first_shop 
  (total_first : ℕ) 
  (books_second : ℕ) 
  (total_second : ℕ) 
  (avg_price : ℕ) :
  total_first = 1500 →
  books_second = 60 →
  total_second = 340 →
  avg_price = 16 →
  ∃ (books_first : ℕ), 
    (total_first + total_second) / (books_first + books_second) = avg_price ∧
    books_first = 55 :=
by sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l3334_333416


namespace NUMINAMATH_CALUDE_select_team_count_l3334_333436

/-- The number of ways to select a team of 4 students from a group of 6 boys and 8 girls, with at least 2 boys -/
def selectTeam (numBoys : ℕ) (numGirls : ℕ) : ℕ :=
  Nat.choose numBoys 2 * Nat.choose numGirls 2 +
  Nat.choose numBoys 3 * Nat.choose numGirls 1 +
  Nat.choose numBoys 4

/-- Theorem stating that the number of ways to select the team is 595 -/
theorem select_team_count :
  selectTeam 6 8 = 595 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l3334_333436


namespace NUMINAMATH_CALUDE_square_inequality_for_negatives_l3334_333470

theorem square_inequality_for_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_for_negatives_l3334_333470


namespace NUMINAMATH_CALUDE_inequality_proof_l3334_333464

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt a + Real.sqrt b)^8 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3334_333464


namespace NUMINAMATH_CALUDE_men_joined_correct_l3334_333455

/-- The number of men who joined the camp -/
def men_joined : ℕ := 30

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

/-- The initial number of days the food would last -/
def initial_days : ℕ := 20

/-- The number of days the food lasts after more men join -/
def final_days : ℕ := 5

/-- The total amount of food in man-days -/
def total_food : ℕ := initial_men * initial_days

theorem men_joined_correct :
  (initial_men + men_joined) * final_days = total_food :=
by sorry

end NUMINAMATH_CALUDE_men_joined_correct_l3334_333455


namespace NUMINAMATH_CALUDE_middle_number_is_eleven_l3334_333408

theorem middle_number_is_eleven (x y z : ℕ) 
  (sum_xy : x + y = 18) 
  (sum_xz : x + z = 23) 
  (sum_yz : y + z = 27) : 
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_is_eleven_l3334_333408


namespace NUMINAMATH_CALUDE_red_triangle_or_blue_quadrilateral_l3334_333431

/-- A type representing the color of an edge --/
inductive Color
| Red
| Blue

/-- A complete graph with 9 vertices --/
def Graph9 := Fin 9 → Fin 9 → Color

/-- A predicate that checks if a graph is complete --/
def is_complete (g : Graph9) : Prop :=
  ∀ i j : Fin 9, i ≠ j → (g i j = Color.Red ∨ g i j = Color.Blue)

/-- A predicate that checks if three vertices form a red triangle --/
def has_red_triangle (g : Graph9) : Prop :=
  ∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g i j = Color.Red ∧ g j k = Color.Red ∧ g i k = Color.Red

/-- A predicate that checks if four vertices form a blue complete quadrilateral --/
def has_blue_quadrilateral (g : Graph9) : Prop :=
  ∃ i j k l : Fin 9, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    g i j = Color.Blue ∧ g i k = Color.Blue ∧ g i l = Color.Blue ∧
    g j k = Color.Blue ∧ g j l = Color.Blue ∧ g k l = Color.Blue

/-- The main theorem --/
theorem red_triangle_or_blue_quadrilateral (g : Graph9) 
  (h : is_complete g) : has_red_triangle g ∨ has_blue_quadrilateral g := by
  sorry

end NUMINAMATH_CALUDE_red_triangle_or_blue_quadrilateral_l3334_333431


namespace NUMINAMATH_CALUDE_principal_calculation_l3334_333404

/-- The compound interest formula for yearly compounding -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem principal_calculation (final_amount : ℝ) (rate : ℝ) (time : ℕ)
  (h_final : final_amount = 3087)
  (h_rate : rate = 0.05)
  (h_time : time = 2) :
  ∃ principal : ℝ, 
    compound_interest principal rate time = final_amount ∧ 
    principal = 2800 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3334_333404


namespace NUMINAMATH_CALUDE_sqrt_six_irrational_between_two_and_three_l3334_333459

theorem sqrt_six_irrational_between_two_and_three :
  ∃ x : ℝ, Irrational x ∧ 2 < x ∧ x < 3 :=
by
  use Real.sqrt 6
  sorry

end NUMINAMATH_CALUDE_sqrt_six_irrational_between_two_and_three_l3334_333459


namespace NUMINAMATH_CALUDE_intersection_M_N_l3334_333410

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3334_333410


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3334_333400

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- first leg
  b : ℝ  -- second leg
  c : ℝ  -- hypotenuse
  h : ℝ  -- altitude to hypotenuse
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  altitude_prop : h * c = a * b  -- property of altitude in right triangle

-- State the theorem
theorem right_triangle_inequality (t : RightTriangle) : t.a + t.b < t.c + t.h := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3334_333400


namespace NUMINAMATH_CALUDE_garden_breadth_l3334_333415

/-- The breadth of a rectangular garden with perimeter 600 meters and length 100 meters is 200 meters. -/
theorem garden_breadth (perimeter length breadth : ℝ) 
  (h1 : perimeter = 600)
  (h2 : length = 100)
  (h3 : perimeter = 2 * (length + breadth)) : 
  breadth = 200 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l3334_333415


namespace NUMINAMATH_CALUDE_peanuts_problem_l3334_333481

/-- The number of peanuts remaining in the jar after a series of distributions and consumptions -/
def peanuts_remaining (initial : ℕ) : ℕ :=
  let brock_ate := initial / 3
  let after_brock := initial - brock_ate
  let per_family := after_brock / 3
  let bonita_per_family := (2 * per_family) / 5
  let after_bonita_per_family := per_family - bonita_per_family
  let after_bonita_total := after_bonita_per_family * 3
  let carlos_ate := after_bonita_total / 5
  after_bonita_total - carlos_ate

/-- Theorem stating that given the initial conditions, 216 peanuts remain in the jar -/
theorem peanuts_problem : peanuts_remaining 675 = 216 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_problem_l3334_333481


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3334_333487

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem subset_implies_a_values (a : ℝ) : 
  B a ⊆ A → a ∈ ({-1/2, 1/3, 0} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3334_333487


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3334_333471

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem f_derivative_at_2 : 
  (deriv f) 2 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3334_333471


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_property_l3334_333447

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility_property :
  ∃! (a b m : ℕ), 
    0 < a ∧ a < m ∧
    0 < b ∧ b < m ∧
    (∀ n : ℕ, n > 0 → ∃ k : ℤ, fibonacci n - a * n * b^n = m * k) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_property_l3334_333447


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l3334_333449

theorem product_of_sums_and_differences : (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * (2 * Real.sqrt 5 - 5 * Real.sqrt 2) = -30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l3334_333449


namespace NUMINAMATH_CALUDE_group_collection_l3334_333469

/-- Calculates the total collection in rupees for a group where each member contributes as many paise as the number of members -/
def total_collection (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Proves that for a group of 68 members, the total collection is 46.24 rupees -/
theorem group_collection :
  total_collection 68 = 46.24 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_l3334_333469


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3334_333403

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) (c : ℝ) : ℝ := 2^n + c

/-- The n-th term of the sequence a_n -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n-1) c

/-- Predicate to check if a sequence is geometric -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n+1) = r * a n

theorem geometric_sequence_condition (c : ℝ) :
  is_geometric_sequence (a · c) ↔ c = -1 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3334_333403


namespace NUMINAMATH_CALUDE_correct_statement_l3334_333444

/-- Proposition p: There exists an x₀ ∈ ℝ such that x₀² + x₀ + 1 ≤ 0 -/
def p : Prop := ∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0

/-- Proposition q: The function f(x) = x^(1/3) is an increasing function -/
def q : Prop := ∀ x y : ℝ, x < y → Real.rpow x (1/3) < Real.rpow y (1/3)

/-- The correct statement is (¬p) ∨ q -/
theorem correct_statement : (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_correct_statement_l3334_333444


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l3334_333457

-- Define the doubling interval in seconds
def doubling_interval : ℕ := 30

-- Define the total time of the experiment in seconds
def total_time : ℕ := 4 * 60

-- Define the final number of bacteria
def final_bacteria : ℕ := 262144

-- Define the function to calculate the number of bacteria after a given time
def bacteria_count (initial : ℕ) (time : ℕ) : ℕ :=
  initial * (2 ^ (time / doubling_interval))

-- Theorem statement
theorem initial_bacteria_count :
  ∃ initial : ℕ, bacteria_count initial total_time = final_bacteria ∧ initial = 1024 :=
sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l3334_333457


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_equal_intercepts_line_equation_l3334_333406

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → Prop

-- Define the intersection point of two lines
def intersection (l1 l2 : Line) : Point :=
  sorry

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  sorry

-- Define a line having equal intercepts on coordinate axes
def equal_intercepts (l : Line) : Prop :=
  sorry

-- Define the lines given in the problem
def line1 : Line := λ x y ↦ 2*x + 3*y - 9 = 0
def line2 : Line := λ x y ↦ 3*x - y - 8 = 0
def line3 : Line := λ x y ↦ 3*x + 4*y - 1 = 0

-- Part 1
theorem perpendicular_line_equation :
  ∀ l : Line,
  passes_through l (intersection line1 line2) →
  perpendicular l line3 →
  l = λ x y ↦ y = (4/3)*x - 3 :=
sorry

-- Part 2
theorem equal_intercepts_line_equation :
  ∀ l : Line,
  passes_through l (intersection line1 line2) →
  equal_intercepts l →
  (l = λ x y ↦ y = -x + 4) ∨ (l = λ x y ↦ y = (1/3)*x) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_equal_intercepts_line_equation_l3334_333406


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3334_333420

-- Define the worth of bananas in terms of oranges
def banana_orange_ratio : ℚ := 12 / (3/4 * 16)

-- Theorem statement
theorem banana_orange_equivalence : 
  banana_orange_ratio * (2/5 * 10 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3334_333420


namespace NUMINAMATH_CALUDE_bobby_sarah_fish_ratio_l3334_333488

/-- The number of fish in each person's aquarium and their relationships -/
structure FishCounts where
  billy : ℕ
  tony : ℕ
  sarah : ℕ
  bobby : ℕ
  billy_count : billy = 10
  tony_count : tony = 3 * billy
  sarah_count : sarah = tony + 5
  total_count : billy + tony + sarah + bobby = 145

/-- The ratio of fish in Bobby's aquarium to Sarah's aquarium -/
def fish_ratio (fc : FishCounts) : ℚ :=
  fc.bobby / fc.sarah

/-- Theorem stating that the ratio of fish in Bobby's aquarium to Sarah's aquarium is 2:1 -/
theorem bobby_sarah_fish_ratio (fc : FishCounts) : fish_ratio fc = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bobby_sarah_fish_ratio_l3334_333488


namespace NUMINAMATH_CALUDE_area_difference_l3334_333468

-- Define the perimeter of the square playground
def square_perimeter : ℝ := 36

-- Define the perimeter of the rectangular basketball court
def rect_perimeter : ℝ := 38

-- Define the width of the rectangular basketball court
def rect_width : ℝ := 15

-- Theorem statement
theorem area_difference :
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rect_length := (rect_perimeter - 2 * rect_width) / 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 21 := by sorry

end NUMINAMATH_CALUDE_area_difference_l3334_333468


namespace NUMINAMATH_CALUDE_total_marbles_l3334_333477

/-- Given a bag of marbles with red, blue, green, and yellow marbles in the ratio 3:4:2:5,
    and 30 yellow marbles, prove that the total number of marbles is 84. -/
theorem total_marbles (red blue green yellow total : ℕ) 
  (h_ratio : red + blue + green + yellow = total)
  (h_proportion : 3 * yellow = 5 * red ∧ 4 * yellow = 5 * blue ∧ 2 * yellow = 5 * green)
  (h_yellow : yellow = 30) : total = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l3334_333477


namespace NUMINAMATH_CALUDE_leonardo_initial_money_l3334_333414

/-- The amount of money Leonardo had initially in his pocket -/
def initial_money : ℚ := 441 / 100

/-- The cost of the chocolate in dollars -/
def chocolate_cost : ℚ := 5

/-- The amount Leonardo borrowed from his friend in dollars -/
def borrowed_amount : ℚ := 59 / 100

/-- The additional amount Leonardo needs in dollars -/
def additional_needed : ℚ := 41 / 100

theorem leonardo_initial_money :
  chocolate_cost = initial_money + borrowed_amount + additional_needed :=
by sorry

end NUMINAMATH_CALUDE_leonardo_initial_money_l3334_333414


namespace NUMINAMATH_CALUDE_total_meat_theorem_l3334_333486

/-- The amount of beef needed for one beef hamburger -/
def beef_per_hamburger : ℚ := 4 / 10

/-- The amount of chicken needed for one chicken hamburger -/
def chicken_per_hamburger : ℚ := 2.5 / 5

/-- The number of beef hamburgers to be made -/
def beef_hamburgers : ℕ := 30

/-- The number of chicken hamburgers to be made -/
def chicken_hamburgers : ℕ := 15

/-- The total amount of meat needed for the given number of beef and chicken hamburgers -/
def total_meat_needed : ℚ := beef_per_hamburger * beef_hamburgers + chicken_per_hamburger * chicken_hamburgers

theorem total_meat_theorem : total_meat_needed = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_total_meat_theorem_l3334_333486


namespace NUMINAMATH_CALUDE_matrix_power_2023_l3334_333495

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l3334_333495


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3334_333497

theorem divisibility_implies_equality (a b : ℕ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) →
  a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3334_333497


namespace NUMINAMATH_CALUDE_barkley_buried_bones_l3334_333417

/-- Proves that Barkley has buried 42 bones after 5 months -/
theorem barkley_buried_bones 
  (bones_per_month : ℕ) 
  (months_passed : ℕ) 
  (available_bones : ℕ) 
  (h1 : bones_per_month = 10)
  (h2 : months_passed = 5)
  (h3 : available_bones = 8) :
  bones_per_month * months_passed - available_bones = 42 := by
  sorry

end NUMINAMATH_CALUDE_barkley_buried_bones_l3334_333417


namespace NUMINAMATH_CALUDE_factorization_3x_squared_minus_12_l3334_333472

theorem factorization_3x_squared_minus_12 (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x_squared_minus_12_l3334_333472


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l3334_333409

/-- The number of ways to choose a lineup from a basketball team --/
def choose_lineup (team_size : ℕ) (lineup_size : ℕ) : ℕ :=
  (team_size - lineup_size + 1).factorial / (team_size - lineup_size).factorial

/-- Theorem: The number of ways to choose a lineup of 6 players from a team of 15 is 3,603,600 --/
theorem basketball_lineup_count :
  choose_lineup 15 6 = 3603600 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l3334_333409


namespace NUMINAMATH_CALUDE_folded_strip_fits_l3334_333452

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rectangular strip
structure RectangularStrip where
  width : ℝ
  length : ℝ

-- Define a folded strip
structure FoldedStrip where
  original : RectangularStrip
  fold_line : ℝ × ℝ → ℝ × ℝ → Prop

-- Define the property of fitting inside a circle
def fits_inside (s : RectangularStrip) (c : Circle) : Prop := sorry

-- Define the property of a folded strip fitting inside a circle
def folded_fits_inside (fs : FoldedStrip) (c : Circle) : Prop := sorry

-- Theorem statement
theorem folded_strip_fits (c : Circle) (s : RectangularStrip) (fs : FoldedStrip) :
  fits_inside s c → fs.original = s → folded_fits_inside fs c := by sorry

end NUMINAMATH_CALUDE_folded_strip_fits_l3334_333452


namespace NUMINAMATH_CALUDE_symmetric_function_axis_l3334_333421

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := 1

-- State the theorem
theorem symmetric_function_axis (x : ℝ) : 
  f x = f (2 - x) → 
  f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
sorry

end NUMINAMATH_CALUDE_symmetric_function_axis_l3334_333421


namespace NUMINAMATH_CALUDE_triangle_shape_l3334_333476

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a * cos(A) = b * cos(B), then the triangle is either isosceles or right-angled. -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ A = B) ∨ A + B = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l3334_333476


namespace NUMINAMATH_CALUDE_unique_gcd_triplet_l3334_333442

-- Define the sets of possible values for x, y, and z
def X : Set ℕ := {6, 8, 12, 18, 24}
def Y : Set ℕ := {14, 20, 28, 44, 56}
def Z : Set ℕ := {5, 15, 18, 27, 42}

-- Define the theorem
theorem unique_gcd_triplet :
  ∃! (a b c x y z : ℕ),
    x ∈ X ∧ y ∈ Y ∧ z ∈ Z ∧
    x = Nat.gcd a b ∧
    y = Nat.gcd b c ∧
    z = Nat.gcd c a ∧
    x = 8 ∧ y = 14 ∧ z = 18 :=
by
  sorry

#check unique_gcd_triplet

end NUMINAMATH_CALUDE_unique_gcd_triplet_l3334_333442


namespace NUMINAMATH_CALUDE_gas_experiment_values_l3334_333443

/-- Represents the state of a gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents the change in gas state -/
structure GasStateChange where
  Δp : ℝ
  ΔV : ℝ

/-- Theorem stating the values of a₁ and a₂ for the given gas experiments -/
theorem gas_experiment_values (initialState : GasState) 
  (h_volume : initialState.volume = 1)
  (h_pressure : initialState.pressure = 10^5)
  (h_temperature : initialState.temperature = 300)
  (experiment1 : GasStateChange → Bool)
  (experiment2 : GasStateChange → Bool)
  (h_exp1 : ∀ change, experiment1 change ↔ change.Δp / change.ΔV = -10^5)
  (h_exp2 : ∀ change, experiment2 change ↔ change.Δp / change.ΔV = -1.4 * 10^5)
  (h_cooling1 : ∀ change, experiment1 change → 
    (change.ΔV > 0 → initialState.temperature > initialState.temperature + change.ΔV) ∧
    (change.ΔV < 0 → initialState.temperature > initialState.temperature - change.ΔV))
  (h_heating2 : ∀ change, experiment2 change → 
    (change.ΔV > 0 → initialState.temperature < initialState.temperature + change.ΔV) ∧
    (change.ΔV < 0 → initialState.temperature < initialState.temperature - change.ΔV)) :
  ∃ (a₁ a₂ : ℝ), a₁ = -10^5 ∧ a₂ = -1.4 * 10^5 := by
  sorry


end NUMINAMATH_CALUDE_gas_experiment_values_l3334_333443


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3334_333484

theorem point_in_fourth_quadrant :
  ∀ x : ℝ, (x^2 + 2 > 0) ∧ (-3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3334_333484


namespace NUMINAMATH_CALUDE_blanket_collection_l3334_333475

theorem blanket_collection (team_size : ℕ) (first_day_per_person : ℕ) (second_day_multiplier : ℕ) (third_day_fixed : ℕ) :
  team_size = 15 →
  first_day_per_person = 2 →
  second_day_multiplier = 3 →
  third_day_fixed = 22 →
  (team_size * first_day_per_person) + 
  (team_size * first_day_per_person * second_day_multiplier) + 
  third_day_fixed = 142 := by
sorry

end NUMINAMATH_CALUDE_blanket_collection_l3334_333475


namespace NUMINAMATH_CALUDE_f_continuous_iff_b_eq_zero_l3334_333433

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x + 4 else 3 * x + b

-- State the theorem
theorem f_continuous_iff_b_eq_zero (b : ℝ) :
  Continuous (f b) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_continuous_iff_b_eq_zero_l3334_333433


namespace NUMINAMATH_CALUDE_factorial_expression_is_perfect_square_l3334_333461

theorem factorial_expression_is_perfect_square (n : ℕ) (h : n ≥ 10) :
  (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial (n + 1) = (n + 2) ^ 2 := by
  sorry

#check factorial_expression_is_perfect_square

end NUMINAMATH_CALUDE_factorial_expression_is_perfect_square_l3334_333461


namespace NUMINAMATH_CALUDE_real_roots_condition_zero_sum_of_squares_l3334_333429

-- Statement 1
theorem real_roots_condition (q : ℝ) :
  q < 1 → ∃ x : ℝ, x^2 + 2*x + q = 0 :=
sorry

-- Statement 2
theorem zero_sum_of_squares (x y : ℝ) :
  x^2 + y^2 = 0 → x = 0 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_zero_sum_of_squares_l3334_333429


namespace NUMINAMATH_CALUDE_inequality_range_l3334_333402

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3334_333402


namespace NUMINAMATH_CALUDE_min_champion_wins_l3334_333445

theorem min_champion_wins (n : ℕ) (h : n = 10 ∨ n = 11) :
  let min_wins := (n / 2 : ℚ).ceil.toNat + 1
  ∀ k : ℕ, (∀ i : ℕ, i < n → i ≠ k → (n - 1).choose 2 ≤ k + i * (k - 1)) →
    min_wins ≤ k := by
  sorry

end NUMINAMATH_CALUDE_min_champion_wins_l3334_333445


namespace NUMINAMATH_CALUDE_absolute_value_non_negative_l3334_333425

theorem absolute_value_non_negative (x : ℝ) : 0 ≤ |x| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_non_negative_l3334_333425


namespace NUMINAMATH_CALUDE_expression_simplification_l3334_333422

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (3 * a / (a^2 - 4)) * (1 - 2 / a) - 4 / (a + 2) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3334_333422


namespace NUMINAMATH_CALUDE_blue_glass_ball_probability_l3334_333480

/-- The probability of drawing a blue glass ball given that a glass ball is drawn -/
theorem blue_glass_ball_probability :
  let total_balls : ℕ := 5 + 11
  let red_glass_balls : ℕ := 2
  let blue_glass_balls : ℕ := 4
  let total_glass_balls : ℕ := red_glass_balls + blue_glass_balls
  (blue_glass_balls : ℚ) / total_glass_balls = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_blue_glass_ball_probability_l3334_333480


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l3334_333466

/-- The smallest positive integer with only two positive divisors -/
def smallest_two_divisors : ℕ := 2

/-- The largest integer less than 150 with exactly three positive divisors -/
def largest_three_divisors_under_150 : ℕ := 121

/-- The theorem stating that the sum of the two defined numbers is 123 -/
theorem sum_of_special_integers : 
  smallest_two_divisors + largest_three_divisors_under_150 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l3334_333466


namespace NUMINAMATH_CALUDE_charlie_same_color_probability_l3334_333496

def total_marbles : ℕ := 10
def red_marbles : ℕ := 3
def green_marbles : ℕ := 3
def blue_marbles : ℕ := 4

def alice_draw : ℕ := 3
def bob_draw : ℕ := 3
def charlie_draw : ℕ := 4

theorem charlie_same_color_probability :
  let total_outcomes := (total_marbles.choose alice_draw) * ((total_marbles - alice_draw).choose bob_draw) * ((total_marbles - alice_draw - bob_draw).choose charlie_draw)
  let favorable_outcomes := 
    2 * (red_marbles.min green_marbles).choose 3 * (total_marbles - red_marbles - green_marbles).choose 1 +
    (blue_marbles.choose 3) * (total_marbles - blue_marbles).choose 1 +
    blue_marbles.choose 4
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 1400 := by
  sorry

end NUMINAMATH_CALUDE_charlie_same_color_probability_l3334_333496


namespace NUMINAMATH_CALUDE_point_on_line_m_value_l3334_333479

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

theorem point_on_line_m_value :
  ∀ m : ℝ,
  let P : Point := ⟨3, m⟩
  let M : Point := ⟨2, -1⟩
  let N : Point := ⟨-3, 4⟩
  collinear P M N → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_m_value_l3334_333479


namespace NUMINAMATH_CALUDE_harmonic_quadrilateral_properties_l3334_333438

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a quadrilateral as a collection of four points
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of a harmonic quadrilateral
def is_harmonic (q : Quadrilateral) : Prop :=
  ∃ (AB CD AC BD AD BC : ℝ),
    AB * CD = AC * BD ∧ AB * CD = AD * BC

-- Define the concyclic property for four points
def are_concyclic (A B C D : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (A.x - center.x)^2 + (A.y - center.y)^2 = radius^2 ∧
    (B.x - center.x)^2 + (B.y - center.y)^2 = radius^2 ∧
    (C.x - center.x)^2 + (C.y - center.y)^2 = radius^2 ∧
    (D.x - center.x)^2 + (D.y - center.y)^2 = radius^2

-- State the theorem
theorem harmonic_quadrilateral_properties
  (ABCD : Quadrilateral)
  (A1 B1 C1 D1 : Point)
  (h1 : is_harmonic ABCD)
  (h2 : is_harmonic ⟨A1, ABCD.B, ABCD.C, ABCD.D⟩)
  (h3 : is_harmonic ⟨ABCD.A, B1, ABCD.C, ABCD.D⟩)
  (h4 : is_harmonic ⟨ABCD.A, ABCD.B, C1, ABCD.D⟩)
  (h5 : is_harmonic ⟨ABCD.A, ABCD.B, ABCD.C, D1⟩) :
  are_concyclic ABCD.A ABCD.B C1 D1 ∧ is_harmonic ⟨A1, B1, C1, D1⟩ :=
sorry

end NUMINAMATH_CALUDE_harmonic_quadrilateral_properties_l3334_333438


namespace NUMINAMATH_CALUDE_stock_certificate_tearing_l3334_333485

theorem stock_certificate_tearing : ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := by
  sorry

end NUMINAMATH_CALUDE_stock_certificate_tearing_l3334_333485


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3334_333430

theorem sum_of_four_numbers : 2143 + 3412 + 4213 + 1324 = 11092 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3334_333430


namespace NUMINAMATH_CALUDE_quadratic_even_iff_m_eq_neg_two_l3334_333474

/-- A quadratic function f(x) = mx^2 + (m+2)mx + 2 is even if and only if m = -2 -/
theorem quadratic_even_iff_m_eq_neg_two (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (m + 2) * m * x + 2 = m * (-x)^2 + (m + 2) * m * (-x) + 2) ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_m_eq_neg_two_l3334_333474


namespace NUMINAMATH_CALUDE_total_vitamins_in_box_vitamins_in_half_bag_l3334_333451

-- Define the number of bags in a box
def bags_per_box : ℕ := 9

-- Define the grams of vitamins per bag
def vitamins_per_bag : ℝ := 0.2

-- Theorem for total vitamins in a box
theorem total_vitamins_in_box :
  bags_per_box * vitamins_per_bag = 1.8 := by sorry

-- Theorem for vitamins in half a bag
theorem vitamins_in_half_bag :
  vitamins_per_bag / 2 = 0.1 := by sorry

end NUMINAMATH_CALUDE_total_vitamins_in_box_vitamins_in_half_bag_l3334_333451


namespace NUMINAMATH_CALUDE_correct_travel_times_l3334_333427

/-- Represents the travel times of Winnie-the-Pooh and Piglet -/
structure TravelTimes where
  pooh : ℝ
  piglet : ℝ

/-- Calculates the travel times based on the given conditions -/
def calculate_travel_times (time_after_meeting_pooh : ℝ) (time_after_meeting_piglet : ℝ) : TravelTimes :=
  let speed_ratio := time_after_meeting_pooh / time_after_meeting_piglet
  let time_before_meeting := time_after_meeting_piglet
  { pooh := time_before_meeting + time_after_meeting_piglet
  , piglet := time_before_meeting + time_after_meeting_pooh }

/-- Theorem stating that the calculated travel times are correct -/
theorem correct_travel_times :
  let result := calculate_travel_times 4 1
  result.pooh = 2 ∧ result.piglet = 6 := by sorry

end NUMINAMATH_CALUDE_correct_travel_times_l3334_333427


namespace NUMINAMATH_CALUDE_grid_31_counts_l3334_333493

/-- Represents a grid with n horizontal and vertical lines -/
structure Grid (n : ℕ) where
  horizontal_lines : ℕ
  vertical_lines : ℕ
  h_lines : horizontal_lines = n
  v_lines : vertical_lines = n

/-- Counts the number of rectangles in a grid -/
def count_rectangles (g : Grid n) : ℕ :=
  (n.choose 2) * (n.choose 2)

/-- Counts the number of squares in a grid with 1:2 distance ratio -/
def count_squares (g : Grid n) : ℕ :=
  let S (k : ℕ) := k * (k + 1) * (2 * k + 1) / 6
  S n - 2 * S (n / 2)

/-- The main theorem about the 31x31 grid -/
theorem grid_31_counts :
  ∃ (g : Grid 31),
    count_rectangles g = 216225 ∧
    count_squares g = 6975 :=
by sorry

end NUMINAMATH_CALUDE_grid_31_counts_l3334_333493


namespace NUMINAMATH_CALUDE_flag_design_count_l3334_333467

/-- The number of colors available for the flag design -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating the number of possible flag designs -/
theorem flag_design_count : num_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l3334_333467


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3334_333456

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the line passing through the focus
def line_through_focus (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p = (focus.1 + t, focus.2 + t)

-- Theorem statement
theorem parabola_intersection_length 
  (A B : PointOnParabola) 
  (h_line_A : line_through_focus (A.x, A.y))
  (h_line_B : line_through_focus (B.x, B.y))
  (h_sum : A.x + B.x = 6) :
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3334_333456


namespace NUMINAMATH_CALUDE_uncle_gift_amount_l3334_333405

/-- The amount of money Geoffrey's uncle gave him --/
def uncle_gift (grandmother_gift aunt_gift total_after_gifts spent_on_games money_left : ℕ) : ℕ :=
  total_after_gifts - grandmother_gift - aunt_gift - money_left

/-- Theorem stating the amount of money Geoffrey's uncle gave him --/
theorem uncle_gift_amount : 
  uncle_gift 20 25 125 105 20 = 60 := by
  sorry

#eval uncle_gift 20 25 125 105 20

end NUMINAMATH_CALUDE_uncle_gift_amount_l3334_333405


namespace NUMINAMATH_CALUDE_not_T_function_id_T_function_exp_T_function_cos_iff_l3334_333435

/-- Definition of T-function -/
def is_T_function (f : ℝ → ℝ) : Prop :=
  ∃ (T : ℝ) (hT : T ≠ 0), ∀ x : ℝ, f (x + T) = T * f x

/-- Statement 1: f(x) = x is not a T-function -/
theorem not_T_function_id :
  ¬ is_T_function id := sorry

/-- Statement 2: g(x) = aˣ is a T-function when 0 < a < 1 -/
theorem T_function_exp (a : ℝ) (ha : 0 < a) (ha' : a < 1) :
  is_T_function (λ x => a^x) := sorry

/-- Statement 3: h(x) = cos(mx) is a T-function iff m = kπ for some k ∈ ℤ -/
theorem T_function_cos_iff (m : ℝ) :
  is_T_function (λ x => Real.cos (m * x)) ↔ ∃ k : ℤ, m = k * Real.pi := sorry

end NUMINAMATH_CALUDE_not_T_function_id_T_function_exp_T_function_cos_iff_l3334_333435


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3334_333426

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 3x + 5 and y = (5k)x + 7 are parallel -/
theorem parallel_lines_k_value : 
  (∀ x y : ℝ, y = 3 * x + 5 ↔ y = (5 * k) * x + 7) → k = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3334_333426


namespace NUMINAMATH_CALUDE_gym_cost_theorem_l3334_333432

/-- Calculates the total cost of two gym memberships for one year -/
def total_gym_cost (cheap_monthly : ℕ) (cheap_signup : ℕ) (months : ℕ) : ℕ :=
  let expensive_monthly := 3 * cheap_monthly
  let cheap_total := cheap_monthly * months + cheap_signup
  let expensive_total := expensive_monthly * months + (expensive_monthly * 4)
  cheap_total + expensive_total

/-- Theorem stating that the total cost for two gym memberships for one year is $650 -/
theorem gym_cost_theorem : total_gym_cost 10 50 12 = 650 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_theorem_l3334_333432


namespace NUMINAMATH_CALUDE_special_triangle_f_measure_l3334_333458

/-- A triangle with two equal angles and the third angle 20 degrees less than the others. -/
structure SpecialTriangle where
  /-- Angle D in degrees -/
  angleD : ℝ
  /-- Angle E in degrees -/
  angleE : ℝ
  /-- Angle F in degrees -/
  angleF : ℝ
  /-- Sum of angles in the triangle is 180 degrees -/
  angle_sum : angleD + angleE + angleF = 180
  /-- Angles D and E are equal -/
  d_eq_e : angleD = angleE
  /-- Angle F is 20 degrees less than angle D -/
  f_less_20 : angleF = angleD - 20

theorem special_triangle_f_measure (t : SpecialTriangle) : t.angleF = 40 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_f_measure_l3334_333458


namespace NUMINAMATH_CALUDE_average_marks_of_all_students_l3334_333494

theorem average_marks_of_all_students
  (batch1_size : ℕ) (batch2_size : ℕ) (batch3_size : ℕ)
  (batch1_avg : ℝ) (batch2_avg : ℝ) (batch3_avg : ℝ)
  (h1 : batch1_size = 40)
  (h2 : batch2_size = 50)
  (h3 : batch3_size = 60)
  (h4 : batch1_avg = 45)
  (h5 : batch2_avg = 55)
  (h6 : batch3_avg = 65) :
  let total_students := batch1_size + batch2_size + batch3_size
  let total_marks := batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg
  total_marks / total_students = 56.33 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_all_students_l3334_333494


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3334_333491

/-- The general form equation of a line perpendicular to 2x+y-5=0 and passing through (1,2) is x-2y+3=0 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (∃ (m b : ℝ), y = m * x + b ∧ m * 2 = -1) →  -- perpendicular line condition
  (1 : ℝ) - 2 * (2 : ℝ) + 3 = 0 →              -- point (1,2) satisfies the equation
  x - 2 * y + 3 = 0                            -- the equation we want to prove
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3334_333491


namespace NUMINAMATH_CALUDE_annes_cats_weight_l3334_333418

/-- Given Anne's cats' weights, prove the total weight she carries -/
theorem annes_cats_weight (female_weight : ℝ) (male_weight_ratio : ℝ) : 
  female_weight = 2 → 
  male_weight_ratio = 2 → 
  female_weight + female_weight * male_weight_ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_annes_cats_weight_l3334_333418


namespace NUMINAMATH_CALUDE_circle_area_decrease_l3334_333489

theorem circle_area_decrease (r : ℝ) (h : r > 0) :
  let new_r := r / 2
  let original_area := π * r^2
  let new_area := π * new_r^2
  (original_area - new_area) / original_area = 3/4 := by sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l3334_333489


namespace NUMINAMATH_CALUDE_cats_theorem_l3334_333453

def cats_problem (siamese house persian first_sale second_sale : ℕ) : Prop :=
  let initial_total : ℕ := siamese + house + persian
  let after_first_sale : ℕ := initial_total - first_sale
  let final_count : ℕ := after_first_sale - second_sale
  final_count = 17

theorem cats_theorem : cats_problem 23 17 29 40 12 := by
  sorry

end NUMINAMATH_CALUDE_cats_theorem_l3334_333453


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3334_333434

theorem sufficient_not_necessary (x : ℝ) :
  (x = 1 → x^2 - 3*x + 2 = 0) ∧
  ¬(x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3334_333434


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l3334_333440

/-- Quadratic function passing through specific points with given minimum --/
theorem quadratic_function_proof (a h k : ℝ) :
  a ≠ 0 →
  a * (1 - h)^2 + k = 3 →
  a * (3 - h)^2 + k = 3 →
  k = -1 →
  a = 4 ∧ h = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l3334_333440


namespace NUMINAMATH_CALUDE_cost_of_jeans_l3334_333460

/-- The cost of a pair of jeans -/
def cost_jeans : ℝ := sorry

/-- The cost of a shirt -/
def cost_shirt : ℝ := sorry

/-- The first condition: 3 pairs of jeans and 6 shirts cost $104.25 -/
axiom condition1 : 3 * cost_jeans + 6 * cost_shirt = 104.25

/-- The second condition: 4 pairs of jeans and 5 shirts cost $112.15 -/
axiom condition2 : 4 * cost_jeans + 5 * cost_shirt = 112.15

/-- Theorem stating that the cost of each pair of jeans is $16.85 -/
theorem cost_of_jeans : cost_jeans = 16.85 := by sorry

end NUMINAMATH_CALUDE_cost_of_jeans_l3334_333460


namespace NUMINAMATH_CALUDE_polynomial_inequality_roots_l3334_333407

theorem polynomial_inequality_roots (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c*x - 8 < 0 ↔ x < 2 ∨ x > 6) → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_roots_l3334_333407


namespace NUMINAMATH_CALUDE_distance_to_fountain_is_30_l3334_333439

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def distance_to_fountain : ℕ := sorry

/-- The total distance Mrs. Hilt walks for all trips to the fountain -/
def total_distance : ℕ := 120

/-- The number of times Mrs. Hilt goes to the water fountain -/
def number_of_trips : ℕ := 4

/-- Theorem stating that the distance to the fountain is 30 feet -/
theorem distance_to_fountain_is_30 : 
  distance_to_fountain = total_distance / number_of_trips :=
sorry

end NUMINAMATH_CALUDE_distance_to_fountain_is_30_l3334_333439


namespace NUMINAMATH_CALUDE_A_inter_B_l3334_333498

def A : Set ℝ := {-2, -1, 0, 1, 2}

def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3}

theorem A_inter_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_l3334_333498


namespace NUMINAMATH_CALUDE_remainder_x_power_10_minus_1_div_x_plus_1_l3334_333490

theorem remainder_x_power_10_minus_1_div_x_plus_1 (x : ℝ) : 
  (x^10 - 1) % (x + 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_x_power_10_minus_1_div_x_plus_1_l3334_333490


namespace NUMINAMATH_CALUDE_complex_number_equality_l3334_333492

theorem complex_number_equality (Z : ℂ) (h : Z * (1 - Complex.I) = 3 - Complex.I) : 
  Z = 2 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3334_333492


namespace NUMINAMATH_CALUDE_train_speed_problem_l3334_333413

theorem train_speed_problem (total_distance : ℝ) (speed_increase : ℝ) (distance_difference : ℝ) (time_difference : ℝ) :
  total_distance = 103 ∧ 
  speed_increase = 4 ∧ 
  distance_difference = 23 ∧ 
  time_difference = 1/4 →
  ∃ (initial_speed : ℝ) (initial_time : ℝ),
    initial_speed = 80 ∧
    initial_speed * initial_time + (initial_speed * initial_time + distance_difference) = total_distance ∧
    (initial_speed + speed_increase) * (initial_time + time_difference) = initial_speed * initial_time + distance_difference :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3334_333413


namespace NUMINAMATH_CALUDE_expression_simplification_l3334_333423

theorem expression_simplification (a b : ℚ) (h1 : a = -2) (h2 : b = 2/3) :
  3 * (2 * a^2 - 3 * a * b - 5 * a - 1) - 6 * (a^2 - a * b + 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3334_333423


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l3334_333499

theorem largest_prime_factor_of_1729 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1729 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l3334_333499


namespace NUMINAMATH_CALUDE_circle_center_sum_l3334_333412

/-- Given a circle with equation x^2 + y^2 - 6x + 8y - 24 = 0, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 24 : ℝ)) →
  h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3334_333412


namespace NUMINAMATH_CALUDE_clara_dina_age_difference_l3334_333473

theorem clara_dina_age_difference : ∃! n : ℕ+, ∃ C D : ℕ+,
  C = D + n ∧
  C - 1 = 3 * (D - 1) ∧
  C = D^3 + 1 ∧
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_clara_dina_age_difference_l3334_333473


namespace NUMINAMATH_CALUDE_choose_four_from_multiset_l3334_333424

/-- Represents a multiset of letters -/
def LetterMultiset : Type := List Char

/-- The specific multiset of letters in our problem -/
def problemMultiset : LetterMultiset := ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

/-- Counts the number of ways to choose k elements from a multiset -/
def countChoices (ms : LetterMultiset) (k : Nat) : Nat :=
  sorry -- Implementation not required for the statement

/-- The main theorem stating that there are 175 ways to choose 4 letters from the given multiset -/
theorem choose_four_from_multiset :
  countChoices problemMultiset 4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_multiset_l3334_333424


namespace NUMINAMATH_CALUDE_problem_solution_l3334_333419

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 4) (h3 : z^2 / x = 8) :
  x = 2^(11/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3334_333419


namespace NUMINAMATH_CALUDE_speedster_convertibles_l3334_333446

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  4 * speedsters = 3 * total →
  5 * convertibles = 3 * speedsters →
  total - speedsters = 30 →
  convertibles = 54 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l3334_333446


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_iff_a_positive_l3334_333465

/-- A complex number represented by its real and imaginary parts -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- The third quadrant of the complex plane -/
def ThirdQuadrant (z : ComplexNumber) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number z = (5-ai)/i for a given real number a -/
def z (a : ℝ) : ComplexNumber :=
  { re := -a, im := -5 }

/-- The main theorem: z(a) is in the third quadrant if and only if a > 0 -/
theorem z_in_third_quadrant_iff_a_positive (a : ℝ) :
  ThirdQuadrant (z a) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_iff_a_positive_l3334_333465


namespace NUMINAMATH_CALUDE_cricket_average_theorem_l3334_333478

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  matches_played : ℕ
  total_runs : ℕ
  
/-- Calculates the batting average -/
def batting_average (stats : CricketStats) : ℚ :=
  stats.total_runs / stats.matches_played

/-- Theorem: If a player has played 5 matches and scoring 69 runs in the next match
    would bring their batting average to 54, then their current batting average is 51 -/
theorem cricket_average_theorem (stats : CricketStats) 
    (h1 : stats.matches_played = 5)
    (h2 : batting_average ⟨stats.matches_played + 1, stats.total_runs + 69⟩ = 54) :
  batting_average stats = 51 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_theorem_l3334_333478
