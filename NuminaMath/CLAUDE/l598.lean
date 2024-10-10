import Mathlib

namespace smallest_non_factor_product_of_48_l598_59822

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product_of_48 (x y : ℕ) :
  x ≠ y →
  x > 0 →
  y > 0 →
  is_factor x 48 →
  is_factor y 48 →
  ¬ is_factor (x * y) 48 →
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → is_factor a 48 → is_factor b 48 → ¬ is_factor (a * b) 48 → x * y ≤ a * b →
  x * y = 18 :=
sorry

end smallest_non_factor_product_of_48_l598_59822


namespace least_n_factorial_divisible_by_9450_l598_59823

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem least_n_factorial_divisible_by_9450 :
  ∃ (n : ℕ), n > 0 ∧ is_factor 9450 (Nat.factorial n) ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬is_factor 9450 (Nat.factorial m) :=
by
  use 10
  sorry

end least_n_factorial_divisible_by_9450_l598_59823


namespace additive_function_characterization_l598_59800

def is_additive (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

theorem additive_function_characterization (f : ℚ → ℚ) (h : is_additive f) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := by
  sorry

end additive_function_characterization_l598_59800


namespace gcd_power_two_l598_59845

theorem gcd_power_two : 
  Nat.gcd (2^2100 - 1) (2^2091 + 31) = Nat.gcd (2^2091 + 31) 511 := by
  sorry

end gcd_power_two_l598_59845


namespace platform_length_calculation_l598_59876

/-- Calculates the length of a platform given train characteristics and crossing time -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmph = 84 →
  crossing_time = 16 →
  ∃ (platform_length : ℝ), abs (platform_length - 233.33) < 0.01 := by
  sorry

#check platform_length_calculation

end platform_length_calculation_l598_59876


namespace girls_average_weight_is_27_l598_59834

/-- Given a class with boys and girls, calculates the average weight of girls -/
def average_weight_of_girls (total_students : ℕ) (num_boys : ℕ) (boys_avg_weight : ℚ) (class_avg_weight : ℚ) : ℚ :=
  let total_weight := class_avg_weight * total_students
  let boys_total_weight := boys_avg_weight * num_boys
  let girls_total_weight := total_weight - boys_total_weight
  let num_girls := total_students - num_boys
  girls_total_weight / num_girls

/-- Theorem stating that the average weight of girls is 27 kgs given the problem conditions -/
theorem girls_average_weight_is_27 : 
  average_weight_of_girls 25 15 48 45 = 27 := by
  sorry

end girls_average_weight_is_27_l598_59834


namespace exists_invariant_point_l598_59815

/-- A set of non-constant functions with specific properties -/
def FunctionSet (G : Set (ℝ → ℝ)) : Prop :=
  ∀ f ∈ G, ∃ a b : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧
  (∀ g ∈ G, (g ∘ f) ∈ G) ∧
  (Function.Bijective f → Function.invFun f ∈ G) ∧
  (∃ xₑ : ℝ, f xₑ = xₑ)

/-- The main theorem -/
theorem exists_invariant_point {G : Set (ℝ → ℝ)} (hG : FunctionSet G) :
  ∃ k : ℝ, ∀ f ∈ G, f k = k := by sorry

end exists_invariant_point_l598_59815


namespace problem_solution_l598_59860

theorem problem_solution : (42 / (9 - 3 * 2)) * 4 = 56 := by
  sorry

end problem_solution_l598_59860


namespace equation_solution_l598_59831

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = (3/2 : ℝ) ∧ 
  (∀ x : ℝ, 2 * (x - 1)^2 = x - 1 ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end equation_solution_l598_59831


namespace initial_games_eq_sum_l598_59895

/-- Represents the number of video games Cody had initially -/
def initial_games : ℕ := 9

/-- Represents the number of video games Cody gave away -/
def games_given_away : ℕ := 4

/-- Represents the number of video games Cody still has -/
def games_remaining : ℕ := 5

/-- Theorem stating that the initial number of games equals the sum of games given away and games remaining -/
theorem initial_games_eq_sum : initial_games = games_given_away + games_remaining := by
  sorry

end initial_games_eq_sum_l598_59895


namespace perpendicular_distance_approx_l598_59832

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  d : Point3D
  a : Point3D
  b : Point3D
  c : Point3D

/-- Calculates the perpendicular distance from a point to a plane defined by three points -/
def perpendicularDistance (p : Point3D) (a b c : Point3D) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem perpendicular_distance_approx (p : Parallelepiped) : 
  p.length = 5 ∧ p.width = 3 ∧ p.height = 2 ∧
  p.d = ⟨0, 0, 0⟩ ∧ p.a = ⟨5, 0, 0⟩ ∧ p.b = ⟨0, 3, 0⟩ ∧ p.c = ⟨0, 0, 2⟩ →
  abs (perpendicularDistance p.d p.a p.b p.c - 1.9) < 0.1 := by
  sorry

end perpendicular_distance_approx_l598_59832


namespace max_t_for_exponential_sequence_range_a_for_quadratic_sequence_l598_59871

/-- Definition of property P(t) for a sequence -/
def has_property_P (a : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ m n : ℕ, m ≠ n → (a m - a n) / (m - n : ℝ) ≥ t

/-- Theorem for part (i) -/
theorem max_t_for_exponential_sequence :
  ∃ t_max : ℝ, (∀ t : ℝ, has_property_P (λ n => (2 : ℝ) ^ n) t → t ≤ t_max) ∧
            has_property_P (λ n => (2 : ℝ) ^ n) t_max :=
sorry

/-- Theorem for part (ii) -/
theorem range_a_for_quadratic_sequence :
  ∃ a_min : ℝ, (∀ a : ℝ, has_property_P (λ n => n^2 - a / n) 10 → a ≥ a_min) ∧
            has_property_P (λ n => n^2 - a_min / n) 10 :=
sorry

end max_t_for_exponential_sequence_range_a_for_quadratic_sequence_l598_59871


namespace sqrt_equation_solution_l598_59821

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 2 * x^2 + 8) = 12 ↔ x = 8 ∨ x = -17/2 := by sorry

end sqrt_equation_solution_l598_59821


namespace parabola_coefficient_l598_59827

/-- Theorem: For a parabola y = ax² + bx + c passing through points (4, 0), (t/3, 0), and (0, 60), the value of a is 45/t. -/
theorem parabola_coefficient (a b c t : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ (x = 4 ∨ x = t/3)) → 
  a * 0^2 + b * 0 + c = 60 →
  a = 45 / t :=
by sorry

end parabola_coefficient_l598_59827


namespace repeating_decimal_problem_l598_59813

theorem repeating_decimal_problem (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  72 * ((1 + (100 * a + 10 * b + c : ℕ) / 999 : ℚ) - (1 + (a / 10 + b / 100 + c / 1000 : ℚ))) = (3 / 5 : ℚ) →
  100 * a + 10 * b + c = 833 := by
sorry

end repeating_decimal_problem_l598_59813


namespace perpendicular_vectors_imply_k_zero_l598_59812

/-- Given vectors a, b, and c in ℝ², prove that if a-c is perpendicular to b, then k = 0 -/
theorem perpendicular_vectors_imply_k_zero (a b c : ℝ × ℝ) (h : a.1 = 3 ∧ a.2 = 1) 
  (h' : b.1 = 1 ∧ b.2 = 3) (h'' : c.1 = k ∧ c.2 = 2) 
  (h''' : (a.1 - c.1) * b.1 + (a.2 - c.2) * b.2 = 0) : 
  k = 0 := by
  sorry

end perpendicular_vectors_imply_k_zero_l598_59812


namespace expression_simplification_and_evaluation_l598_59887

theorem expression_simplification_and_evaluation :
  let x : ℝ := 6 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180)
  ((x / (x - 2) - x / (x + 2)) / (4 * x / (x - 2))) = 1 / 4 := by
  sorry

end expression_simplification_and_evaluation_l598_59887


namespace crazy_silly_school_series_remaining_books_l598_59882

/-- Given a series of books, calculate the number of books remaining to be read -/
def booksRemaining (totalBooks readBooks : ℕ) : ℕ :=
  totalBooks - readBooks

/-- Theorem: In a series of 32 books, if 17 have been read, 15 remain to be read -/
theorem crazy_silly_school_series_remaining_books :
  booksRemaining 32 17 = 15 := by
  sorry

end crazy_silly_school_series_remaining_books_l598_59882


namespace lunch_combinations_eq_27_l598_59897

/-- Represents a category of food items in the cafeteria -/
structure FoodCategory where
  options : Finset String
  size_eq_three : options.card = 3

/-- Represents the cafeteria menu -/
structure CafeteriaMenu where
  main_dishes : FoodCategory
  beverages : FoodCategory
  snacks : FoodCategory

/-- A function to calculate the number of distinct lunch combinations -/
def count_lunch_combinations (menu : CafeteriaMenu) : ℕ :=
  menu.main_dishes.options.card * menu.beverages.options.card * menu.snacks.options.card

/-- Theorem stating that the number of distinct lunch combinations is 27 -/
theorem lunch_combinations_eq_27 (menu : CafeteriaMenu) :
  count_lunch_combinations menu = 27 := by
  sorry

#check lunch_combinations_eq_27

end lunch_combinations_eq_27_l598_59897


namespace curve_is_hyperbola_with_foci_on_x_axis_l598_59880

/-- The curve represented by the equation x²/(sin θ + 3) + y²/(sin θ - 2) = 1 -/
def curve (x y θ : ℝ) : Prop :=
  x^2 / (Real.sin θ + 3) + y^2 / (Real.sin θ - 2) = 1

/-- The curve is a hyperbola with foci on the x-axis -/
theorem curve_is_hyperbola_with_foci_on_x_axis :
  ∀ x y θ, curve x y θ → 
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 - y^2/b^2 = 1) ∧ 
    (∃ c : ℝ, c > 0 ∧ (∃ f₁ f₂ : ℝ × ℝ, f₁.1 = c ∧ f₁.2 = 0 ∧ f₂.1 = -c ∧ f₂.2 = 0)) :=
by sorry

end curve_is_hyperbola_with_foci_on_x_axis_l598_59880


namespace intersection_A_B_l598_59841

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l598_59841


namespace complex_equation_solution_l598_59828

theorem complex_equation_solution (a : ℝ) : 
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end complex_equation_solution_l598_59828


namespace number_problem_l598_59870

theorem number_problem : ∃ x : ℚ, 34 + 3 * x = 49 ∧ x = 5 := by
  sorry

end number_problem_l598_59870


namespace find_2a_plus_b_l598_59873

-- Define the functions
def f (a b x : ℝ) : ℝ := 2 * a * x - 3 * b
def g (x : ℝ) : ℝ := 5 * x + 4
def h (a b x : ℝ) : ℝ := g (f a b x)

-- State the theorem
theorem find_2a_plus_b (a b : ℝ) :
  (∀ x, h a b (2 * x - 9) = x) →
  2 * a + b = 1 / 15 := by
  sorry

end find_2a_plus_b_l598_59873


namespace inequality_solution_set_l598_59886

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) → m < -13/11 := by
  sorry

end inequality_solution_set_l598_59886


namespace geometric_series_problem_l598_59809

theorem geometric_series_problem (a r : ℝ) (h1 : r ≠ 1) (h2 : r > 0) : 
  (a / (1 - r) = 15) → (a / (1 - r^4) = 9) → r = 1/3 := by
  sorry

end geometric_series_problem_l598_59809


namespace cos_sum_seventh_roots_l598_59816

theorem cos_sum_seventh_roots : Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (8 * Real.pi / 7) = -1/2 := by
  sorry

end cos_sum_seventh_roots_l598_59816


namespace triangle_heights_order_l598_59864

/-- Given a triangle with sides a, b, c and corresponding heights ha, hb, hc,
    if a > b > c, then ha < hb < hc -/
theorem triangle_heights_order (a b c ha hb hc : ℝ) :
  a > 0 → b > 0 → c > 0 →  -- positive sides
  ha > 0 → hb > 0 → hc > 0 →  -- positive heights
  a > b → b > c →  -- order of sides
  a * ha = b * hb →  -- area equality
  b * hb = c * hc →  -- area equality
  ha < hb ∧ hb < hc := by
  sorry


end triangle_heights_order_l598_59864


namespace large_bucket_capacity_l598_59847

theorem large_bucket_capacity (small_bucket : ℝ) (large_bucket : ℝ) : 
  (large_bucket = 2 * small_bucket + 3) →
  (2 * small_bucket + 5 * large_bucket = 63) →
  large_bucket = 11 := by
sorry

end large_bucket_capacity_l598_59847


namespace second_year_probability_l598_59826

/-- Represents the academic year of a student -/
inductive AcademicYear
| FirstYear
| SecondYear
| ThirdYear
| Postgraduate

/-- Represents the department of a student -/
inductive Department
| Science
| Arts
| Engineering

/-- Represents the number of students in each academic year and department -/
def studentCount : AcademicYear → Department → ℕ
| AcademicYear.FirstYear, Department.Science => 300
| AcademicYear.FirstYear, Department.Arts => 200
| AcademicYear.FirstYear, Department.Engineering => 100
| AcademicYear.SecondYear, Department.Science => 250
| AcademicYear.SecondYear, Department.Arts => 150
| AcademicYear.SecondYear, Department.Engineering => 50
| AcademicYear.ThirdYear, Department.Science => 300
| AcademicYear.ThirdYear, Department.Arts => 200
| AcademicYear.ThirdYear, Department.Engineering => 50
| AcademicYear.Postgraduate, Department.Science => 200
| AcademicYear.Postgraduate, Department.Arts => 100
| AcademicYear.Postgraduate, Department.Engineering => 100

/-- The total number of students in the sample -/
def totalStudents : ℕ := 2000

/-- Theorem: The probability of selecting a second-year student from the group of students
    who are not third-year and not in the Science department is 2/7 -/
theorem second_year_probability :
  let nonThirdYearNonScience := (studentCount AcademicYear.FirstYear Department.Arts
                                + studentCount AcademicYear.FirstYear Department.Engineering
                                + studentCount AcademicYear.SecondYear Department.Arts
                                + studentCount AcademicYear.SecondYear Department.Engineering
                                + studentCount AcademicYear.Postgraduate Department.Arts
                                + studentCount AcademicYear.Postgraduate Department.Engineering)
  let secondYearNonScience := (studentCount AcademicYear.SecondYear Department.Arts
                              + studentCount AcademicYear.SecondYear Department.Engineering)
  (secondYearNonScience : ℚ) / nonThirdYearNonScience = 2 / 7 := by
  sorry

end second_year_probability_l598_59826


namespace tamara_brownie_pans_l598_59853

def total_revenue : ℕ := 32
def brownie_price : ℕ := 2
def pieces_per_pan : ℕ := 8

theorem tamara_brownie_pans : 
  total_revenue / (brownie_price * pieces_per_pan) = 2 := by
  sorry

end tamara_brownie_pans_l598_59853


namespace exists_ratio_eq_rational_l598_59849

def u : ℕ → ℚ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then u (n / 2) + u ((n - 1) / 2) else u (n / 2)

theorem exists_ratio_eq_rational (k : ℚ) (hk : k > 0) :
  ∃ n : ℕ, u n / u (n + 1) = k :=
by sorry

end exists_ratio_eq_rational_l598_59849


namespace population_increase_l598_59825

theorem population_increase (a b c : ℝ) :
  let increase_0_to_1 := 1 + a / 100
  let increase_1_to_2 := 1 + b / 100
  let increase_2_to_3 := 1 + c / 100
  let total_increase := increase_0_to_1 * increase_1_to_2 * increase_2_to_3 - 1
  total_increase * 100 = a + b + c + (a * b + b * c + a * c) / 100 + a * b * c / 10000 :=
by sorry

end population_increase_l598_59825


namespace building_height_is_270_l598_59881

/-- Calculates the height of a building with specified floor heights -/
def building_height (total_stories : ℕ) (first_half_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := (total_stories / 2) * first_half_height
  let second_half := (total_stories / 2) * (first_half_height + height_increase)
  first_half + second_half

/-- Proves that the height of the specified building is 270 feet -/
theorem building_height_is_270 :
  building_height 20 12 3 = 270 := by
  sorry

#eval building_height 20 12 3

end building_height_is_270_l598_59881


namespace water_consumption_problem_l598_59842

/-- The water consumption problem -/
theorem water_consumption_problem 
  (total_water : ℝ) 
  (initial_people : ℕ) 
  (initial_days : ℕ) 
  (later_people : ℕ) 
  (later_days : ℕ) 
  (h1 : total_water = 18.9)
  (h2 : initial_people = 6)
  (h3 : initial_days = 4)
  (h4 : later_people = 7)
  (h5 : later_days = 2) :
  ∃ (x : ℝ), 
    x = 6 ∧ 
    (initial_people * (total_water / (initial_people * initial_days)) * later_days + 
     x * (total_water / (initial_people * initial_days)) * later_days = total_water) := by
  sorry

end water_consumption_problem_l598_59842


namespace intersection_theorem_l598_59863

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (k : ℝ) : ℝ → ℝ := λ x ↦ k * |x| + 2

/-- The curve C₂ in Cartesian coordinates -/
def C₂ : ℝ × ℝ → Prop := λ p ↦ (p.1 + 1)^2 + p.2^2 = 4

/-- The number of intersection points between C₁ and C₂ -/
def numIntersections (k : ℝ) : ℕ := sorry

theorem intersection_theorem (k : ℝ) :
  numIntersections k = 3 → k = -4/3 := by sorry

end intersection_theorem_l598_59863


namespace tangent_line_at_point_l598_59858

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 - y = 0

-- Define the point of tangency
def point : ℝ × ℝ := (-2, -8)

-- Define the proposed tangent line equation
def tangent_line (x y : ℝ) : Prop := 12*x - y + 16 = 0

-- Theorem statement
theorem tangent_line_at_point :
  ∀ x y : ℝ,
  curve x y →
  (x, y) = point →
  tangent_line x y :=
sorry

end tangent_line_at_point_l598_59858


namespace same_heads_probability_l598_59810

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents the result of tossing two coins -/
def TwoCoins := (CoinToss × CoinToss)

/-- The sample space of all possible outcomes when two people each toss two coins -/
def SampleSpace := (TwoCoins × TwoCoins)

/-- Counts the number of heads in a two-coin toss -/
def countHeads : TwoCoins → Nat
| (CoinToss.Heads, CoinToss.Heads) => 2
| (CoinToss.Heads, CoinToss.Tails) => 1
| (CoinToss.Tails, CoinToss.Heads) => 1
| (CoinToss.Tails, CoinToss.Tails) => 0

/-- Checks if two two-coin tosses have the same number of heads -/
def sameHeads (t1 t2 : TwoCoins) : Bool :=
  countHeads t1 = countHeads t2

/-- The number of elements in the sample space -/
def totalOutcomes : Nat := 16

/-- The number of favorable outcomes (same number of heads) -/
def favorableOutcomes : Nat := 6

/-- The probability of getting the same number of heads -/
def probability : Rat := favorableOutcomes / totalOutcomes

theorem same_heads_probability : probability = 3 / 8 := by
  sorry


end same_heads_probability_l598_59810


namespace money_distribution_l598_59898

theorem money_distribution (total : ℕ) (vasim_share : ℕ) : 
  vasim_share = 1500 →
  ∃ (faruk_share ranjith_share : ℕ),
    faruk_share + vasim_share + ranjith_share = total ∧
    5 * faruk_share = 3 * vasim_share ∧
    6 * faruk_share = 3 * ranjith_share ∧
    ranjith_share - faruk_share = 900 := by
  sorry

end money_distribution_l598_59898


namespace cube_geometry_l598_59867

-- Define a cube
def Cube : Type := Unit

-- Define a vertex of a cube
def Vertex (c : Cube) : Type := Unit

-- Define a set of 4 vertices
def FourVertices (c : Cube) : Type := Fin 4 → Vertex c

-- Define a spatial quadrilateral
def SpatialQuadrilateral (c : Cube) (v : FourVertices c) : Prop := sorry

-- Define a tetrahedron
def Tetrahedron (c : Cube) (v : FourVertices c) : Prop := sorry

-- Define an equilateral triangle
def EquilateralTriangle (c : Cube) (v1 v2 v3 : Vertex c) : Prop := sorry

-- Define an isosceles right-angled triangle
def IsoscelesRightTriangle (c : Cube) (v1 v2 v3 : Vertex c) : Prop := sorry

-- Theorem statement
theorem cube_geometry (c : Cube) : 
  (∃ v : FourVertices c, SpatialQuadrilateral c v) ∧ 
  (∃ v : FourVertices c, Tetrahedron c v ∧ 
    (∀ face : Fin 4 → Fin 3, EquilateralTriangle c (v (face 0)) (v (face 1)) (v (face 2)))) ∧
  (∃ v : FourVertices c, Tetrahedron c v ∧ 
    (∃ face : Fin 4 → Fin 3, EquilateralTriangle c (v (face 0)) (v (face 1)) (v (face 2))) ∧
    (∃ faces : Fin 3 → (Fin 4 → Fin 3), 
      ∀ i : Fin 3, IsoscelesRightTriangle c (v ((faces i) 0)) (v ((faces i) 1)) (v ((faces i) 2)))) :=
by
  sorry

end cube_geometry_l598_59867


namespace arithmetic_sequence_slope_l598_59843

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  sum_def : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  a_def : ∀ n, a n = a 1 + (n - 1) * d

/-- The slope of the line passing through P(n, a_n) and Q(n+2, a_{n+2}) is 4 -/
theorem arithmetic_sequence_slope (seq : ArithmeticSequence)
  (h1 : seq.sum 2 = 10)
  (h2 : seq.sum 5 = 55) :
  ∀ n : ℕ, n ≥ 1 → (seq.a (n + 2) - seq.a n) / 2 = 4 :=
by sorry

end arithmetic_sequence_slope_l598_59843


namespace absolute_value_inequality_l598_59817

theorem absolute_value_inequality (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x + 1| < 2) ↔ (-3 < a ∧ a < 1) :=
by sorry

end absolute_value_inequality_l598_59817


namespace power_product_equality_l598_59875

theorem power_product_equality : 3^5 * 7^5 = 4084101 := by
  sorry

end power_product_equality_l598_59875


namespace zoey_holiday_months_l598_59899

/-- The number of holidays Zoey takes per month -/
def holidays_per_month : ℕ := 2

/-- The total number of holidays Zoey took -/
def total_holidays : ℕ := 24

/-- The number of months Zoey took holidays for -/
def months_of_holidays : ℕ := total_holidays / holidays_per_month

/-- Theorem: The number of months Zoey took holidays for is 12 -/
theorem zoey_holiday_months : months_of_holidays = 12 := by
  sorry

end zoey_holiday_months_l598_59899


namespace square_of_1031_l598_59839

theorem square_of_1031 : (1031 : ℕ)^2 = 1060961 := by sorry

end square_of_1031_l598_59839


namespace a_4_equals_zero_l598_59884

/-- Given a sequence {aₙ} with general term aₙ = n² - 3n - 4 for n ∈ ℕ*, prove that a₄ = 0 -/
theorem a_4_equals_zero (a : ℕ+ → ℤ) (h : ∀ n : ℕ+, a n = n.val ^ 2 - 3 * n.val - 4) :
  a 4 = 0 := by
  sorry

end a_4_equals_zero_l598_59884


namespace loaded_cartons_l598_59830

/-- Given information about cartons of canned juice, prove the number of loaded cartons. -/
theorem loaded_cartons (total_cartons : ℕ) (cans_per_carton : ℕ) (cans_left : ℕ) : 
  total_cartons = 50 →
  cans_per_carton = 20 →
  cans_left = 200 →
  total_cartons - (cans_left / cans_per_carton) = 40 :=
by sorry

end loaded_cartons_l598_59830


namespace gas_mixture_ratio_l598_59819

-- Define the gases and elements
inductive Gas : Type
| A : Gas  -- CO2
| B : Gas  -- O2

inductive Element : Type
| C : Element
| O : Element

-- Define the molar mass function
def molarMass : Gas → ℝ
| Gas.A => 44  -- Molar mass of CO2
| Gas.B => 32  -- Molar mass of O2

-- Define the number of atoms of each element in each gas molecule
def atomCount : Gas → Element → ℕ
| Gas.A, Element.C => 1
| Gas.A, Element.O => 2
| Gas.B, Element.C => 0
| Gas.B, Element.O => 2

-- Define the mass ratio of C to O in the mixed gas
def massRatio (x y : ℝ) : Prop :=
  (12 * x) / (16 * (2 * x + 2 * y)) = 1 / 8

-- Define the volume ratio of A to B
def volumeRatio (x y : ℝ) : Prop :=
  x / y = 1 / 2

-- The theorem to prove
theorem gas_mixture_ratio : 
  ∀ (x y : ℝ), x > 0 → y > 0 → massRatio x y → volumeRatio x y :=
sorry

end gas_mixture_ratio_l598_59819


namespace hyperbola_eccentricity_l598_59890

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Represents a point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem: Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (P Q : HyperbolaPoint h) (F₁ : ℝ × ℝ) :
  (∃ (line : ℝ → ℝ × ℝ), 
    line 0 = right_focus h ∧ 
    (∃ t₁ t₂, line t₁ = (P.x, P.y) ∧ line t₂ = (Q.x, Q.y)) ∧
    ((P.x - Q.x) * (P.x - F₁.1) + (P.y - Q.y) * (P.y - F₁.2) = 0) ∧
    ((P.x - Q.x)^2 + (P.y - Q.y)^2 = (P.x - F₁.1)^2 + (P.y - F₁.2)^2)) →
  eccentricity h = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end hyperbola_eccentricity_l598_59890


namespace no_exact_table_count_l598_59837

theorem no_exact_table_count : ¬∃ (t : ℕ), 
  3 * (8 * t) + 4 * (2 * t) + 4 * t = 656 := by
  sorry

end no_exact_table_count_l598_59837


namespace sunday_school_average_class_size_l598_59877

/-- The average class size in a Sunday school with two classes -/
theorem sunday_school_average_class_size 
  (three_year_olds : ℕ) 
  (four_year_olds : ℕ) 
  (five_year_olds : ℕ) 
  (six_year_olds : ℕ) 
  (h1 : three_year_olds = 13)
  (h2 : four_year_olds = 20)
  (h3 : five_year_olds = 15)
  (h4 : six_year_olds = 22) :
  (three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2 = 35 := by
  sorry

#check sunday_school_average_class_size

end sunday_school_average_class_size_l598_59877


namespace rectangular_plot_length_difference_l598_59888

theorem rectangular_plot_length_difference (length breadth perimeter : ℝ) : 
  length = 63 ∧ 
  perimeter = 200 ∧ 
  perimeter = 2 * (length + breadth) → 
  length - breadth = 26 := by
sorry

end rectangular_plot_length_difference_l598_59888


namespace class_selection_ways_l598_59846

def total_classes : ℕ := 10
def advanced_classes : ℕ := 3
def classes_to_select : ℕ := 5
def min_advanced : ℕ := 2

theorem class_selection_ways : 
  (Nat.choose advanced_classes min_advanced) * 
  (Nat.choose (total_classes - advanced_classes) (classes_to_select - min_advanced)) = 105 := by
  sorry

end class_selection_ways_l598_59846


namespace original_number_proof_l598_59878

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 5 * (1 / x)) :
  x = Real.sqrt 2 / 20 := by
  sorry

end original_number_proof_l598_59878


namespace system_solution_l598_59818

theorem system_solution : ∃ (x y : ℝ), x + y = 8 ∧ x - 3*y = 4 ∧ x = 7 ∧ y = 1 := by
  sorry

end system_solution_l598_59818


namespace paint_usage_proof_l598_59836

theorem paint_usage_proof (total_paint : ℝ) (second_week_fraction : ℝ) (total_used : ℝ) 
  (h1 : total_paint = 360)
  (h2 : second_week_fraction = 1/6)
  (h3 : total_used = 135) :
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint + 
    second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used ∧
    first_week_fraction = 1/4 := by
  sorry

end paint_usage_proof_l598_59836


namespace student_arrangement_theorem_l598_59804

/-- Represents the number of students in the group -/
def total_students : ℕ := 9

/-- Represents the probability of selecting at least one girl when choosing 3 students -/
def prob_at_least_one_girl : ℚ := 16/21

/-- Calculates the number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- Theorem stating the main result -/
theorem student_arrangement_theorem :
  ∃ (male_count female_count : ℕ),
    male_count + female_count = total_students ∧
    (choose total_students 3 - choose male_count 3) / (choose total_students 3) = prob_at_least_one_girl ∧
    male_count = 6 ∧
    female_count = 3 ∧
    (choose male_count 2 * choose (male_count - 2) 2 * choose (male_count - 4) 2) / (permute 3 3) *
    (permute female_count female_count) * (permute (female_count + 1) 3) *
    ((permute 2 2) ^ 3) = 17280 :=
by sorry

end student_arrangement_theorem_l598_59804


namespace sum_of_roots_quadratic_l598_59869

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) := by
  sorry

end sum_of_roots_quadratic_l598_59869


namespace selling_price_calculation_l598_59874

/-- Calculates the selling price of an article given its cost price and profit percentage -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem: The selling price of an article with cost price 480 and profit percentage 25% is 600 -/
theorem selling_price_calculation :
  selling_price 480 25 = 600 := by
  sorry

end selling_price_calculation_l598_59874


namespace parallel_vectors_k_value_l598_59806

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (2, -1)

/-- Function to check if two vectors are parallel -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

/-- Main theorem -/
theorem parallel_vectors_k_value :
  ∃ (k : ℝ), are_parallel (a.1 + k * c.1, a.2 + k * c.2) (2 * b.1 - a.1, 2 * b.2 - a.2) ∧ k = 16 := by
  sorry

end parallel_vectors_k_value_l598_59806


namespace missing_village_population_l598_59866

def village_count : Nat := 7
def known_populations : List Nat := [803, 900, 1100, 1023, 980, 1249]
def average_population : Nat := 1000

theorem missing_village_population :
  village_count * average_population - known_populations.sum = 945 := by
  sorry

end missing_village_population_l598_59866


namespace apple_products_total_cost_l598_59824

/-- Calculates the total cost of an iPhone and iWatch after discounts and cashback -/
theorem apple_products_total_cost 
  (iphone_price : ℝ) 
  (iwatch_price : ℝ) 
  (iphone_discount : ℝ) 
  (iwatch_discount : ℝ) 
  (cashback_rate : ℝ) 
  (h1 : iphone_price = 800) 
  (h2 : iwatch_price = 300) 
  (h3 : iphone_discount = 0.15) 
  (h4 : iwatch_discount = 0.10) 
  (h5 : cashback_rate = 0.02) : 
  ℝ := by
  sorry

#check apple_products_total_cost

end apple_products_total_cost_l598_59824


namespace f_monotone_decreasing_l598_59803

-- Define the function f(x)
def f (x : ℝ) : ℝ := x - x^2 - x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x₁ x₂ : ℝ, -1/3 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₂ < f x₁ := by
  sorry

end f_monotone_decreasing_l598_59803


namespace gain_percentage_cloth_sale_l598_59889

/-- Calculates the gain percentage given the total quantity sold and the profit quantity -/
def gainPercentage (totalQuantity : ℕ) (profitQuantity : ℕ) : ℚ :=
  (profitQuantity : ℚ) / (totalQuantity : ℚ)

/-- Theorem: The gain percentage is 1/6 when selling 60 meters of cloth and gaining the selling price of 10 meters as profit -/
theorem gain_percentage_cloth_sale : 
  gainPercentage 60 10 = 1 / 6 := by
  sorry

end gain_percentage_cloth_sale_l598_59889


namespace power_division_eight_sixtyfour_l598_59835

theorem power_division_eight_sixtyfour : 8^15 / 64^7 = 8 := by
  sorry

end power_division_eight_sixtyfour_l598_59835


namespace smaller_number_proof_l598_59807

theorem smaller_number_proof (x y : ℝ) 
  (sum_eq : x + y = 16)
  (diff_eq : x - y = 4)
  (prod_eq : x * y = 60) :
  min x y = 6 := by
sorry

end smaller_number_proof_l598_59807


namespace original_denominator_problem_l598_59829

theorem original_denominator_problem (d : ℝ) : 
  (3 : ℝ) / d ≠ 0 →
  (3 + 3) / (d + 3) = (1 : ℝ) / 3 →
  d = 15 := by
  sorry

end original_denominator_problem_l598_59829


namespace complex_equation_solution_l598_59851

theorem complex_equation_solution :
  ∃ x : ℤ, x - (28 - (37 - (15 - 17))) = 56 ∧ x = 45 := by sorry

end complex_equation_solution_l598_59851


namespace seven_rings_four_fingers_l598_59879

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings fingers * 
  Nat.factorial fingers * 
  Nat.choose (total_rings - 1) (fingers - 1)

/-- Theorem stating the number of ring arrangements for 7 rings on 4 fingers -/
theorem seven_rings_four_fingers : 
  ring_arrangements 7 4 = 29400 := by
  sorry

end seven_rings_four_fingers_l598_59879


namespace oil_fraction_after_replacements_l598_59859

def tank_capacity : ℚ := 20
def replacement_amount : ℚ := 5
def num_replacements : ℕ := 5

def fraction_remaining (n : ℕ) : ℚ := (3/4) ^ n

theorem oil_fraction_after_replacements :
  fraction_remaining num_replacements = 243/1024 := by
  sorry

end oil_fraction_after_replacements_l598_59859


namespace no_negative_roots_l598_59811

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
  sorry

end no_negative_roots_l598_59811


namespace real_estate_problem_l598_59894

-- Define the constants
def total_sets : ℕ := 80
def cost_A : ℕ := 90
def price_A : ℕ := 102
def cost_B : ℕ := 60
def price_B : ℕ := 70
def min_funds : ℕ := 5700
def max_A : ℕ := 32

-- Define the variables
variable (x : ℕ) -- number of Type A sets
variable (W : ℕ → ℕ) -- profit function
variable (a : ℚ) -- price reduction for Type A

-- Define the theorem
theorem real_estate_problem :
  (∀ x, W x = 2 * x + 800) ∧
  (x ≥ 30 ∧ x ≤ 32) ∧
  (∀ a, 0 < a ∧ a ≤ 3 →
    (0 < a ∧ a < 2 → x = 32) ∧
    (a = 2 → true) ∧
    (2 < a ∧ a ≤ 3 → x = 30)) :=
sorry

end real_estate_problem_l598_59894


namespace fifth_term_is_correct_l598_59891

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
| 0 => 2*x + y
| 1 => 2*x - y
| 2 => 2*x*y
| 3 => 2*x / y
| n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that the fifth term of the sequence is -77/10 -/
theorem fifth_term_is_correct (x y : ℚ) :
  arithmetic_sequence x y 0 = 2*x + y →
  arithmetic_sequence x y 1 = 2*x - y →
  arithmetic_sequence x y 2 = 2*x*y →
  arithmetic_sequence x y 3 = 2*x / y →
  arithmetic_sequence x y 4 = -77/10 :=
by sorry

end fifth_term_is_correct_l598_59891


namespace speed_ratio_A_to_B_l598_59872

-- Define the work completion rates for A and B
def work_rate_B : ℚ := 1 / 12
def work_rate_A_and_B : ℚ := 1 / 4

-- Define A's work rate in terms of B's
def work_rate_A : ℚ := work_rate_A_and_B - work_rate_B

-- Theorem statement
theorem speed_ratio_A_to_B : 
  work_rate_A / work_rate_B = 2 := by
  sorry

end speed_ratio_A_to_B_l598_59872


namespace integer_solution_for_equation_l598_59805

theorem integer_solution_for_equation (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 15) + 1 = (x + b) * (x + c)) →
  (a = 13 ∨ a = 17) := by
  sorry

end integer_solution_for_equation_l598_59805


namespace julia_parrot_weeks_l598_59861

/-- Represents the problem of determining how long Julia has had her parrot -/
theorem julia_parrot_weeks : 
  ∀ (total_weekly_cost rabbit_weekly_cost total_spent rabbit_weeks : ℕ),
  total_weekly_cost = 30 →
  rabbit_weekly_cost = 12 →
  rabbit_weeks = 5 →
  total_spent = 114 →
  ∃ (parrot_weeks : ℕ),
    parrot_weeks * (total_weekly_cost - rabbit_weekly_cost) = 
      total_spent - (rabbit_weeks * rabbit_weekly_cost) ∧
    parrot_weeks = 3 :=
by sorry

end julia_parrot_weeks_l598_59861


namespace evaluate_expression_l598_59883

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 5) = 14 := by
sorry

end evaluate_expression_l598_59883


namespace lcm_ratio_implies_gcd_l598_59893

theorem lcm_ratio_implies_gcd (X Y : ℕ+) : 
  Nat.lcm X Y = 180 → X * 6 = Y * 5 → Nat.gcd X Y = 6 := by
  sorry

end lcm_ratio_implies_gcd_l598_59893


namespace power_twelve_half_l598_59840

theorem power_twelve_half : (12 : ℕ) ^ ((12 : ℕ) / 2) = 2985984 := by sorry

end power_twelve_half_l598_59840


namespace average_daily_attendance_l598_59892

def monday_attendance : ℕ := 10
def tuesday_attendance : ℕ := 15
def wednesday_to_friday_attendance : ℕ := 10
def total_days : ℕ := 5

def total_attendance : ℕ := monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance

theorem average_daily_attendance :
  (total_attendance : ℚ) / total_days = 11 := by sorry

end average_daily_attendance_l598_59892


namespace range_of_function_l598_59848

theorem range_of_function (x : ℝ) : 
  1/3 ≤ (2 - Real.cos x) / (2 + Real.cos x) ∧ (2 - Real.cos x) / (2 + Real.cos x) ≤ 3 := by
  sorry

end range_of_function_l598_59848


namespace direct_proportion_m_value_l598_59868

-- Define the function y as a direct proportion function
def is_direct_proportion (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → (m - 2) * x^(m^2 - 3) = k * x

-- Theorem statement
theorem direct_proportion_m_value :
  (∃ m : ℝ, is_direct_proportion m) → (∃ m : ℝ, is_direct_proportion m ∧ m = -2) :=
sorry

end direct_proportion_m_value_l598_59868


namespace ac_over_b_squared_eq_one_l598_59885

/-- A quadratic equation with real coefficients -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  has_imaginary_roots : ∃ (x₁ x₂ : ℂ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ ∧ x₁.im ≠ 0 ∧ x₂.im ≠ 0
  x₁_cubed_real : ∃ (x₁ : ℂ), (a * x₁^2 + b * x₁ + c = 0) ∧ (∃ (r : ℝ), x₁^3 = r)

/-- Theorem stating that ac/b^2 = 1 for a quadratic equation satisfying the given conditions -/
theorem ac_over_b_squared_eq_one (eq : QuadraticEquation) : eq.a * eq.c / eq.b^2 = 1 := by
  sorry

end ac_over_b_squared_eq_one_l598_59885


namespace max_digits_product_four_digit_numbers_l598_59852

theorem max_digits_product_four_digit_numbers :
  ∀ a b : ℕ, 1000 ≤ a ∧ a ≤ 9999 → 1000 ≤ b ∧ b ≤ 9999 →
  ∃ n : ℕ, n ≤ 8 ∧ a * b < 10^n :=
by sorry

end max_digits_product_four_digit_numbers_l598_59852


namespace inequality_proof_l598_59865

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q) 
  (h2 : a * c ≥ p^2) 
  (h3 : p^2 > 0) : 
  b * d ≤ q^2 := by
  sorry

end inequality_proof_l598_59865


namespace rope_cutting_problem_l598_59802

theorem rope_cutting_problem (a b c : ℕ) 
  (ha : a = 45) (hb : b = 60) (hc : c = 75) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end rope_cutting_problem_l598_59802


namespace arithmetic_sequence_sum_ratio_l598_59854

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Main theorem: If S_6 / S_3 = 4 for an arithmetic sequence, then S_9 / S_6 = 9/4 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 4) : 
  seq.S 9 / seq.S 6 = 9/4 := by
  sorry

end arithmetic_sequence_sum_ratio_l598_59854


namespace geometric_sequence_property_l598_59856

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  a 2 * a 3 = 5 ∧
  a 5 * a 6 = 10

/-- Theorem stating the property of the 8th and 9th terms -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 8 * a 9 = 20 := by
  sorry

end geometric_sequence_property_l598_59856


namespace e_percentage_of_d_l598_59850

-- Define the variables
variable (a b c d e : ℝ)

-- Define the relationships between the variables
def relationship_d : Prop := d = 0.4 * a ∧ d = 0.35 * b
def relationship_e : Prop := e = 0.5 * b ∧ e = 0.2 * c
def relationship_c : Prop := c = 0.3 * a ∧ c = 0.25 * b

-- Theorem statement
theorem e_percentage_of_d 
  (hd : relationship_d a b d)
  (he : relationship_e b c e)
  (hc : relationship_c a b c) :
  e / d = 0.15 := by sorry

end e_percentage_of_d_l598_59850


namespace tournament_dominating_set_exists_l598_59808

/-- Represents a directed graph where vertices are players and edges represent wins. -/
structure TournamentGraph where
  players : Finset ℕ
  wins : players → players → Prop

/-- A tournament graph is complete if every player has played against every other player exactly once. -/
def IsCompleteTournament (g : TournamentGraph) : Prop :=
  ∀ p q : g.players, p ≠ q → (g.wins p q ∨ g.wins q p) ∧ ¬(g.wins p q ∧ g.wins q p)

/-- A set of players dominates the rest if every other player has lost to at least one player in the set. -/
def DominatingSet (g : TournamentGraph) (s : Finset g.players) : Prop :=
  ∀ p : g.players, p ∉ s → ∃ q ∈ s, g.wins q p

theorem tournament_dominating_set_exists (g : TournamentGraph) 
  (h_complete : IsCompleteTournament g) (h_size : g.players.card = 14) :
  ∃ s : Finset g.players, s.card = 3 ∧ DominatingSet g s := by sorry

end tournament_dominating_set_exists_l598_59808


namespace fiona_final_piles_count_l598_59801

/-- Represents the number of distinct final pile configurations in Fiona's card arranging process. -/
def fiona_final_piles (n : ℕ) : ℕ :=
  if n ≥ 2 then 2^(n-2) else 1

/-- The theorem stating the number of distinct final pile configurations in Fiona's card arranging process. -/
theorem fiona_final_piles_count (n : ℕ) :
  (∀ k : ℕ, k < n → ∃ (m : ℕ), m ≤ n ∧ fiona_final_piles k = fiona_final_piles m) →
  fiona_final_piles n = if n ≥ 2 then 2^(n-2) else 1 :=
by sorry

end fiona_final_piles_count_l598_59801


namespace max_product_value_l598_59857

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value :
  (∀ x, -7 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ x, f x * g x = 21) ∧
  (∀ x, f x * g x ≤ 21) :=
sorry

end max_product_value_l598_59857


namespace cuboid_volume_with_margin_eq_l598_59838

/-- The volume of points inside or within two units of a cuboid with dimensions 5 by 6 by 8 units -/
def cuboid_volume_with_margin : ℝ := sorry

/-- The dimensions of the cuboid -/
def cuboid_dimensions : Fin 3 → ℕ
  | 0 => 5
  | 1 => 6
  | 2 => 8
  | _ => 0

/-- The margin around the cuboid -/
def margin : ℕ := 2

/-- Theorem stating that the volume of points inside or within two units of the cuboid 
    is equal to (2136 + 140π)/3 cubic units -/
theorem cuboid_volume_with_margin_eq : 
  cuboid_volume_with_margin = (2136 + 140 * Real.pi) / 3 := by sorry

end cuboid_volume_with_margin_eq_l598_59838


namespace smallest_positive_solution_l598_59844

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x + 1 ∧
  ∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y + 1 → x ≤ y ∧
  x = (-7 - Real.sqrt 349) / 50 :=
by sorry

end smallest_positive_solution_l598_59844


namespace proportion_solution_l598_59855

theorem proportion_solution (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y ∧ x = 1.65 → y = 11 := by
  sorry

end proportion_solution_l598_59855


namespace bowling_score_ratio_l598_59862

theorem bowling_score_ratio (total_score : ℕ) (third_score : ℕ) : 
  total_score = 810 →
  third_score = 162 →
  ∃ (first_score second_score : ℕ),
    first_score + second_score + third_score = total_score ∧
    first_score = second_score / 3 →
    second_score / third_score = 3 := by
sorry

end bowling_score_ratio_l598_59862


namespace problem_solution_l598_59896

theorem problem_solution :
  let M : ℕ := 3009 / 3
  let N : ℕ := (2 * M) / 3
  let X : ℤ := M - N
  X = 335 := by sorry

end problem_solution_l598_59896


namespace people_per_car_l598_59820

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) :
  total_people / num_cars = 21 :=
by sorry

end people_per_car_l598_59820


namespace gardener_tree_rows_l598_59833

/-- Proves that the initial number of rows is 24 given the gardener's tree planting conditions -/
theorem gardener_tree_rows : ∀ r : ℕ, 
  (42 * r = 28 * (r + 12)) → r = 24 := by
  sorry

end gardener_tree_rows_l598_59833


namespace systematic_sampling_distribution_l598_59814

/-- Represents a building in the school -/
inductive Building
| A
| B
| C

/-- Represents the systematic sampling method -/
def systematicSampling (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  let interval := total / sampleSize
  List.range (total - start + 1)
    |> List.filter (fun i => (i + start - 1) % interval == 0)
    |> List.map (fun i => i + start - 1)

/-- Assigns a student number to a building -/
def assignBuilding (studentNumber : ℕ) : Building :=
  if studentNumber ≤ 200 then Building.A
  else if studentNumber ≤ 295 then Building.B
  else Building.C

/-- Counts the number of students selected for each building -/
def countSelectedStudents (selectedStudents : List ℕ) : ℕ × ℕ × ℕ :=
  selectedStudents.foldl
    (fun (a, b, c) student =>
      match assignBuilding student with
      | Building.A => (a + 1, b, c)
      | Building.B => (a, b + 1, c)
      | Building.C => (a, b, c + 1))
    (0, 0, 0)

theorem systematic_sampling_distribution :
  let totalStudents := 400
  let sampleSize := 50
  let firstNumber := 3
  let selectedStudents := systematicSampling totalStudents sampleSize firstNumber
  let (buildingA, buildingB, buildingC) := countSelectedStudents selectedStudents
  buildingA = 25 ∧ buildingB = 12 ∧ buildingC = 13 := by
  sorry


end systematic_sampling_distribution_l598_59814
