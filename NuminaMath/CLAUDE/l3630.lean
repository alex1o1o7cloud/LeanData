import Mathlib

namespace distance_between_trees_l3630_363034

/-- Given a yard with trees planted at equal distances, calculate the distance between consecutive trees. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 255 ∧ num_trees = 18 → 
  (yard_length / (num_trees - 1 : ℝ)) = 15 := by
  sorry

end distance_between_trees_l3630_363034


namespace cylinder_equation_l3630_363031

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSurface c

theorem cylinder_equation (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSurface c) := by
  sorry

end cylinder_equation_l3630_363031


namespace cans_storage_l3630_363014

theorem cans_storage (cans_per_row : ℕ) (shelves_per_closet : ℕ) (cans_per_closet : ℕ) :
  cans_per_row = 12 →
  shelves_per_closet = 10 →
  cans_per_closet = 480 →
  (cans_per_closet / cans_per_row) / shelves_per_closet = 4 :=
by sorry

end cans_storage_l3630_363014


namespace parallel_line_plane_l3630_363069

-- Define the types for lines and planes
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem statement
theorem parallel_line_plane 
  (α β : Plane) (n : Line) 
  (h1 : parallel α β) 
  (h2 : subset n α) : 
  line_parallel n β :=
sorry

end parallel_line_plane_l3630_363069


namespace monotonic_decreasing_intervals_of_neg_tan_l3630_363013

open Real

noncomputable def f (x : ℝ) := -tan x

theorem monotonic_decreasing_intervals_of_neg_tan :
  ∀ (k : ℤ) (x y : ℝ),
    x ∈ Set.Ioo (k * π - π / 2) (k * π + π / 2) →
    y ∈ Set.Ioo (k * π - π / 2) (k * π + π / 2) →
    x < y →
    f x > f y :=
by sorry

end monotonic_decreasing_intervals_of_neg_tan_l3630_363013


namespace impossible_equal_sum_distribution_l3630_363049

theorem impossible_equal_sum_distribution : ∀ n : ℕ, 2 ≤ n → n ≤ 14 →
  ¬ ∃ (partition : List (List ℕ)), 
    (∀ group ∈ partition, ∀ x ∈ group, 1 ≤ x ∧ x ≤ 14) ∧
    (partition.length = n) ∧
    (∀ group ∈ partition, group.sum = 105 / n) ∧
    (partition.join.toFinset = Finset.range 14) :=
by sorry

end impossible_equal_sum_distribution_l3630_363049


namespace unique_three_digit_number_divisible_by_11_l3630_363024

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def divisible_by (a b : ℕ) : Prop := b ∣ a

theorem unique_three_digit_number_divisible_by_11 :
  ∃! n : ℕ, is_three_digit n ∧ 
            units_digit n = 3 ∧ 
            hundreds_digit n = 6 ∧ 
            divisible_by n 11 ∧
            n = 693 :=
sorry

end unique_three_digit_number_divisible_by_11_l3630_363024


namespace correct_representation_l3630_363074

/-- Represents "a number that is 3 more than twice x" -/
def number_3_more_than_twice_x (x : ℝ) : ℝ := 2 * x + 3

/-- The algebraic expression 2x + 3 correctly represents "a number that is 3 more than twice x" -/
theorem correct_representation (x : ℝ) :
  number_3_more_than_twice_x x = 2 * x + 3 := by
  sorry

end correct_representation_l3630_363074


namespace median_exists_for_seven_prices_l3630_363097

theorem median_exists_for_seven_prices (prices : List ℝ) (h : prices.length = 7) :
  ∃ (median : ℝ), median ∈ prices ∧ 
    (prices.filter (λ x => x ≤ median)).length ≥ 4 ∧
    (prices.filter (λ x => x ≥ median)).length ≥ 4 := by
  sorry

end median_exists_for_seven_prices_l3630_363097


namespace quadratic_roots_l3630_363001

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  passesThrough : f (-3) = 0 ∧ f (-2) = -3 ∧ f 0 = -3

/-- The roots of the quadratic function -/
def roots (qf : QuadraticFunction) : Set ℝ :=
  {x : ℝ | qf.f x = 0}

/-- Theorem stating the roots of the quadratic function -/
theorem quadratic_roots (qf : QuadraticFunction) : roots qf = {-3, 1} := by
  sorry

end quadratic_roots_l3630_363001


namespace binomial_coefficient_two_l3630_363040

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l3630_363040


namespace girls_in_class_l3630_363080

/-- The number of boys in the class -/
def num_boys : ℕ := 13

/-- The number of ways to select 1 girl and 2 boys -/
def num_selections : ℕ := 780

/-- The number of girls in the class -/
def num_girls : ℕ := 10

theorem girls_in_class : 
  num_girls * (num_boys.choose 2) = num_selections :=
sorry

end girls_in_class_l3630_363080


namespace car_ordering_theorem_l3630_363012

/-- Represents a car with its speeds in different road segments -/
structure Car where
  citySpeed : ℝ
  nonCitySpeed : ℝ

/-- Represents a point on the road -/
structure RoadPoint where
  cityDistance : ℝ
  nonCityDistance : ℝ

/-- The theorem statement -/
theorem car_ordering_theorem 
  (cars : Fin 10 → Car) 
  (points : Fin 2011 → RoadPoint) :
  ∃ i j, i ≠ j ∧ 
    (∀ (c₁ c₂ : Fin 10), 
      (cars c₁).citySpeed / (cars c₂).citySpeed < (cars c₁).nonCitySpeed / (cars c₂).nonCitySpeed →
      ((points i).cityDistance / (cars c₁).citySpeed + (points i).nonCityDistance / (cars c₁).nonCitySpeed <
       (points i).cityDistance / (cars c₂).citySpeed + (points i).nonCityDistance / (cars c₂).nonCitySpeed) ↔
      ((points j).cityDistance / (cars c₁).citySpeed + (points j).nonCityDistance / (cars c₁).nonCitySpeed <
       (points j).cityDistance / (cars c₂).citySpeed + (points j).nonCityDistance / (cars c₂).nonCitySpeed)) :=
by sorry

end car_ordering_theorem_l3630_363012


namespace number_count_l3630_363033

theorem number_count (average_all : ℝ) (average_group1 : ℝ) (average_group2 : ℝ) (average_group3 : ℝ) 
  (h1 : average_all = 3.9)
  (h2 : average_group1 = 3.4)
  (h3 : average_group2 = 3.85)
  (h4 : average_group3 = 4.45) :
  ∃ (n : ℕ), n = 6 ∧ (n : ℝ) * average_all = 2 * (average_group1 + average_group2 + average_group3) := by
  sorry

end number_count_l3630_363033


namespace factorization_equality_l3630_363065

theorem factorization_equality (x : ℝ) : -x^3 - 2*x^2 - x = -x*(x+1)^2 := by
  sorry

end factorization_equality_l3630_363065


namespace x_fourth_plus_reciprocal_l3630_363055

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x^2 + 1/x^2 = 5) : x^4 + 1/x^4 = 23 := by
  sorry

end x_fourth_plus_reciprocal_l3630_363055


namespace expression_evaluation_l3630_363079

theorem expression_evaluation : 
  |((4:ℝ)^2 - 8*((3:ℝ)^2 - 12))^2| - |Real.sin (5*π/6) - Real.cos (11*π/3)| = 1600 := by
  sorry

end expression_evaluation_l3630_363079


namespace arithmetic_sequence_20th_term_l3630_363045

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
sorry

end arithmetic_sequence_20th_term_l3630_363045


namespace circle_condition_relationship_l3630_363009

theorem circle_condition_relationship :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → (x - 1)^2 + y^2 ≤ 4) ∧
  (∃ x y : ℝ, (x - 1)^2 + y^2 ≤ 4 ∧ x^2 + y^2 > 1) :=
by sorry

end circle_condition_relationship_l3630_363009


namespace certain_number_proof_l3630_363064

theorem certain_number_proof (z : ℤ) (h1 : z % 9 = 6) 
  (h2 : ∃ x : ℤ, ∃ m : ℤ, (z + x) / 9 = m) : 
  ∃ x : ℤ, x = 3 ∧ ∃ m : ℤ, (z + x) / 9 = m :=
sorry

end certain_number_proof_l3630_363064


namespace fixed_distance_theorem_l3630_363036

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_fixed_distance (p a b : V) : Prop :=
  ∃ (k : ℝ), ∀ (q : V), ‖p - q‖ = k → q = (4/3 : ℝ) • a - (1/3 : ℝ) • b

theorem fixed_distance_theorem (a b p : V) 
  (h : ‖p - b‖ = 2 * ‖p - a‖) : is_fixed_distance p a b := by
  sorry

end fixed_distance_theorem_l3630_363036


namespace smallest_number_with_ten_even_five_or_seven_l3630_363060

def containsDigit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def isEvenWithFiveOrSeven (n : ℕ) : Prop :=
  n % 2 = 0 ∧ (containsDigit n 5 ∨ containsDigit n 7)

theorem smallest_number_with_ten_even_five_or_seven : 
  (∃! m : ℕ, m > 0 ∧ (∃ S : Finset ℕ, Finset.card S = 10 ∧ 
    (∀ n ∈ S, n < m ∧ isEvenWithFiveOrSeven n) ∧
    (∀ n : ℕ, n < m → isEvenWithFiveOrSeven n → n ∈ S))) ∧
  (∀ m : ℕ, m > 0 → (∃ S : Finset ℕ, Finset.card S = 10 ∧ 
    (∀ n ∈ S, n < m ∧ isEvenWithFiveOrSeven n) ∧
    (∀ n : ℕ, n < m → isEvenWithFiveOrSeven n → n ∈ S)) → m ≥ 160) :=
by sorry

end smallest_number_with_ten_even_five_or_seven_l3630_363060


namespace odd_function_value_at_negative_one_l3630_363047

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + m)
  (m : ℝ) :
  f (-1) = -3 :=
sorry

end odd_function_value_at_negative_one_l3630_363047


namespace right_triangles_problem_l3630_363027

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define a right triangle
def RightTriangle (A B C : ℝ × ℝ) := Triangle A B C ∧ True

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem right_triangles_problem 
  (A B C D : ℝ × ℝ) 
  (a : ℝ) 
  (h1 : RightTriangle A B C)
  (h2 : RightTriangle A B D)
  (h3 : Length B C = 3)
  (h4 : Length A C = a)
  (h5 : Length A D = 1) :
  Length B D = Real.sqrt (a^2 + 8) := by
  sorry


end right_triangles_problem_l3630_363027


namespace fence_length_proof_l3630_363081

theorem fence_length_proof (darren_length : ℝ) (doug_length : ℝ) : 
  darren_length = 1.2 * doug_length →
  darren_length = 360 →
  darren_length + doug_length = 660 :=
by
  sorry

end fence_length_proof_l3630_363081


namespace gcd_111_1850_l3630_363053

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by
  sorry

end gcd_111_1850_l3630_363053


namespace triangle_theorem_l3630_363011

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2 * Real.sqrt 3) : 
  t.A = 2 * Real.pi / 3 ∧ 
  (∀ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A → s ≤ Real.sqrt 3) ∧
  (∃ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A ∧ s = Real.sqrt 3) :=
by sorry

end

end triangle_theorem_l3630_363011


namespace largest_divisor_of_1615_l3630_363007

theorem largest_divisor_of_1615 (n : ℕ) : n ≤ 5 ↔ n * 1615 ≤ 8640 ∧ n * 1615 ≥ 1000 := by
  sorry

end largest_divisor_of_1615_l3630_363007


namespace product_sign_l3630_363061

theorem product_sign (a b c d e : ℝ) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : 
  a * b^2 * c * d^4 * e < 0 := by
sorry

end product_sign_l3630_363061


namespace largest_b_for_no_real_roots_l3630_363083

theorem largest_b_for_no_real_roots : ∃ (b : ℤ),
  (∀ (x : ℝ), x^3 + b*x^2 + 15*x + 22 ≠ 0) ∧
  (∀ (b' : ℤ), b' > b → ∃ (x : ℝ), x^3 + b'*x^2 + 15*x + 22 = 0) ∧
  b = 5 := by
  sorry


end largest_b_for_no_real_roots_l3630_363083


namespace difference_61st_terms_l3630_363056

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def sequenceC (n : ℕ) : ℝ := arithmeticSequence 20 15 n

def sequenceD (n : ℕ) : ℝ := arithmeticSequence 20 (-15) n

theorem difference_61st_terms :
  |sequenceC 61 - sequenceD 61| = 1800 := by
  sorry

end difference_61st_terms_l3630_363056


namespace perpendicular_slope_l3630_363050

/-- Given a line with equation 5x - 2y = 10, the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) →
  (slope_of_perpendicular_line : ℝ) = -2/5 :=
by sorry

end perpendicular_slope_l3630_363050


namespace minimum_value_and_range_of_a_l3630_363096

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1/2 * x^2 - x

theorem minimum_value_and_range_of_a :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = 2) →
  (∃ x_min : ℝ, x_min > 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ f a x_min) →
  (f a 2 = -2 * Real.log 2) ∧
  (∀ x : ℝ, x > Real.exp 1 → f a x - a * x > 0) →
  a ≤ (Real.exp 2 - 2 * Real.exp 1) / (2 * (Real.exp 1 - 1)) :=
sorry

end minimum_value_and_range_of_a_l3630_363096


namespace divisible_by_64_l3630_363054

theorem divisible_by_64 (n : ℕ+) : ∃ k : ℤ, 3^(2*n.val + 2) - 8*n.val - 9 = 64*k := by
  sorry

end divisible_by_64_l3630_363054


namespace total_distance_is_62_l3630_363077

/-- Calculates the total distance walked over three days given specific conditions --/
def total_distance_walked (day1_distance : ℕ) (day1_speed : ℕ) : ℕ :=
  let day1_hours := day1_distance / day1_speed
  let day2_hours := day1_hours - 1
  let day2_speed := day1_speed + 1
  let day2_distance := day2_hours * day2_speed
  let day3_hours := day1_hours
  let day3_speed := day2_speed
  let day3_distance := day3_hours * day3_speed
  day1_distance + day2_distance + day3_distance

/-- Theorem stating that the total distance walked is 62 miles --/
theorem total_distance_is_62 : total_distance_walked 18 3 = 62 := by
  sorry

end total_distance_is_62_l3630_363077


namespace negative_one_half_less_than_negative_one_third_l3630_363035

theorem negative_one_half_less_than_negative_one_third :
  -1/2 < -1/3 := by sorry

end negative_one_half_less_than_negative_one_third_l3630_363035


namespace fixed_point_on_line_l3630_363093

theorem fixed_point_on_line (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 := by
  sorry

end fixed_point_on_line_l3630_363093


namespace parallelepiped_vector_sum_l3630_363088

/-- In a parallelepiped ABCD-A₁B₁C₁D₁, if AC₁ = x⋅AB + 2y⋅BC + 3z⋅CC₁, then x + y + z = 11/6 -/
theorem parallelepiped_vector_sum (ABCD_A₁B₁C₁D₁ : Set (EuclideanSpace ℝ (Fin 3)))
  (AB BC CC₁ AC₁ : EuclideanSpace ℝ (Fin 3)) (x y z : ℝ) :
  AC₁ = x • AB + (2 * y) • BC + (3 * z) • CC₁ →
  AC₁ = AB + BC + CC₁ →
  x + y + z = 11 / 6 := by
  sorry

end parallelepiped_vector_sum_l3630_363088


namespace sam_has_46_balloons_l3630_363043

/-- Given the number of red balloons Fred and Dan have, and the total number of red balloons,
    calculate the number of red balloons Sam has. -/
def sams_balloons (fred_balloons dan_balloons total_balloons : ℕ) : ℕ :=
  total_balloons - (fred_balloons + dan_balloons)

/-- Theorem stating that given the specific numbers of balloons in the problem,
    Sam must have 46 red balloons. -/
theorem sam_has_46_balloons :
  sams_balloons 10 16 72 = 46 := by
  sorry

end sam_has_46_balloons_l3630_363043


namespace function_properties_l3630_363073

/-- Given a function f(x) = a*sin(2x) + cos(2x) where f(π/3) = (√3 - 1)/2,
    prove properties about the value of a, the maximum value of f(x),
    and the intervals where f(x) is monotonically decreasing. -/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * Real.sin (2 * x) + Real.cos (2 * x)) :
  f (π / 3) = (Real.sqrt 3 - 1) / 2 →
  (a = 1 ∧ 
   (∃ M, M = Real.sqrt 2 ∧ ∀ x, f x ≤ M) ∧
   ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + π / 4) (k * π + 3 * π / 4), 
     ∀ y ∈ Set.Icc (k * π + π / 4) (k * π + 3 * π / 4), 
       x ≤ y → f y ≤ f x) :=
by
  sorry

end function_properties_l3630_363073


namespace sqrt_equation_solution_l3630_363071

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (5 + n) = 7 → n = 44 := by
  sorry

end sqrt_equation_solution_l3630_363071


namespace book_sale_price_l3630_363086

theorem book_sale_price (total_books : ℕ) (sold_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ) : 
  sold_books = (2 : ℕ) * total_books / 3 →
  unsold_books = 30 →
  sold_books + unsold_books = total_books →
  total_amount = 255 →
  total_amount / sold_books = 17/4 := by
  sorry

#eval (17 : ℚ) / 4  -- This should evaluate to 4.25

end book_sale_price_l3630_363086


namespace total_cost_is_44_l3630_363068

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of a single cookie in dollars -/
def cookie_cost : ℕ := 1

/-- The number of sandwiches to purchase -/
def num_sandwiches : ℕ := 4

/-- The number of sodas to purchase -/
def num_sodas : ℕ := 6

/-- The number of cookies to purchase -/
def num_cookies : ℕ := 10

/-- Theorem stating that the total cost of the purchase is $44 -/
theorem total_cost_is_44 :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost = 44 := by
  sorry

end total_cost_is_44_l3630_363068


namespace triangle_sine_inequality_l3630_363084

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  -2 < Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ 3 * Real.sqrt 3 / 2 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sqrt 3 / 2 ↔
   A = 7 * π / 9 ∧ B = π / 9 ∧ C = π / 9) :=
by sorry

end triangle_sine_inequality_l3630_363084


namespace teacher_age_l3630_363010

/-- Given a class of students and their teacher, calculate the teacher's age based on how it affects the class average. -/
theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 22 →
  student_avg_age = 21 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 44 :=
by sorry

end teacher_age_l3630_363010


namespace x_in_terms_of_z_l3630_363075

theorem x_in_terms_of_z (x y z : ℝ) 
  (eq1 : 0.35 * (400 + y) = 0.20 * x)
  (eq2 : x = 2 * z^2)
  (eq3 : y = 3 * z - 5) :
  x = 2 * z^2 := by
sorry

end x_in_terms_of_z_l3630_363075


namespace function_properties_l3630_363004

def f (x a : ℝ) : ℝ := (4*a + 2)*x^2 + (9 - 6*a)*x - 4*a + 4

theorem function_properties :
  (∀ a : ℝ, ∃ x : ℝ, f x a = 0) ∧
  (∃ a : ℤ, ∃ x : ℤ, f (x : ℝ) (a : ℝ) = 0) ∧
  ({a : ℤ | ∃ x : ℤ, f (x : ℝ) (a : ℝ) = 0} = {-2, -1, 0, 1}) :=
by sorry

end function_properties_l3630_363004


namespace clara_owes_mandy_l3630_363099

/-- The amount Clara owes Mandy for cleaning rooms -/
def amount_owed (rate : ℚ) (rooms : ℚ) (discount_threshold : ℚ) (discount_rate : ℚ) : ℚ :=
  let base_amount := rate * rooms
  if rooms > discount_threshold then
    base_amount * (1 - discount_rate)
  else
    base_amount

/-- Theorem stating the amount Clara owes Mandy -/
theorem clara_owes_mandy :
  let rate : ℚ := 15 / 4
  let rooms : ℚ := 12 / 5
  let discount_threshold : ℚ := 2
  let discount_rate : ℚ := 1 / 10
  amount_owed rate rooms discount_threshold discount_rate = 81 / 10 := by
  sorry

end clara_owes_mandy_l3630_363099


namespace fib_sum_product_l3630_363003

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem: F_{m+n} = F_{m-1} * F_n + F_m * F_{n+1} for all non-negative integers m and n -/
theorem fib_sum_product (m n : ℕ) : fib (m + n) = fib (m - 1) * fib n + fib m * fib (n + 1) := by
  sorry

end fib_sum_product_l3630_363003


namespace inverse_proportion_l3630_363098

/-- Given that x is inversely proportional to y, prove that if x = 4 when y = 2, then x = -8/5 when y = -5 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  -5 * (-8/5 : ℝ) = k := by
sorry

end inverse_proportion_l3630_363098


namespace problem_solution_l3630_363032

theorem problem_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = 100) (h4 : (3/10) * a = (1/5) * b) : b = 60 := by
  sorry

end problem_solution_l3630_363032


namespace arithmetic_progression_nested_l3630_363059

/-- An arithmetic progression of distinct positive integers -/
def ArithmeticProgression (s : ℕ → ℕ) : Prop :=
  ∃ a b : ℤ, a ≠ 0 ∧ ∀ n : ℕ, s n = a * n + b

/-- The sequence is strictly increasing -/
def StrictlyIncreasing (s : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → s m < s n

/-- All elements in the sequence are positive -/
def AllPositive (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < s n

theorem arithmetic_progression_nested (s : ℕ → ℕ) :
  ArithmeticProgression s →
  StrictlyIncreasing s →
  AllPositive s →
  ArithmeticProgression (fun n ↦ s (s n)) ∧
  StrictlyIncreasing (fun n ↦ s (s n)) ∧
  AllPositive (fun n ↦ s (s n)) :=
by sorry

end arithmetic_progression_nested_l3630_363059


namespace store_shelves_proof_l3630_363044

/-- Calculates the number of shelves needed to store coloring books -/
def shelves_needed (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (initial_stock - books_sold) / books_per_shelf

/-- Proves that the number of shelves needed is 7 given the problem conditions -/
theorem store_shelves_proof :
  shelves_needed 86 37 7 = 7 := by
  sorry

end store_shelves_proof_l3630_363044


namespace select_two_from_four_l3630_363092

theorem select_two_from_four (n : ℕ) (k : ℕ) : n = 4 → k = 2 → Nat.choose n k = 6 := by
  sorry

end select_two_from_four_l3630_363092


namespace collinear_points_y_value_l3630_363020

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_y_value :
  let A : Point := { x := 4, y := 8 }
  let B : Point := { x := 2, y := 4 }
  let C : Point := { x := 3, y := y }
  collinear A B C → y = 6 := by
  sorry

end collinear_points_y_value_l3630_363020


namespace six_digit_number_divisible_by_7_8_9_l3630_363085

theorem six_digit_number_divisible_by_7_8_9 : ∃ (n₁ n₂ : ℕ),
  n₁ ≠ n₂ ∧
  523000 ≤ n₁ ∧ n₁ < 524000 ∧
  523000 ≤ n₂ ∧ n₂ < 524000 ∧
  n₁ % 7 = 0 ∧ n₁ % 8 = 0 ∧ n₁ % 9 = 0 ∧
  n₂ % 7 = 0 ∧ n₂ % 8 = 0 ∧ n₂ % 9 = 0 :=
by
  sorry

end six_digit_number_divisible_by_7_8_9_l3630_363085


namespace son_age_proof_l3630_363028

theorem son_age_proof (son_age man_age : ℕ) : 
  man_age = son_age + 22 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end son_age_proof_l3630_363028


namespace arithmetic_sequence_sum_l3630_363021

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement that a_3 and a_10 are roots of x^2 - 3x - 5 = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  a 3 ^ 2 - 3 * a 3 - 5 = 0 ∧ a 10 ^ 2 - 3 * a 10 - 5 = 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : roots_condition a) : 
  a 5 + a 8 = 3 := by
  sorry

end arithmetic_sequence_sum_l3630_363021


namespace sheila_picnic_probability_l3630_363018

-- Define the probabilities
def rain_probability : ℝ := 0.5
def picnic_if_rain : ℝ := 0.3
def picnic_if_sunny : ℝ := 0.7

-- Theorem statement
theorem sheila_picnic_probability :
  rain_probability * picnic_if_rain + (1 - rain_probability) * picnic_if_sunny = 0.5 := by
  sorry

#eval rain_probability * picnic_if_rain + (1 - rain_probability) * picnic_if_sunny

end sheila_picnic_probability_l3630_363018


namespace shirt_problem_l3630_363029

/-- Given the prices of sarees and shirts, prove the number of shirts that can be bought for $2400 -/
theorem shirt_problem (S T : ℚ) (h1 : 2 * S + 4 * T = 1600) (h2 : S + 6 * T = 1600) :
  ∃ X : ℚ, X * T = 2400 ∧ X = 12 := by
  sorry

end shirt_problem_l3630_363029


namespace ratio_xyz_l3630_363038

theorem ratio_xyz (x y z : ℝ) 
  (h1 : 0.6 * x = 0.3 * y)
  (h2 : 0.8 * z = 0.4 * x)
  (h3 : z = 2 * y) :
  ∃ (k : ℝ), k > 0 ∧ x = 4 * k ∧ y = k ∧ z = 2 * k := by
  sorry

end ratio_xyz_l3630_363038


namespace nested_sqrt_value_l3630_363090

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_sqrt_value_l3630_363090


namespace ellipse_tangent_and_normal_l3630_363042

noncomputable def ellipse (t : ℝ) : ℝ × ℝ := (4 * Real.cos t, 3 * Real.sin t)

theorem ellipse_tangent_and_normal (t : ℝ) :
  let (x₀, y₀) := ellipse (π/3)
  let tangent_slope := -(3 * Real.cos (π/3)) / (4 * Real.sin (π/3))
  let normal_slope := -1 / tangent_slope
  (∀ x y, y - y₀ = tangent_slope * (x - x₀) ↔ y = -Real.sqrt 3 / 4 * x + 2 * Real.sqrt 3) ∧
  (∀ x y, y - y₀ = normal_slope * (x - x₀) ↔ y = 4 / Real.sqrt 3 * x - 7 * Real.sqrt 3 / 3) :=
by sorry

end ellipse_tangent_and_normal_l3630_363042


namespace net_pay_rate_l3630_363052

-- Define the given conditions
def travel_time : ℝ := 3
def speed : ℝ := 60
def fuel_efficiency : ℝ := 30
def pay_rate : ℝ := 0.60
def gas_price : ℝ := 2.50

-- Define the theorem
theorem net_pay_rate : 
  let distance := travel_time * speed
  let gas_used := distance / fuel_efficiency
  let earnings := distance * pay_rate
  let gas_cost := gas_used * gas_price
  let net_earnings := earnings - gas_cost
  net_earnings / travel_time = 31 := by sorry

end net_pay_rate_l3630_363052


namespace quadratic_equation_real_roots_l3630_363002

theorem quadratic_equation_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ + (m - 1) = 0 ∧ x₂^2 - m*x₂ + (m - 1) = 0 := by
  sorry

end quadratic_equation_real_roots_l3630_363002


namespace managers_salary_l3630_363039

/-- Proves that the manager's salary is 11500 given the conditions of the problem -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 24 →
  avg_salary = 1500 →
  salary_increase = 400 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase : ℕ) = 11500 :=
by sorry

end managers_salary_l3630_363039


namespace jason_cantaloupes_l3630_363089

theorem jason_cantaloupes (total keith fred : ℕ) (h1 : total = 65) (h2 : keith = 29) (h3 : fred = 16) :
  total - keith - fred = 20 := by
  sorry

end jason_cantaloupes_l3630_363089


namespace area_transformation_l3630_363030

-- Define a function representing the area between a curve and the x-axis
noncomputable def area_between_curve_and_x_axis (f : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem area_transformation (g : ℝ → ℝ) 
  (h : area_between_curve_and_x_axis g = 8) : 
  area_between_curve_and_x_axis (λ x => 4 * g (x + 3)) = 32 := by
  sorry

end area_transformation_l3630_363030


namespace D_72_l3630_363062

/-- D(n) is the number of ways to express n as a product of integers greater than 1, considering order as distinct -/
def D (n : ℕ) : ℕ := sorry

/-- The prime factorization of 72 -/
def prime_factorization_72 : List (ℕ × ℕ) := [(2, 3), (3, 2)]

theorem D_72 : D 72 = 22 := by sorry

end D_72_l3630_363062


namespace triangle_max_area_l3630_363070

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (2 * a + b) * Real.cos C + c * Real.cos B = 0 →
  c = 6 →
  ∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 ∧
    ∀ (S' : ℝ), S' = 1/2 * a * b * Real.sin C → S' ≤ S :=
by sorry

end triangle_max_area_l3630_363070


namespace total_sequences_value_l3630_363063

/-- The number of students in the first class -/
def students_class1 : ℕ := 12

/-- The number of students in the second class -/
def students_class2 : ℕ := 13

/-- The number of meetings per week for each class -/
def meetings_per_week : ℕ := 3

/-- The total number of different sequences of students solving problems for both classes in a week -/
def total_sequences : ℕ := (students_class1 * students_class2) ^ meetings_per_week

theorem total_sequences_value : total_sequences = 3796416 := by sorry

end total_sequences_value_l3630_363063


namespace watson_class_composition_l3630_363048

/-- Represents the number of students in each grade level in Ms. Watson's class -/
structure ClassComposition where
  kindergartners : Nat
  first_graders : Nat
  second_graders : Nat

/-- The total number of students in Ms. Watson's class -/
def total_students (c : ClassComposition) : Nat :=
  c.kindergartners + c.first_graders + c.second_graders

/-- Theorem stating that given the conditions of Ms. Watson's class, 
    there are 4 second graders -/
theorem watson_class_composition :
  ∃ (c : ClassComposition),
    c.kindergartners = 14 ∧
    c.first_graders = 24 ∧
    total_students c = 42 ∧
    c.second_graders = 4 := by
  sorry

end watson_class_composition_l3630_363048


namespace cube_root_sum_reciprocal_cube_l3630_363008

theorem cube_root_sum_reciprocal_cube (x : ℝ) : 
  x = Real.rpow 4 (1/3) + Real.rpow 2 (1/3) + 1 → (1 + 1/x)^3 = 2 := by
  sorry

end cube_root_sum_reciprocal_cube_l3630_363008


namespace floor_cube_difference_l3630_363023

theorem floor_cube_difference : 
  ⌊(2007^3 : ℝ) / (2005 * 2006) - (2008^3 : ℝ) / (2006 * 2007)⌋ = -4 := by
  sorry

end floor_cube_difference_l3630_363023


namespace adams_initial_money_l3630_363041

theorem adams_initial_money (initial_amount : ℚ) : 
  (initial_amount - 21) / 21 = 10 / 3 → initial_amount = 91 :=
by sorry

end adams_initial_money_l3630_363041


namespace smallest_even_five_digit_number_tens_place_l3630_363046

def Digits : Finset ℕ := {1, 2, 3, 5, 8}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  (∀ d : ℕ, d ∈ Digits → (n.digits 10).count d = 1) ∧
  (∀ d : ℕ, d ∉ Digits → (n.digits 10).count d = 0)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem smallest_even_five_digit_number_tens_place :
  ∃ n : ℕ, is_valid_number n ∧ is_even n ∧
    (∀ m : ℕ, is_valid_number m ∧ is_even m → n ≤ m) ∧
    tens_digit n = 8 :=
sorry

end smallest_even_five_digit_number_tens_place_l3630_363046


namespace correct_decision_probability_l3630_363057

theorem correct_decision_probability (p : ℝ) (h : p = 0.8) :
  let n := 3  -- number of consultants
  let prob_two_correct := Nat.choose n 2 * p^2 * (1 - p)
  let prob_three_correct := Nat.choose n 3 * p^3
  prob_two_correct + prob_three_correct = 0.896 :=
sorry

end correct_decision_probability_l3630_363057


namespace subtraction_of_decimals_l3630_363078

theorem subtraction_of_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtraction_of_decimals_l3630_363078


namespace fabric_cutting_l3630_363076

theorem fabric_cutting (fabric_length fabric_width dress_length dress_width : ℕ) 
  (h1 : fabric_length = 140)
  (h2 : fabric_width = 75)
  (h3 : dress_length = 45)
  (h4 : dress_width = 26)
  : ∃ (n : ℕ), n ≥ 8 ∧ n * dress_length * dress_width ≤ fabric_length * fabric_width := by
  sorry

end fabric_cutting_l3630_363076


namespace probability_theorem_l3630_363087

/-- Represents the number of students in each language class and their combinations --/
structure LanguageEnrollment where
  total : ℕ
  french : ℕ
  spanish : ℕ
  german : ℕ
  french_spanish : ℕ
  french_german : ℕ
  spanish_german : ℕ
  all_three : ℕ

/-- Calculates the probability of selecting at least one student from each language class --/
def probability_all_languages (e : LanguageEnrollment) : ℚ :=
  let total_combinations := (e.total.choose 3)
  let favorable_outcomes := 
    (e.french - e.french_spanish - e.french_german + e.all_three) * 
    (e.spanish - e.french_spanish - e.spanish_german + e.all_three) * 
    (e.german - e.french_german - e.spanish_german + e.all_three) +
    e.french_spanish * (e.german - e.french_german - e.spanish_german + e.all_three) +
    e.french_german * (e.spanish - e.french_spanish - e.spanish_german + e.all_three) +
    e.spanish_german * (e.french - e.french_spanish - e.french_german + e.all_three)
  favorable_outcomes / total_combinations

/-- The main theorem to prove --/
theorem probability_theorem (e : LanguageEnrollment) 
  (h1 : e.total = 40)
  (h2 : e.french = 26)
  (h3 : e.spanish = 29)
  (h4 : e.german = 12)
  (h5 : e.french_spanish = 9)
  (h6 : e.french_german = 9)
  (h7 : e.spanish_german = 9)
  (h8 : e.all_three = 2) :
  probability_all_languages e = 76 / 4940 := by
  sorry

end probability_theorem_l3630_363087


namespace monthly_income_calculation_l3630_363095

/-- Proves that given the spending percentages and savings amount, the monthly income is 40000 --/
theorem monthly_income_calculation (income : ℝ) 
  (household_percent : income * (45 / 100) = income * 0.45)
  (clothes_percent : income * (25 / 100) = income * 0.25)
  (medicines_percent : income * (7.5 / 100) = income * 0.075)
  (savings : income * (1 - 0.45 - 0.25 - 0.075) = 9000) :
  income = 40000 := by
  sorry

end monthly_income_calculation_l3630_363095


namespace sphere_in_cylinder_ratio_l3630_363051

theorem sphere_in_cylinder_ratio : 
  ∀ (r h : ℝ),
  r > 0 →
  h > 0 →
  (4 / 3 * π * r^3) * 2 = π * r^2 * h →
  h / (2 * r) = 4 / 3 :=
λ r h hr hh vol_eq ↦ by
  sorry

end sphere_in_cylinder_ratio_l3630_363051


namespace households_using_both_brands_l3630_363037

/-- Proves that the number of households using both brands of soap is 25 --/
theorem households_using_both_brands (total : ℕ) (neither : ℕ) (only_a : ℕ) (h1 : total = 240) (h2 : neither = 80) (h3 : only_a = 60) : 
  ∃ (both : ℕ), both = 25 ∧ total = neither + only_a + both + 3 * both := by
  sorry

end households_using_both_brands_l3630_363037


namespace min_people_for_valid_arrangement_l3630_363025

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any additional
    person must sit next to someone already seated. -/
def valid_arrangement (table : CircularTable) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ table.total_chairs →
    ∃ i j, i ≠ j ∧
           i ≤ table.seated_people ∧
           j ≤ table.seated_people ∧
           (k = i ∨ k = j ∨ (i < k ∧ k < j) ∨ (j < k ∧ k < i) ∨ (k < i ∧ j < k) ∨ (k < j ∧ i < k))

/-- The main theorem stating that 20 is the minimum number of people required
    for a valid arrangement on a table with 80 chairs. -/
theorem min_people_for_valid_arrangement :
  ∀ n : ℕ, n < 20 →
    ¬(valid_arrangement { total_chairs := 80, seated_people := n }) ∧
    (valid_arrangement { total_chairs := 80, seated_people := 20 }) := by
  sorry

end min_people_for_valid_arrangement_l3630_363025


namespace opposite_of_negative_two_thirds_l3630_363019

-- Define the opposite of a rational number
def opposite (x : ℚ) : ℚ := -x

-- Theorem statement
theorem opposite_of_negative_two_thirds : 
  opposite (-2/3) = 2/3 := by sorry

end opposite_of_negative_two_thirds_l3630_363019


namespace simplify_polynomial_l3630_363072

theorem simplify_polynomial (y : ℝ) :
  (2*y - 1) * (4*y^10 + 2*y^9 + 4*y^8 + 2*y^7) = 8*y^11 + 6*y^9 - 2*y^7 := by
  sorry

end simplify_polynomial_l3630_363072


namespace negative_calculation_l3630_363058

theorem negative_calculation : 
  ((-4) + (-5) < 0) ∧ 
  ((-4) - (-5) ≥ 0) ∧ 
  ((-4) * (-5) ≥ 0) ∧ 
  ((-4) / (-5) ≥ 0) := by
  sorry

end negative_calculation_l3630_363058


namespace total_material_needed_l3630_363022

-- Define the dimensions of the tablecloth
def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54

-- Define the dimensions of a napkin
def napkin_length : ℕ := 6
def napkin_width : ℕ := 7

-- Define the number of napkins
def num_napkins : ℕ := 8

-- Theorem to prove
theorem total_material_needed :
  tablecloth_length * tablecloth_width + num_napkins * napkin_length * napkin_width = 5844 :=
by sorry

end total_material_needed_l3630_363022


namespace k_values_l3630_363067

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (1 - p.2) / (1 + p.1) = 3}

def B (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 3}

theorem k_values (k : ℝ) : A ∩ B k = ∅ → k = 2 ∨ k = -3 := by
  sorry

end k_values_l3630_363067


namespace situps_total_l3630_363026

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie does sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 
  barney_situps * barney_minutes + 
  carrie_situps * carrie_minutes + 
  jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end situps_total_l3630_363026


namespace opposite_of_negative_2023_l3630_363082

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = 2023 := by
  sorry

end opposite_of_negative_2023_l3630_363082


namespace nancy_hula_hoop_time_l3630_363091

theorem nancy_hula_hoop_time (morgan_time casey_time nancy_time : ℕ) : 
  morgan_time = 21 →
  morgan_time = 3 * casey_time →
  nancy_time = casey_time + 3 →
  nancy_time = 10 := by
sorry

end nancy_hula_hoop_time_l3630_363091


namespace solution_set_f_gt_3_range_of_a_l3630_363094

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Define the function g (although not used in the proof)
def g (x : ℝ) : ℝ := |2*x - 1| + 3

-- Theorem 1: The solution set of f(x) > 3 is (0, +∞)
theorem solution_set_f_gt_3 : 
  {x : ℝ | f x > 3} = {x : ℝ | x > 0} := by sorry

-- Theorem 2: If f(x) + 1 < 4^a - 5×2^a has a solution, then a < 0 or a > 2
theorem range_of_a (a : ℝ) : 
  (∃ x, f x + 1 < 4^a - 5*2^a) → (a < 0 ∨ a > 2) := by sorry

end solution_set_f_gt_3_range_of_a_l3630_363094


namespace photo_arrangement_count_l3630_363005

theorem photo_arrangement_count :
  let total_people : ℕ := 7
  let adjacent_pair : ℕ := 1  -- A and B treated as one unit
  let separated_pair : ℕ := 2  -- C and D
  let other_people : ℕ := total_people - adjacent_pair - separated_pair
  
  let total_elements : ℕ := adjacent_pair + other_people + 1
  let adjacent_pair_arrangements : ℕ := 2  -- A and B can switch
  let spaces_for_separated : ℕ := total_elements + 1

  (total_elements.factorial * adjacent_pair_arrangements * 
   (spaces_for_separated * (spaces_for_separated - 1))) = 960 := by
  sorry

end photo_arrangement_count_l3630_363005


namespace percentage_problem_l3630_363015

theorem percentage_problem (P : ℝ) : 
  (0.2 * 580 = (P / 100) * 120 + 80) → P = 30 := by
  sorry

end percentage_problem_l3630_363015


namespace poster_cost_l3630_363006

theorem poster_cost (initial_amount : ℕ) (notebook_cost : ℕ) (bookmark_cost : ℕ)
  (poster_count : ℕ) (leftover : ℕ) :
  initial_amount = 40 →
  notebook_cost = 12 →
  bookmark_cost = 4 →
  poster_count = 2 →
  leftover = 14 →
  (initial_amount - notebook_cost - bookmark_cost - leftover) / poster_count = 13 := by
  sorry

end poster_cost_l3630_363006


namespace house_market_value_l3630_363016

/-- Proves that the market value of a house is $500,000 given the specified conditions --/
theorem house_market_value : 
  ∀ (market_value selling_price revenue_per_person : ℝ),
  selling_price = market_value * 1.2 →
  selling_price = 4 * revenue_per_person →
  revenue_per_person * 0.9 = 135000 →
  market_value = 500000 := by
  sorry

end house_market_value_l3630_363016


namespace binary_101101_equals_45_l3630_363017

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_equals_45_l3630_363017


namespace peaches_in_basket_c_l3630_363066

theorem peaches_in_basket_c (total_baskets : ℕ) (avg_fruits : ℕ) 
  (fruits_a : ℕ) (fruits_b : ℕ) (fruits_d : ℕ) (fruits_e : ℕ) :
  total_baskets = 5 →
  avg_fruits = 25 →
  fruits_a = 15 →
  fruits_b = 30 →
  fruits_d = 25 →
  fruits_e = 35 →
  (total_baskets * avg_fruits) - (fruits_a + fruits_b + fruits_d + fruits_e) = 20 :=
by
  sorry

end peaches_in_basket_c_l3630_363066


namespace coefficient_x_squared_eq_40_l3630_363000

/-- The coefficient of x^2 in the expansion of (1+2x)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 2) * 2^2

/-- Theorem stating that the coefficient of x^2 in (1+2x)^5 is 40 -/
theorem coefficient_x_squared_eq_40 : coefficient_x_squared = 40 := by
  sorry

end coefficient_x_squared_eq_40_l3630_363000
