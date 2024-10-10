import Mathlib

namespace parallelogram_height_l2714_271455

theorem parallelogram_height 
  (area : ℝ) 
  (base : ℝ) 
  (h1 : area = 308) 
  (h2 : base = 22) : 
  area / base = 14 := by
sorry

end parallelogram_height_l2714_271455


namespace range_of_m_for_two_zeros_l2714_271474

/-- Given a function f and a real number m, g is defined as their sum -/
def g (f : ℝ → ℝ) (m : ℝ) : ℝ → ℝ := λ x ↦ f x + m

/-- The main theorem -/
theorem range_of_m_for_two_zeros (ω : ℝ) (h_ω_pos : ω > 0) 
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 2 * (Real.cos (ω * x / 2))^2) 
  (h_period : ∀ x, f (x + 2 * Real.pi / 3) = f x) :
  {m : ℝ | ∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ z₁ ∈ Set.Icc 0 (Real.pi / 3) ∧ z₂ ∈ Set.Icc 0 (Real.pi / 3) ∧ 
    g f m z₁ = 0 ∧ g f m z₂ = 0 ∧ ∀ z ∈ Set.Icc 0 (Real.pi / 3), g f m z = 0 → z = z₁ ∨ z = z₂} = 
  Set.Ioc (-3) (-2) :=
sorry

end range_of_m_for_two_zeros_l2714_271474


namespace question_selection_probability_l2714_271481

/-- The probability of selecting an algebra question first and a geometry question second -/
def prob_AB (total_questions : ℕ) (algebra_questions : ℕ) (geometry_questions : ℕ) : ℚ :=
  (algebra_questions : ℚ) / total_questions * (geometry_questions : ℚ) / (total_questions - 1)

/-- The probability of selecting a geometry question second given an algebra question was selected first -/
def prob_B_given_A (total_questions : ℕ) (algebra_questions : ℕ) (geometry_questions : ℕ) : ℚ :=
  (geometry_questions : ℚ) / (total_questions - 1)

theorem question_selection_probability :
  let total_questions := 5
  let algebra_questions := 2
  let geometry_questions := 3
  prob_AB total_questions algebra_questions geometry_questions = 3 / 10 ∧
  prob_B_given_A total_questions algebra_questions geometry_questions = 3 / 4 := by
  sorry

end question_selection_probability_l2714_271481


namespace keith_seashells_l2714_271469

/-- Proves the number of seashells Keith found given the problem conditions -/
theorem keith_seashells (mary_shells : ℕ) (total_shells : ℕ) (cracked_shells : ℕ) :
  mary_shells = 2 →
  total_shells = 7 →
  cracked_shells = 9 →
  total_shells - mary_shells = 5 :=
by sorry

end keith_seashells_l2714_271469


namespace correct_divisor_problem_l2714_271446

theorem correct_divisor_problem (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_answer : ℕ) (correct_answer : ℕ) :
  dividend = incorrect_divisor * incorrect_answer →
  dividend / correct_answer = 36 →
  incorrect_divisor = 48 →
  incorrect_answer = 24 →
  correct_answer = 32 →
  36 = dividend / correct_answer :=
by sorry

end correct_divisor_problem_l2714_271446


namespace art_students_count_l2714_271471

/-- Represents the number of students taking art in a high school -/
def students_taking_art (total students_taking_music students_taking_both students_taking_neither : ℕ) : ℕ :=
  total - students_taking_music - students_taking_neither + students_taking_both

/-- Theorem stating that 10 students are taking art given the conditions -/
theorem art_students_count :
  students_taking_art 500 30 10 470 = 10 := by
  sorry

end art_students_count_l2714_271471


namespace power_function_property_l2714_271486

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) (h2 : f 4 = 2) : f 9 = 3 := by
  sorry

end power_function_property_l2714_271486


namespace smallest_number_with_five_primes_including_even_l2714_271430

def is_prime (n : ℕ) : Prop := sorry

def has_five_different_prime_factors (n : ℕ) : Prop := sorry

def has_even_prime_factor (n : ℕ) : Prop := sorry

theorem smallest_number_with_five_primes_including_even :
  ∀ n : ℕ, 
    has_five_different_prime_factors n ∧ 
    has_even_prime_factor n → 
    n ≥ 2310 :=
sorry

end smallest_number_with_five_primes_including_even_l2714_271430


namespace min_additional_marbles_for_lisa_l2714_271404

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem min_additional_marbles_for_lisa : min_additional_marbles 12 34 = 44 := by
  sorry

end min_additional_marbles_for_lisa_l2714_271404


namespace distance_between_opposite_faces_of_unit_octahedron_l2714_271458

/-- A regular octahedron is a polyhedron with 8 faces, where each face is an equilateral triangle -/
structure RegularOctahedron where
  side_length : ℝ

/-- The distance between two opposite faces of a regular octahedron -/
def distance_between_opposite_faces (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem: In a regular octahedron with side length 1, the distance between two opposite faces is √6/3 -/
theorem distance_between_opposite_faces_of_unit_octahedron :
  let o : RegularOctahedron := ⟨1⟩
  distance_between_opposite_faces o = Real.sqrt 6 / 3 := by
  sorry

end distance_between_opposite_faces_of_unit_octahedron_l2714_271458


namespace line_equation_through_point_parallel_to_line_l2714_271475

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_through_point_parallel_to_line 
  (given_line : Line) 
  (point : Point) 
  (h_point : point.x = 2 ∧ point.y = 1) 
  (h_given_line : given_line.a = 2 ∧ given_line.b = -1 ∧ given_line.c = 2) :
  ∃ (result_line : Line), 
    result_line.a = 2 ∧ 
    result_line.b = -1 ∧ 
    result_line.c = -3 ∧
    parallel result_line given_line ∧
    on_line point result_line :=
  sorry

end line_equation_through_point_parallel_to_line_l2714_271475


namespace gain_percentage_l2714_271424

/-- 
If the cost price of 50 articles equals the selling price of 46 articles, 
then the gain percentage is (1/11.5) * 100.
-/
theorem gain_percentage (C S : ℝ) (h : 50 * C = 46 * S) : 
  (S - C) / C * 100 = (1 / 11.5) * 100 := by
  sorry

end gain_percentage_l2714_271424


namespace circle_tangent_to_line_circle_through_origin_l2714_271480

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem for part I
theorem circle_tangent_to_line :
  ∃ m : ℝ, ∀ x y : ℝ, circle_C x y m ∧ line_l x y →
    (x + 1/2)^2 + (y - 3)^2 = 1/8 :=
sorry

-- Theorem for part II
theorem circle_through_origin :
  ∃ m : ℝ, m = -3/2 ∧
    (∀ x1 y1 x2 y2 : ℝ,
      (circle_C x1 y1 m ∧ line_l x1 y1) ∧
      (circle_C x2 y2 m ∧ line_l x2 y2) ∧
      x1 ≠ x2 →
      x1 * x2 + y1 * y2 = 0) :=
sorry

end circle_tangent_to_line_circle_through_origin_l2714_271480


namespace square_area_given_circle_l2714_271463

-- Define the area of the circle
def circle_area : ℝ := 39424

-- Define the relationship between square perimeter and circle radius
def square_perimeter_equals_circle_radius (square_side : ℝ) (circle_radius : ℝ) : Prop :=
  4 * square_side = circle_radius

-- Theorem statement
theorem square_area_given_circle (square_side : ℝ) (circle_radius : ℝ) :
  circle_area = Real.pi * circle_radius^2 →
  square_perimeter_equals_circle_radius square_side circle_radius →
  square_side^2 = 784 := by
  sorry

end square_area_given_circle_l2714_271463


namespace base_conversion_and_sum_l2714_271438

-- Define the value of 537 in base 8
def base_8_value : ℕ := 5 * 8^2 + 3 * 8^1 + 7 * 8^0

-- Define the value of 1C2E in base 16, where C = 12 and E = 14
def base_16_value : ℕ := 1 * 16^3 + 12 * 16^2 + 2 * 16^1 + 14 * 16^0

-- Theorem statement
theorem base_conversion_and_sum :
  base_8_value + base_16_value = 7565 := by
  sorry

end base_conversion_and_sum_l2714_271438


namespace solution_set_part1_range_of_a_part2_l2714_271498

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = Set.Ioo (-2) (-2/3) := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ x ∈ Set.Icc 2 3, (∀ a : ℝ, f a x > 0) → a ∈ Set.Ioo (-5/2) (-2) := by sorry

end solution_set_part1_range_of_a_part2_l2714_271498


namespace transformed_curve_equation_l2714_271453

/-- Given a curve and a scaling transformation, prove the equation of the transformed curve -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  y = (1/3) * Real.sin (2 * x) →  -- Original curve equation
  x' = 2 * x →                    -- x-scaling
  y' = 3 * y →                    -- y-scaling
  y' = Real.sin x' :=             -- Transformed curve equation
by sorry

end transformed_curve_equation_l2714_271453


namespace larger_cuboid_width_l2714_271411

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- The smaller cuboid -/
def small_cuboid : Cuboid := { length := 5, width := 6, height := 3 }

/-- The larger cuboid -/
def large_cuboid (w : ℝ) : Cuboid := { length := 18, width := w, height := 2 }

/-- The number of smaller cuboids that can be formed from the larger cuboid -/
def num_small_cuboids : ℕ := 6

theorem larger_cuboid_width :
  ∃ w : ℝ, volume (large_cuboid w) = num_small_cuboids * volume small_cuboid ∧ w = 15 := by
  sorry

end larger_cuboid_width_l2714_271411


namespace trapezoid_ab_length_l2714_271494

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 280
  sum_sides : ab + cd = 280
  -- The ratio of areas is 5:2
  ratio_constraint : area_ratio = 5 / 2

/-- Theorem: In a trapezoid with given properties, AB = 200 -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 200 := by
  sorry


end trapezoid_ab_length_l2714_271494


namespace min_base_sum_l2714_271444

theorem min_base_sum : 
  ∃ (a b : ℕ+), 
    (3 * a.val + 5 = 4 * b.val + 2) ∧ 
    (∀ (c d : ℕ+), (3 * c.val + 5 = 4 * d.val + 2) → (a.val + b.val ≤ c.val + d.val)) ∧
    (a.val + b.val = 13) := by
  sorry

end min_base_sum_l2714_271444


namespace gasoline_cost_calculation_l2714_271443

/-- Represents the cost of gasoline per liter -/
def gasoline_cost : ℝ := sorry

/-- Represents the trip distance one way in kilometers -/
def one_way_distance : ℝ := 150

/-- Represents the cost of the first car rental option per day, excluding gasoline -/
def first_option_cost : ℝ := 50

/-- Represents the cost of the second car rental option per day, including gasoline -/
def second_option_cost : ℝ := 90

/-- Represents the distance a liter of gasoline can cover in kilometers -/
def km_per_liter : ℝ := 15

/-- Represents the amount saved by choosing the first option over the second option -/
def savings : ℝ := 22

theorem gasoline_cost_calculation : gasoline_cost = 3.4 := by
  sorry

end gasoline_cost_calculation_l2714_271443


namespace travel_distance_proof_l2714_271447

def speed_limit : ℝ := 60
def speed_above_limit : ℝ := 15
def travel_time : ℝ := 2

theorem travel_distance_proof :
  let actual_speed := speed_limit + speed_above_limit
  actual_speed * travel_time = 150 := by sorry

end travel_distance_proof_l2714_271447


namespace product_of_roots_l2714_271433

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 22 → 
  ∃ y : ℝ, (y + 3) * (y - 5) = 22 ∧ x * y = -37 := by
sorry

end product_of_roots_l2714_271433


namespace molecular_weight_C4H10_is_58_12_l2714_271496

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The number of carbon atoms in C4H10 -/
def carbon_count : ℕ := 4

/-- The number of hydrogen atoms in C4H10 -/
def hydrogen_count : ℕ := 10

/-- The molecular weight of C4H10 in atomic mass units (amu) -/
def molecular_weight_C4H10 : ℝ := carbon_weight * carbon_count + hydrogen_weight * hydrogen_count

/-- Theorem stating that the molecular weight of C4H10 is 58.12 amu -/
theorem molecular_weight_C4H10_is_58_12 : 
  molecular_weight_C4H10 = 58.12 := by sorry

end molecular_weight_C4H10_is_58_12_l2714_271496


namespace gcd_polynomial_and_multiple_l2714_271485

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  36000 ∣ x → 
  Nat.gcd ((5*x + 3)*(11*x + 2)*(6*x + 7)*(3*x + 8) : ℤ).natAbs x.natAbs = 144 := by
sorry

end gcd_polynomial_and_multiple_l2714_271485


namespace swim_club_percentage_passed_l2714_271448

/-- The percentage of swim club members who have passed the lifesaving test -/
def percentage_passed (total_members : ℕ) (not_passed_with_course : ℕ) (not_passed_without_course : ℕ) : ℚ :=
  1 - (not_passed_with_course + not_passed_without_course : ℚ) / total_members

theorem swim_club_percentage_passed :
  percentage_passed 100 40 30 = 30 / 100 := by
  sorry

end swim_club_percentage_passed_l2714_271448


namespace total_amount_spent_l2714_271409

/-- Calculates the total amount spent on pencils, cucumbers, and notebooks given specific conditions --/
theorem total_amount_spent 
  (initial_cost : ℝ)
  (pencil_discount : ℝ)
  (notebook_discount : ℝ)
  (pencil_tax : ℝ)
  (cucumber_tax : ℝ)
  (cucumber_count : ℕ)
  (notebook_count : ℕ)
  (h1 : initial_cost = 20)
  (h2 : pencil_discount = 0.2)
  (h3 : notebook_discount = 0.3)
  (h4 : pencil_tax = 0.05)
  (h5 : cucumber_tax = 0.1)
  (h6 : cucumber_count = 100)
  (h7 : notebook_count = 25) :
  (cucumber_count / 2 : ℝ) * (initial_cost * (1 - pencil_discount) * (1 + pencil_tax)) +
  (cucumber_count : ℝ) * (initial_cost * (1 + cucumber_tax)) +
  (notebook_count : ℝ) * (initial_cost * (1 - notebook_discount)) = 3390 :=
by sorry

end total_amount_spent_l2714_271409


namespace binomial_product_l2714_271432

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 7) = 4 * x^2 - 25 * x - 21 := by
  sorry

end binomial_product_l2714_271432


namespace isosceles_triangle_n_value_l2714_271479

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ (side1 = side3 ∧ side2 ≠ side1) ∨ (side2 = side3 ∧ side1 ≠ side2)

/-- The quadratic equation x^2 - 8x + n = 0 -/
def quadratic_equation (x n : ℝ) : Prop :=
  x^2 - 8*x + n = 0

/-- Theorem statement -/
theorem isosceles_triangle_n_value :
  ∀ (t : IsoscelesTriangle) (n : ℝ),
    ((t.side1 = 3 ∨ t.side2 = 3 ∨ t.side3 = 3) ∧
     (quadratic_equation t.side1 n ∧ quadratic_equation t.side2 n) ∨
     (quadratic_equation t.side1 n ∧ quadratic_equation t.side3 n) ∨
     (quadratic_equation t.side2 n ∧ quadratic_equation t.side3 n)) →
    n = 15 ∨ n = 16 := by
  sorry

end isosceles_triangle_n_value_l2714_271479


namespace missing_files_l2714_271436

/-- Proves that the number of missing files is 15 --/
theorem missing_files (total_files : ℕ) (afternoon_files : ℕ) : 
  total_files = 60 → 
  afternoon_files = 15 → 
  total_files - (total_files / 2 + afternoon_files) = 15 := by
  sorry

end missing_files_l2714_271436


namespace solution_set_f_gt_g_range_of_a_l2714_271422

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := |2*x - 2|

-- Theorem for the solution set of f(x) > g(x)
theorem solution_set_f_gt_g :
  {x : ℝ | f x > g x} = {x : ℝ | 2/3 < x ∧ x < 2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, 2 * f x + g x > a * x + 1} = {a : ℝ | -4 ≤ a ∧ a < 1} := by sorry

end solution_set_f_gt_g_range_of_a_l2714_271422


namespace scaled_tetrahedron_volume_ratio_l2714_271437

-- Define a regular tetrahedron
def RegularTetrahedron : Type := Unit

-- Define a function to scale down coordinates
def scaleDown (t : RegularTetrahedron) : RegularTetrahedron := sorry

-- Define a function to calculate the volume of a tetrahedron
def volume (t : RegularTetrahedron) : ℝ := sorry

-- Theorem statement
theorem scaled_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) : 
  volume (scaleDown t) / volume t = 1 / 8 := by sorry

end scaled_tetrahedron_volume_ratio_l2714_271437


namespace complex_fraction_sum_l2714_271490

theorem complex_fraction_sum : (2 + 2 * Complex.I) / Complex.I + (1 + Complex.I) / (1 - Complex.I) = 2 - Complex.I := by
  sorry

end complex_fraction_sum_l2714_271490


namespace jade_transactions_jade_transactions_proof_l2714_271499

/-- Proves that Jade handled 80 transactions given the specified conditions. -/
theorem jade_transactions : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mabel_transactions anthony_transactions cal_transactions jade_transactions =>
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    cal_transactions = anthony_transactions * 2 / 3 →
    jade_transactions = cal_transactions + 14 →
    jade_transactions = 80

/-- Proof of the theorem -/
theorem jade_transactions_proof : jade_transactions 90 99 66 80 := by
  sorry

end jade_transactions_jade_transactions_proof_l2714_271499


namespace distance_equals_speed_times_time_l2714_271460

/-- The distance between Emily's house and Timothy's house -/
def distance : ℝ := 10

/-- Emily's speed in miles per hour -/
def speed : ℝ := 5

/-- Time taken for Emily to reach Timothy's house in hours -/
def time : ℝ := 2

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_equals_speed_times_time : distance = speed * time := by
  sorry

end distance_equals_speed_times_time_l2714_271460


namespace overall_percentage_favor_l2714_271407

-- Define the given percentages
def starting_favor_percent : ℝ := 0.40
def experienced_favor_percent : ℝ := 0.70

-- Define the number of surveyed entrepreneurs
def num_starting : ℕ := 300
def num_experienced : ℕ := 500

-- Define the total number surveyed
def total_surveyed : ℕ := num_starting + num_experienced

-- Define the number in favor for each group
def num_starting_favor : ℝ := starting_favor_percent * num_starting
def num_experienced_favor : ℝ := experienced_favor_percent * num_experienced

-- Define the total number in favor
def total_favor : ℝ := num_starting_favor + num_experienced_favor

-- Theorem to prove
theorem overall_percentage_favor :
  (total_favor / total_surveyed) * 100 = 58.75 := by
  sorry

end overall_percentage_favor_l2714_271407


namespace monotonic_range_k_negative_range_k_l2714_271487

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

-- Theorem for part (1)
theorem monotonic_range_k (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, Monotone (g a b)) ↔ (k ≤ -2 ∨ k ≥ 6) :=
sorry

-- Theorem for part (2)
theorem negative_range_k (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x ∈ Set.Icc 1 2, g a b k x < 0) ↔ k > 9/2 :=
sorry

end monotonic_range_k_negative_range_k_l2714_271487


namespace jills_salary_l2714_271483

/-- Proves that given the conditions of Jill's income allocation, her net monthly salary is $3600 -/
theorem jills_salary (salary : ℝ) 
  (h1 : salary / 5 * 0.15 = 108) : salary = 3600 := by
  sorry

end jills_salary_l2714_271483


namespace perfect_square_condition_l2714_271473

theorem perfect_square_condition (m : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 4 * x^2 - m * x * y + 9 * y^2 = k^2) →
  m = 12 ∨ m = -12 := by
sorry

end perfect_square_condition_l2714_271473


namespace turban_count_is_one_l2714_271442

/-- The number of turbans given as part of the annual salary -/
def turban_count : ℕ := sorry

/-- The price of one turban in Rupees -/
def turban_price : ℕ := 30

/-- The base salary in Rupees -/
def base_salary : ℕ := 90

/-- The total annual salary in Rupees -/
def annual_salary : ℕ := base_salary + turban_count * turban_price

/-- The fraction of the year worked by the servant -/
def fraction_worked : ℚ := 3/4

/-- The amount received by the servant after 9 months in Rupees -/
def amount_received : ℕ := 60 + turban_price

theorem turban_count_is_one :
  (fraction_worked * annual_salary = amount_received) → turban_count = 1 := by
  sorry

end turban_count_is_one_l2714_271442


namespace min_employees_for_agency_l2714_271406

/-- Represents the number of employees needed for different pollution monitoring tasks -/
structure EmployeeRequirements where
  water : ℕ
  air : ℕ
  both : ℕ
  soil : ℕ

/-- Calculates the minimum number of employees needed given the requirements -/
def minEmployees (req : EmployeeRequirements) : ℕ :=
  req.water + req.air - req.both

/-- Theorem stating that given the specific requirements, 160 employees are needed -/
theorem min_employees_for_agency (req : EmployeeRequirements) 
  (h_water : req.water = 120)
  (h_air : req.air = 105)
  (h_both : req.both = 65)
  (h_soil : req.soil = 40)
  : minEmployees req = 160 := by
  sorry

#eval minEmployees { water := 120, air := 105, both := 65, soil := 40 }

end min_employees_for_agency_l2714_271406


namespace jelly_bean_probability_l2714_271450

-- Define the number of jelly beans for each color
def red_beans : ℕ := 7
def green_beans : ℕ := 9
def yellow_beans : ℕ := 8
def blue_beans : ℕ := 10
def orange_beans : ℕ := 5

-- Define the total number of jelly beans
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + orange_beans

-- Define the number of blue or orange jelly beans
def blue_or_orange_beans : ℕ := blue_beans + orange_beans

-- Theorem statement
theorem jelly_bean_probability : 
  (blue_or_orange_beans : ℚ) / (total_beans : ℚ) = 5 / 13 := by
  sorry

end jelly_bean_probability_l2714_271450


namespace seven_is_unique_solution_l2714_271413

/-- Product of all prime numbers less than n -/
def n_question_mark (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range n)).prod id

/-- The theorem stating that 7 is the only solution -/
theorem seven_is_unique_solution :
  ∃! (n : ℕ), n > 3 ∧ n_question_mark n = 2 * n + 16 :=
sorry

end seven_is_unique_solution_l2714_271413


namespace least_three_digit_multiple_of_13_l2714_271416

theorem least_three_digit_multiple_of_13 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 13 ∣ n → 104 ≤ n :=
by sorry

end least_three_digit_multiple_of_13_l2714_271416


namespace divisibility_property_l2714_271449

theorem divisibility_property (a b c d u : ℤ) 
  (h1 : u ∣ a * c) 
  (h2 : u ∣ b * c + a * d) 
  (h3 : u ∣ b * d) : 
  (u ∣ b * c) ∧ (u ∣ a * d) := by
  sorry

end divisibility_property_l2714_271449


namespace sum_of_factors_l2714_271451

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 10*x + 21 = (x + a) * (x + b)) →
  (∀ x, x^2 + 3*x - 88 = (x + b) * (x - c)) →
  a + b + c = 18 := by
sorry

end sum_of_factors_l2714_271451


namespace ab_value_l2714_271470

theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 + b^2 = 3) (h2 : a^4 + b^4 = 15/4) : a * b = Real.sqrt 42 / 4 := by
  sorry

end ab_value_l2714_271470


namespace solve_turtle_problem_l2714_271464

def turtle_problem (kristen_turtles : ℕ) (kris_ratio : ℚ) (trey_multiplier : ℕ) : Prop :=
  let kris_turtles : ℚ := kris_ratio * kristen_turtles
  let trey_turtles : ℚ := trey_multiplier * kris_turtles
  (trey_turtles - kristen_turtles : ℚ) = 9

theorem solve_turtle_problem :
  turtle_problem 12 (1/4) 7 := by
  sorry

end solve_turtle_problem_l2714_271464


namespace abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_negation_true_implies_converse_true_l2714_271457

-- Statement 1
theorem abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three :
  (∀ x : ℝ, |x - 1| < 2 → x < 3) ∧
  ¬(∀ x : ℝ, x < 3 → |x - 1| < 2) :=
sorry

-- Statement 2
theorem negation_true_implies_converse_true (P Q : Prop) :
  (¬(P → Q) → (Q → P)) :=
sorry

end abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_negation_true_implies_converse_true_l2714_271457


namespace max_inscribed_rectangle_area_l2714_271478

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-coordinate for a given x on the parabola -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: Maximum inscribed rectangle area in a parabola -/
theorem max_inscribed_rectangle_area
  (p : Parabola)
  (vertex_x : p.y_at 3 = -5)
  (point_on_parabola : p.y_at 5 = 15) :
  ∃ (area : ℝ), area = 10 ∧ 
  ∀ (rect_area : ℝ), 
    (∃ (x1 x2 : ℝ), 
      x1 < x2 ∧ 
      p.y_at x1 = 0 ∧ 
      p.y_at x2 = 0 ∧ 
      rect_area = (x2 - x1) * min (p.y_at ((x1 + x2) / 2)) 0) →
    rect_area ≤ area :=
by sorry

end max_inscribed_rectangle_area_l2714_271478


namespace bill_amount_is_1550_l2714_271456

/-- Calculates the amount of a bill given its true discount, due date, and interest rate. -/
def bill_amount (true_discount : ℚ) (months : ℚ) (annual_rate : ℚ) : ℚ :=
  let present_value := true_discount / (annual_rate * (months / 12) / (1 + annual_rate * (months / 12)))
  present_value + true_discount

/-- Theorem stating that the bill amount is 1550 given the specified conditions. -/
theorem bill_amount_is_1550 :
  bill_amount 150 9 (16 / 100) = 1550 := by
  sorry

end bill_amount_is_1550_l2714_271456


namespace f_monotone_increasing_on_neg_reals_l2714_271420

-- Define the function f(x) = -|x|
def f (x : ℝ) : ℝ := -abs x

-- State the theorem
theorem f_monotone_increasing_on_neg_reals :
  MonotoneOn f (Set.Iic 0) := by sorry

end f_monotone_increasing_on_neg_reals_l2714_271420


namespace max_third_side_length_l2714_271495

theorem max_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 10) :
  ∃ (x : ℕ), x ≤ 14 ∧
  ∀ (y : ℕ), (y : ℝ) < a + b ∧ (y : ℝ) > |a - b| → y ≤ x :=
by sorry

end max_third_side_length_l2714_271495


namespace point_in_first_quadrant_l2714_271435

/-- Given a complex number Z = 1 + i, prove that the point corresponding to 1/Z + Z 
    lies in the first quadrant. -/
theorem point_in_first_quadrant (Z : ℂ) (h : Z = 1 + Complex.I) : 
  let W := Z⁻¹ + Z
  0 < W.re ∧ 0 < W.im := by
  sorry

end point_in_first_quadrant_l2714_271435


namespace greatest_b_proof_l2714_271408

/-- The greatest integer b for which x^2 + bx + 17 ≠ 0 for all real x -/
def greatest_b : ℤ := 8

theorem greatest_b_proof :
  (∀ x : ℝ, x^2 + (greatest_b : ℝ) * x + 17 ≠ 0) ∧
  (∀ b : ℤ, b > greatest_b → ∃ x : ℝ, x^2 + (b : ℝ) * x + 17 = 0) :=
by sorry

#check greatest_b_proof

end greatest_b_proof_l2714_271408


namespace quadratic_two_distinct_roots_l2714_271462

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 + 6 * x + 3 = 0 ∧ 
   (k - 1) * y^2 + 6 * y + 3 = 0) ↔ 
  (k < 4 ∧ k ≠ 1) :=
by sorry

end quadratic_two_distinct_roots_l2714_271462


namespace xyz_product_absolute_value_l2714_271491

theorem xyz_product_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (heq1 : x + 1 / y = y + 1 / z)
  (heq2 : y + 1 / z = z + 1 / x + 1) :
  |x * y * z| = 1 := by
sorry

end xyz_product_absolute_value_l2714_271491


namespace red_peaches_per_basket_l2714_271476

theorem red_peaches_per_basket 
  (num_baskets : ℕ) 
  (green_per_basket : ℕ) 
  (total_peaches : ℕ) 
  (h1 : num_baskets = 15)
  (h2 : green_per_basket = 4)
  (h3 : total_peaches = 345) :
  (total_peaches - num_baskets * green_per_basket) / num_baskets = 19 := by
  sorry

end red_peaches_per_basket_l2714_271476


namespace cubic_polynomial_sum_range_l2714_271425

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_sum_range (a b c d : ℝ) (h_a : a ≠ 0) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 2 * f a b c d 2 = t ∧ 3 * f a b c d 3 = t ∧ 4 * f a b c d 4 = t) →
  (∃ y : ℝ, 0 < y ∧ y < 1 ∧ f a b c d 1 + f a b c d 5 = y) :=
by sorry

end cubic_polynomial_sum_range_l2714_271425


namespace quadratic_inequality_solution_condition_l2714_271489

theorem quadratic_inequality_solution_condition (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) :=
by sorry

end quadratic_inequality_solution_condition_l2714_271489


namespace prob_sum_equals_seven_ninths_l2714_271431

def total_balls : ℕ := 9
def black_balls : ℕ := 5
def white_balls : ℕ := 4

def P_A : ℚ := black_balls / total_balls
def P_B_given_A : ℚ := white_balls / (total_balls - 1)

theorem prob_sum_equals_seven_ninths :
  (P_A * P_B_given_A) + P_B_given_A = 7 / 9 :=
sorry

end prob_sum_equals_seven_ninths_l2714_271431


namespace max_value_of_a_l2714_271482

theorem max_value_of_a (x y a : ℝ) 
  (h1 : x > 1/3) 
  (h2 : y > 1) 
  (h3 : ∀ (x y : ℝ), x > 1/3 → y > 1 → 
    (9 * x^2) / (a^2 * (y-1)) + (y^2) / (a^2 * (3*x-1)) ≥ 1) : 
  a ≤ 2 * Real.sqrt 2 := by
sorry

end max_value_of_a_l2714_271482


namespace cube_volume_ratio_l2714_271472

/-- Calculates the number of whole cubes that fit along a given dimension -/
def cubesAlongDimension (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the volume of a rectangular box -/
def boxVolume (length width height : ℕ) : ℕ :=
  length * width * height

/-- Calculates the volume of a cube -/
def cubeVolume (size : ℕ) : ℕ :=
  size * size * size

/-- Calculates the total volume occupied by cubes in the box -/
def occupiedVolume (boxLength boxWidth boxHeight cubeSize : ℕ) : ℕ :=
  let numCubesLength := cubesAlongDimension boxLength cubeSize
  let numCubesWidth := cubesAlongDimension boxWidth cubeSize
  let numCubesHeight := cubesAlongDimension boxHeight cubeSize
  let totalCubes := numCubesLength * numCubesWidth * numCubesHeight
  totalCubes * cubeVolume cubeSize

theorem cube_volume_ratio (boxLength boxWidth boxHeight cubeSize : ℕ) 
  (h1 : boxLength = 4)
  (h2 : boxWidth = 7)
  (h3 : boxHeight = 8)
  (h4 : cubeSize = 2) :
  (occupiedVolume boxLength boxWidth boxHeight cubeSize : ℚ) / 
  (boxVolume boxLength boxWidth boxHeight : ℚ) = 6 / 7 := by
  sorry

end cube_volume_ratio_l2714_271472


namespace candy_cost_450_l2714_271417

/-- The cost of buying a specified number of chocolate candies. -/
def candy_cost (total_candies : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of buying 450 chocolate candies is $120, given that a box of 30 candies costs $8. -/
theorem candy_cost_450 : candy_cost 450 30 8 = 120 := by
  sorry

end candy_cost_450_l2714_271417


namespace missing_figure_proof_l2714_271497

theorem missing_figure_proof (x : ℝ) : (0.75 / 100) * x = 0.06 ↔ x = 8 := by sorry

end missing_figure_proof_l2714_271497


namespace range_of_x_when_a_is_one_range_of_a_l2714_271493

-- Define the conditions
def p (a x : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 5

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, p 1 x ∧ q x → 2 < x ∧ x < 4 := by sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, p a x → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p a x)) →
  5/4 < a ∧ a ≤ 2 := by sorry

#check range_of_x_when_a_is_one
#check range_of_a

end range_of_x_when_a_is_one_range_of_a_l2714_271493


namespace walnut_trees_remaining_l2714_271414

/-- The number of walnut trees remaining after removal -/
def remaining_trees (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

theorem walnut_trees_remaining : remaining_trees 6 4 = 2 := by
  sorry

end walnut_trees_remaining_l2714_271414


namespace no_self_referential_function_l2714_271401

theorem no_self_referential_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) := by
  sorry

end no_self_referential_function_l2714_271401


namespace matrix_product_proof_l2714_271454

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 1, 3, -1; 0, 5, 2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 4; -2, 0, 0; 3, 0, -2]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-7, -2, 14; -8, -1, 6; -4, 0, -4]

theorem matrix_product_proof : A * B = C := by sorry

end matrix_product_proof_l2714_271454


namespace smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5_l2714_271466

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if four consecutive natural numbers are all prime, false otherwise -/
def fourConsecutivePrimes (a b c d : ℕ) : Prop := 
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1

theorem smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5 :
  ∃ (a b c d : ℕ),
    fourConsecutivePrimes a b c d ∧
    a > 10 ∧
    (a + b + c + d) % 5 = 0 ∧
    (a + b + c + d = 60) ∧
    ∀ (w x y z : ℕ),
      fourConsecutivePrimes w x y z →
      w > 10 →
      (w + x + y + z) % 5 = 0 →
      (w + x + y + z) ≥ 60 :=
sorry

end smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5_l2714_271466


namespace simplify_fourth_roots_l2714_271402

theorem simplify_fourth_roots : Real.sqrt (Real.sqrt 81) - Real.sqrt (Real.sqrt 256) = -1 := by
  sorry

end simplify_fourth_roots_l2714_271402


namespace fraction_simplification_l2714_271418

theorem fraction_simplification :
  1 / (1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4 + 1 / (1/3)^5) = 1 / 360 := by
  sorry

end fraction_simplification_l2714_271418


namespace max_students_equal_distribution_l2714_271412

theorem max_students_equal_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ toys % students = 0 ∧
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ toys % n ≠ 0)) ↔
  (Nat.gcd pens toys = 41) :=
sorry

end max_students_equal_distribution_l2714_271412


namespace complex_fraction_equality_l2714_271405

theorem complex_fraction_equality : ∀ (z₁ z₂ : ℂ), 
  z₁ = -1 + 3*I ∧ z₂ = 1 + I → (z₁ + z₂) / (z₁ - z₂) = 1 - I :=
by
  sorry

end complex_fraction_equality_l2714_271405


namespace group_size_from_weight_change_l2714_271427

/-- The number of people in a group where replacing a 35 kg person with a 55 kg person
    increases the average weight by 2.5 kg is 8. -/
theorem group_size_from_weight_change (n : ℕ) : 
  (n : ℝ) * 2.5 = 55 - 35 → n = 8 := by
  sorry

end group_size_from_weight_change_l2714_271427


namespace star_minus_emilio_sum_equals_104_l2714_271428

def star_list := List.range 40 |>.map (· + 1)

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list := star_list.map replace_three_with_two

theorem star_minus_emilio_sum_equals_104 :
  star_list.sum - emilio_list.sum = 104 := by
  sorry

end star_minus_emilio_sum_equals_104_l2714_271428


namespace solution_satisfies_system_l2714_271421

theorem solution_satisfies_system :
  let solutions : List (ℝ × ℝ) := [(5, -3), (5, 3), (-Real.sqrt 118 / 2, 3 * Real.sqrt 2 / 2), (-Real.sqrt 118 / 2, -3 * Real.sqrt 2 / 2)]
  ∀ (x y : ℝ), (x, y) ∈ solutions →
    (x^2 + y^2 = 34 ∧ x - y + Real.sqrt ((x - y) / (x + y)) = 20 / (x + y)) := by
  sorry

end solution_satisfies_system_l2714_271421


namespace min_value_symmetric_circle_l2714_271423

/-- Given a circle and a line, if the circle is symmetric about the line,
    then the minimum value of 1/a + 2/b is 3 -/
theorem min_value_symmetric_circle (x y a b : ℝ) :
  x^2 + y^2 - 2*x - 4*y + 3 = 0 →
  a > 0 →
  b > 0 →
  a*x + b*y = 3 →
  (∃ (c : ℝ), c > 0 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a'*x + b'*y = 3 → 1/a' + 2/b' ≥ c) →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀*x + b₀*y = 3 ∧ 1/a₀ + 2/b₀ = 3) :=
by sorry


end min_value_symmetric_circle_l2714_271423


namespace cube_of_negative_product_l2714_271439

theorem cube_of_negative_product (a b : ℝ) :
  (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := by sorry

end cube_of_negative_product_l2714_271439


namespace shorties_eating_today_l2714_271467

/-- Represents the number of shorties who eat donuts every day -/
def daily_eaters : ℕ := 6

/-- Represents the number of shorties who eat donuts every other day -/
def bi_daily_eaters : ℕ := 8

/-- Represents the number of shorties who ate donuts yesterday -/
def yesterday_eaters : ℕ := 11

/-- Theorem stating that the number of shorties who will eat donuts today is 9 -/
theorem shorties_eating_today : 
  ∃ (today_eaters : ℕ), today_eaters = 9 ∧
  today_eaters = daily_eaters + (bi_daily_eaters - (yesterday_eaters - daily_eaters)) :=
by
  sorry


end shorties_eating_today_l2714_271467


namespace total_fish_l2714_271468

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 9) : 
  lilly_fish + rosy_fish = 19 := by
sorry

end total_fish_l2714_271468


namespace cubic_inequality_solution_l2714_271488

theorem cubic_inequality_solution (x : ℝ) :
  x^3 + x^2 - 4*x - 4 < 0 ↔ x < -2 ∨ (-1 < x ∧ x < 2) :=
sorry

end cubic_inequality_solution_l2714_271488


namespace cricket_average_score_l2714_271492

theorem cricket_average_score (matches1 matches2 : ℕ) (avg1 avg2 : ℚ) :
  matches1 = 10 →
  matches2 = 15 →
  avg1 = 60 →
  avg2 = 70 →
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 := by
  sorry

end cricket_average_score_l2714_271492


namespace common_chord_length_is_2_sqrt_5_l2714_271465

/-- Two circles C₁ and C₂ in a 2D plane -/
structure TwoCircles where
  /-- Center of circle C₁ -/
  center1 : ℝ × ℝ
  /-- Radius of circle C₁ -/
  radius1 : ℝ
  /-- Center of circle C₂ -/
  center2 : ℝ × ℝ
  /-- Radius of circle C₂ -/
  radius2 : ℝ

/-- The length of the common chord between two intersecting circles -/
def commonChordLength (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem: The length of the common chord between the given intersecting circles is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 :
  let circles : TwoCircles := {
    center1 := (2, 1),
    radius1 := Real.sqrt 10,
    center2 := (-6, -3),
    radius2 := Real.sqrt 50
  }
  commonChordLength circles = 2 * Real.sqrt 5 := by
  sorry

end common_chord_length_is_2_sqrt_5_l2714_271465


namespace simplify_expression_l2714_271400

theorem simplify_expression (x y : ℝ) : 20 * (x + y) - 19 * (y + x) = x + y := by
  sorry

end simplify_expression_l2714_271400


namespace complex_equation_sum_l2714_271403

theorem complex_equation_sum (a b : ℝ) :
  (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I → a + b = 1 := by
  sorry

end complex_equation_sum_l2714_271403


namespace quadratic_root_value_l2714_271461

theorem quadratic_root_value (v : ℝ) : 
  (8 * ((-26 - Real.sqrt 450) / 10)^2 + 26 * ((-26 - Real.sqrt 450) / 10) + v = 0) → 
  v = 113 / 16 := by
sorry

end quadratic_root_value_l2714_271461


namespace no_valid_bracelet_arrangement_l2714_271419

/-- The number of bracelets Elizabeth has -/
def n : ℕ := 100

/-- The number of bracelets Elizabeth wears each day -/
def k : ℕ := 3

/-- Represents a valid arrangement of bracelets -/
structure BraceletArrangement where
  days : ℕ
  worn : Fin days → Finset (Fin n)
  size_correct : ∀ d, (worn d).card = k
  all_pairs_once : ∀ i j, i < j → ∃! d, i ∈ worn d ∧ j ∈ worn d

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_bracelet_arrangement : ¬ ∃ arr : BraceletArrangement, True := by
  sorry

end no_valid_bracelet_arrangement_l2714_271419


namespace sample_mean_estimates_population_mean_l2714_271459

/-- A type to represent statistical populations -/
structure Population where
  mean : ℝ

/-- A type to represent samples from a population -/
structure Sample where
  mean : ℝ

/-- Predicate to determine if a sample mean is an estimate of a population mean -/
def is_estimate_of (s : Sample) (p : Population) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ |s.mean - p.mean| < ε

/-- Theorem stating that a sample mean is an estimate of the population mean -/
theorem sample_mean_estimates_population_mean (s : Sample) (p : Population) :
  is_estimate_of s p :=
sorry

end sample_mean_estimates_population_mean_l2714_271459


namespace quadratic_solution_l2714_271410

theorem quadratic_solution (b : ℤ) : 
  ((-5 : ℤ)^2 + b * (-5) - 35 = 0) → b = -2 := by
  sorry

end quadratic_solution_l2714_271410


namespace min_distance_curve_line_l2714_271440

theorem min_distance_curve_line (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (x y : ℝ), Real.log (x + 1) + y - 3 * x = 0 → 
  ∀ (u v : ℝ), 2 * u - v + Real.sqrt 5 = 0 → 
  (y - v)^2 + (x - u)^2 ≥ min_val :=
sorry

end min_distance_curve_line_l2714_271440


namespace square_of_105_l2714_271429

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end square_of_105_l2714_271429


namespace cylinder_prism_height_equality_l2714_271415

/-- The height of a cylinder is equal to the height of a rectangular prism 
    when they have the same volume and base area. -/
theorem cylinder_prism_height_equality 
  (V : ℝ) -- Volume of both shapes
  (A : ℝ) -- Base area of both shapes
  (h_cylinder : ℝ) -- Height of the cylinder
  (h_prism : ℝ) -- Height of the rectangular prism
  (h_cylinder_def : h_cylinder = V / A) -- Definition of cylinder height
  (h_prism_def : h_prism = V / A) -- Definition of prism height
  : h_cylinder = h_prism := by
  sorry

end cylinder_prism_height_equality_l2714_271415


namespace school_competition_selections_l2714_271441

theorem school_competition_selections (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  (n.choose m) * m.factorial = 60 := by
  sorry

end school_competition_selections_l2714_271441


namespace base12_addition_l2714_271434

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Digit12 to its corresponding natural number --/
def digit12ToNat (d : Digit12) : ℕ :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11
  | Digit12.C => 12

/-- Represents a number in base 12 --/
def Number12 := List Digit12

/-- Converts a Number12 to its corresponding natural number --/
def number12ToNat (n : Number12) : ℕ :=
  n.foldr (fun d acc => digit12ToNat d + 12 * acc) 0

/-- The theorem to be proved --/
theorem base12_addition :
  let n1 : Number12 := [Digit12.C, Digit12.D9, Digit12.D7]
  let n2 : Number12 := [Digit12.D2, Digit12.D6, Digit12.A]
  let result : Number12 := [Digit12.D3, Digit12.D4, Digit12.D1, Digit12.B]
  number12ToNat n1 + number12ToNat n2 = number12ToNat result := by
  sorry

end base12_addition_l2714_271434


namespace complement_of_union_equals_set_l2714_271477

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- Theorem statement
theorem complement_of_union_equals_set (U A B : Set Int) :
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x : Int | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

-- Note: We use \ for set difference (complement) in Lean

end complement_of_union_equals_set_l2714_271477


namespace pentagon_angle_C_l2714_271426

/-- Represents the angles of a pentagon in degrees -/
structure PentagonAngles where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- Defines the properties of the pentagon's angles -/
def is_valid_pentagon (p : PentagonAngles) : Prop :=
  p.A > 0 ∧ p.B > 0 ∧ p.C > 0 ∧ p.D > 0 ∧ p.E > 0 ∧
  p.A < p.B ∧ p.B < p.C ∧ p.C < p.D ∧ p.D < p.E ∧
  p.A + p.B + p.C + p.D + p.E = 540 ∧
  ∃ d : ℝ, d > 0 ∧ 
    p.B - p.A = d ∧
    p.C - p.B = d ∧
    p.D - p.C = d ∧
    p.E - p.D = d

theorem pentagon_angle_C (p : PentagonAngles) 
  (h : is_valid_pentagon p) : p.C = 108 := by
  sorry

end pentagon_angle_C_l2714_271426


namespace sqrt_product_simplification_l2714_271452

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p) = 42 * p * Real.sqrt (7 * p) := by
  sorry

end sqrt_product_simplification_l2714_271452


namespace bisecting_line_slope_intercept_sum_l2714_271445

/-- Triangle ABC with vertices A(0, 8), B(2, 0), C(8, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Check if a line bisects the area of a triangle -/
def bisects_area (l : Line) (t : Triangle) : Prop := sorry

/-- The line through point B that bisects the area of the triangle -/
def bisecting_line (t : Triangle) : Line := sorry

/-- The theorem to be proved -/
theorem bisecting_line_slope_intercept_sum (t : Triangle) :
  t.A = (0, 8) ∧ t.B = (2, 0) ∧ t.C = (8, 0) →
  let l := bisecting_line t
  l.slope + l.y_intercept = -2 := by sorry

end bisecting_line_slope_intercept_sum_l2714_271445


namespace square_area_with_point_l2714_271484

/-- A square with a point inside satisfying certain distance conditions -/
structure SquareWithPoint where
  -- The side length of the square
  a : ℝ
  -- Coordinates of point P
  x : ℝ
  y : ℝ
  -- Conditions
  square_positive : 0 < a
  inside_square : 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a
  distance_to_A : x^2 + y^2 = 4
  distance_to_B : (a - x)^2 + y^2 = 9
  distance_to_C : (a - x)^2 + (a - y)^2 = 16

/-- The area of a square with a point inside satisfying certain distance conditions is 10 + √63 -/
theorem square_area_with_point (s : SquareWithPoint) : s.a^2 = 10 + Real.sqrt 63 := by
  sorry

end square_area_with_point_l2714_271484
