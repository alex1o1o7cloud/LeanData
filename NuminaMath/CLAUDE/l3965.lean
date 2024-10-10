import Mathlib

namespace hamster_lifespan_l3965_396500

theorem hamster_lifespan (fish_lifespan dog_lifespan hamster_lifespan : ℝ) 
  (h1 : fish_lifespan = dog_lifespan + 2)
  (h2 : dog_lifespan = 4 * hamster_lifespan)
  (h3 : fish_lifespan = 12) : 
  hamster_lifespan = 2.5 := by
sorry

end hamster_lifespan_l3965_396500


namespace washing_loads_proof_l3965_396572

def washing_machine_capacity : ℕ := 8
def num_shirts : ℕ := 39
def num_sweaters : ℕ := 33

theorem washing_loads_proof :
  let total_clothes := num_shirts + num_sweaters
  let num_loads := (total_clothes + washing_machine_capacity - 1) / washing_machine_capacity
  num_loads = 9 := by sorry

end washing_loads_proof_l3965_396572


namespace inequalities_hold_l3965_396516

theorem inequalities_hold (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) : 
  (m - Real.sqrt m ≥ -1/4) ∧ 
  (1/2 * (m + n)^2 + 1/4 * (m + n) ≥ m * Real.sqrt n + n * Real.sqrt m) := by
  sorry

#check inequalities_hold

end inequalities_hold_l3965_396516


namespace circle_center_sum_l3965_396519

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 48, 
    the sum of the coordinates of its center is 7 -/
theorem circle_center_sum : 
  ∀ (h k : ℝ), 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 48 ↔ (x - h)^2 + (y - k)^2 = 2) →
  h + k = 7 := by
sorry

end circle_center_sum_l3965_396519


namespace card_arrangement_probability_l3965_396569

/-- The probability of arranging cards to form a specific word --/
theorem card_arrangement_probability (n : ℕ) (n1 n2 : ℕ) (h1 : n = n1 + n2) (h2 : n1 = 2) (h3 : n2 = 3) :
  (1 : ℚ) / (n.factorial / (n1.factorial * n2.factorial)) = 1 / 10 := by
  sorry

end card_arrangement_probability_l3965_396569


namespace quadratic_unique_solution_l3965_396581

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + 2 * c = 20) →                  -- condition on a and c
  (a < c) →                           -- additional condition
  (a = 10 - 5 * Real.sqrt 2 ∧ c = 5 + (5 * Real.sqrt 2) / 2) := by
  sorry

end quadratic_unique_solution_l3965_396581


namespace butterfly_ratio_is_three_to_one_l3965_396506

/-- The ratio of time a butterfly spends as a larva to the time spent in a cocoon -/
def butterfly_development_ratio (total_time cocoon_time : ℕ) : ℚ :=
  (total_time - cocoon_time : ℚ) / cocoon_time

/-- Theorem stating that for a butterfly with 120 days total development time and 30 days in cocoon,
    the ratio of time spent as a larva to time in cocoon is 3:1 -/
theorem butterfly_ratio_is_three_to_one :
  butterfly_development_ratio 120 30 = 3 := by
  sorry

end butterfly_ratio_is_three_to_one_l3965_396506


namespace triangle_inequality_l3965_396560

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end triangle_inequality_l3965_396560


namespace circumradius_leg_ratio_not_always_equal_l3965_396599

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ

/-- The ratio of circumradii is not always equal to the ratio of leg lengths for two isosceles triangles with different leg lengths -/
theorem circumradius_leg_ratio_not_always_equal 
  (t1 t2 : IsoscelesTriangle) 
  (h : t1.leg ≠ t2.leg) : 
  ¬ ∀ (t1 t2 : IsoscelesTriangle), t1.circumradius / t2.circumradius = t1.leg / t2.leg :=
by sorry

end circumradius_leg_ratio_not_always_equal_l3965_396599


namespace student_tickets_sold_l3965_396597

theorem student_tickets_sold (student_price non_student_price total_tickets total_revenue : ℕ)
  (h1 : student_price = 9)
  (h2 : non_student_price = 11)
  (h3 : total_tickets = 2000)
  (h4 : total_revenue = 20960) :
  ∃ student_tickets : ℕ,
    student_tickets * student_price + (total_tickets - student_tickets) * non_student_price = total_revenue ∧
    student_tickets = 520 :=
by sorry

end student_tickets_sold_l3965_396597


namespace complex_number_equal_parts_l3965_396594

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) * Complex.I
  (z.re = z.im) → a = -1 := by
  sorry

end complex_number_equal_parts_l3965_396594


namespace binary_10101_is_21_l3965_396550

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_is_21_l3965_396550


namespace remainder_proof_l3965_396563

theorem remainder_proof (k : ℤ) : (k * 1127 * 1129) % 12 = 3 := by
  sorry

end remainder_proof_l3965_396563


namespace candidates_calculation_l3965_396510

theorem candidates_calculation (candidates : ℕ) : 
  (candidates * 7 / 100 = candidates * 6 / 100 + 83) → 
  candidates = 8300 := by
sorry

end candidates_calculation_l3965_396510


namespace runner_stop_time_l3965_396551

theorem runner_stop_time (total_distance : ℝ) (first_pace second_pace stop_time : ℝ) 
  (h1 : total_distance = 10)
  (h2 : first_pace = 8)
  (h3 : second_pace = 7)
  (h4 : stop_time = 8)
  (h5 : first_pace > second_pace)
  (h6 : stop_time / (first_pace - second_pace) + 
        (stop_time / (first_pace - second_pace)) * second_pace = total_distance) :
  (stop_time / (first_pace - second_pace)) * second_pace = 56 := by
  sorry


end runner_stop_time_l3965_396551


namespace crosswalk_wait_probability_l3965_396534

/-- Represents the duration of the red light in seconds -/
def red_light_duration : ℕ := 40

/-- Represents the minimum waiting time in seconds for which we calculate the probability -/
def min_wait_time : ℕ := 15

/-- The probability of waiting at least 'min_wait_time' seconds for a green light when encountering a red light -/
def wait_probability : ℚ := 5/8

/-- Theorem stating that the probability of waiting at least 'min_wait_time' seconds for a green light
    when encountering a red light of duration 'red_light_duration' is equal to 'wait_probability' -/
theorem crosswalk_wait_probability :
  (red_light_duration - min_wait_time : ℚ) / red_light_duration = wait_probability := by
  sorry

end crosswalk_wait_probability_l3965_396534


namespace sqrt_meaningful_range_l3965_396579

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end sqrt_meaningful_range_l3965_396579


namespace initial_markup_percentage_l3965_396556

/-- Given a store's pricing strategy and profit margin, 
    prove the initial markup percentage. -/
theorem initial_markup_percentage 
  (initial_cost : ℝ) 
  (markup_percentage : ℝ) 
  (new_year_markup : ℝ) 
  (february_discount : ℝ) 
  (february_profit : ℝ) 
  (h1 : new_year_markup = 0.25) 
  (h2 : february_discount = 0.09) 
  (h3 : february_profit = 0.365) : 
  1.365 = (1 + markup_percentage) * 1.25 * 0.91 := by
  sorry

#check initial_markup_percentage

end initial_markup_percentage_l3965_396556


namespace woman_birth_year_l3965_396535

theorem woman_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 < 1950) 
  (h3 : x^2 + x ≤ 2000) : x^2 = 1936 := by
  sorry

end woman_birth_year_l3965_396535


namespace unique_solution_l3965_396539

theorem unique_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (6 - y) = 9)
  (eq2 : y * (6 - z) = 9)
  (eq3 : z * (6 - x) = 9) :
  x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end unique_solution_l3965_396539


namespace complex_number_in_first_quadrant_l3965_396543

/-- The complex number z = (2-i)/(1-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l3965_396543


namespace arithmetic_sequence_common_difference_l3965_396530

/-- Given an arithmetic sequence {a_n} with a_1 = -2 and S_3 = 0, 
    where S_n is the sum of the first n terms, 
    prove that the common difference is 2. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, S n = (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →  -- Definition of S_n
  a 1 = -2 →                                                     -- a_1 = -2
  S 3 = 0 →                                                      -- S_3 = 0
  a 2 - a 1 = 2 :=                                               -- Common difference is 2
by sorry

end arithmetic_sequence_common_difference_l3965_396530


namespace quadratic_vertex_l3965_396501

/-- The quadratic function f(x) = -3(x+2)^2 + 1 has vertex coordinates (-2, 1) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ -3 * (x + 2)^2 + 1
  (∀ x, f x ≤ f (-2)) ∧ f (-2) = 1 := by
  sorry

end quadratic_vertex_l3965_396501


namespace right_triangle_hypotenuse_l3965_396561

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end right_triangle_hypotenuse_l3965_396561


namespace problem_statement_l3965_396521

theorem problem_statement (a b c : ℝ) 
  (h1 : -10 ≤ a) (h2 : a < 0) 
  (h3 : 0 < a) (h4 : a < b) (h5 : b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) := by
  sorry

end problem_statement_l3965_396521


namespace absolute_value_nonnegative_l3965_396568

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end absolute_value_nonnegative_l3965_396568


namespace art_class_gender_difference_l3965_396522

theorem art_class_gender_difference (total_students : ℕ) 
  (boy_ratio girl_ratio : ℕ) (h1 : total_students = 42) 
  (h2 : boy_ratio = 3) (h3 : girl_ratio = 4) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boy_ratio * girls = girl_ratio * boys ∧ 
    girls - boys = 6 := by
sorry

end art_class_gender_difference_l3965_396522


namespace x_value_l3965_396593

theorem x_value (x y z : ℤ) 
  (eq1 : x + y + z = 14)
  (eq2 : x - y - z = 60)
  (eq3 : x + z = 2*y) : 
  x = 37 := by
sorry

end x_value_l3965_396593


namespace pipe_cut_theorem_l3965_396582

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 68 →
  difference = 12 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 28 := by
sorry

end pipe_cut_theorem_l3965_396582


namespace imaginary_part_of_z_l3965_396541

theorem imaginary_part_of_z (z : ℂ) : z = (2 * Complex.I^2 + 4) / (Complex.I + 1) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l3965_396541


namespace factory_production_average_l3965_396540

theorem factory_production_average (first_25_avg : ℝ) (last_5_avg : ℝ) (total_days : ℕ) :
  first_25_avg = 60 →
  last_5_avg = 48 →
  total_days = 30 →
  (25 * first_25_avg + 5 * last_5_avg) / total_days = 58 := by
  sorry

end factory_production_average_l3965_396540


namespace honda_cars_in_chennai_l3965_396555

def total_cars : ℕ := 900
def red_honda_percentage : ℚ := 90 / 100
def total_red_percentage : ℚ := 60 / 100
def red_non_honda_percentage : ℚ := 225 / 1000

theorem honda_cars_in_chennai :
  ∃ (h : ℕ), h = 500 ∧
  (h : ℚ) * red_honda_percentage + (total_cars - h : ℚ) * red_non_honda_percentage = (total_cars : ℚ) * total_red_percentage :=
by sorry

end honda_cars_in_chennai_l3965_396555


namespace average_after_removing_two_numbers_l3965_396583

theorem average_after_removing_two_numbers
  (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) (final_avg : ℚ)
  (h1 : n = 50)
  (h2 : initial_avg = 38)
  (h3 : removed1 = 45)
  (h4 : removed2 = 55)
  (h5 : final_avg = 37.5) :
  initial_avg * n - (removed1 + removed2) = final_avg * (n - 2) :=
by sorry

end average_after_removing_two_numbers_l3965_396583


namespace inverse_f_at_4_l3965_396570

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- State the symmetry condition
axiom symmetry_condition (x : ℝ) : f (x + 1) + f (1 - x) = 4

-- State that f has an inverse
axiom has_inverse : Function.Bijective f

-- State that f(4) = 0
axiom f_at_4 : f 4 = 0

-- Theorem to prove
theorem inverse_f_at_4 : f_inv 4 = -2 := by sorry

end inverse_f_at_4_l3965_396570


namespace cylinder_properties_l3965_396558

/-- Properties of a cylinder with height 15 and radius 5 -/
theorem cylinder_properties :
  let h : ℝ := 15
  let r : ℝ := 5
  let total_surface_area : ℝ := 2 * Real.pi * r * r + 2 * Real.pi * r * h
  let volume : ℝ := Real.pi * r * r * h
  (total_surface_area = 200 * Real.pi) ∧ (volume = 375 * Real.pi) := by
  sorry

end cylinder_properties_l3965_396558


namespace quadratic_root_relation_l3965_396512

/-- Given two quadratic equations where the roots of one are three times the roots of the other,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (s₁ s₂ : ℝ),
    (s₁ + s₂ = -p ∧ s₁ * s₂ = m) ∧
    (3*s₁ + 3*s₂ = -m ∧ 9*s₁ * s₂ = n)) →
  n / p = 27 := by
sorry

end quadratic_root_relation_l3965_396512


namespace congruence_unique_solution_l3965_396552

theorem congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1212 [ZMOD 10] ∧ n = 8 := by
  sorry

end congruence_unique_solution_l3965_396552


namespace triangular_region_area_ratio_l3965_396526

/-- Represents a square divided into a 6x6 grid -/
structure GridSquare where
  side_length : ℝ
  grid_size : ℕ := 6

/-- Represents the triangular region in the GridSquare -/
structure TriangularRegion (gs : GridSquare) where
  vertex1 : ℝ × ℝ  -- Midpoint of one side
  vertex2 : ℝ × ℝ  -- Diagonal corner of 4x4 block
  vertex3 : ℝ × ℝ  -- Midpoint of adjacent side

/-- Calculates the area of the GridSquare -/
def area_grid_square (gs : GridSquare) : ℝ :=
  gs.side_length ^ 2

/-- Calculates the area of the TriangularRegion -/
noncomputable def area_triangular_region (gs : GridSquare) (tr : TriangularRegion gs) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The main theorem stating the ratio of areas -/
theorem triangular_region_area_ratio (gs : GridSquare) (tr : TriangularRegion gs) :
  area_triangular_region gs tr / area_grid_square gs = 1 / 24 := by
  sorry

end triangular_region_area_ratio_l3965_396526


namespace non_trivial_solutions_l3965_396598

theorem non_trivial_solutions (a b : ℝ) : 
  (∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a^2 + b^2 = 2*a*b) ∧ 
  (∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ a^2 + b^2 = 3*(a+b)) ∧ 
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ∧
  (∀ a b : ℝ, a^2 + b^2 = (a+b)^2 → a = 0 ∨ b = 0) :=
by sorry


end non_trivial_solutions_l3965_396598


namespace correct_selection_ways_l3965_396518

def total_students : ℕ := 50
def class_leaders : ℕ := 2
def students_to_select : ℕ := 5

def selection_ways : ℕ := sorry

theorem correct_selection_ways :
  selection_ways = Nat.choose class_leaders 1 * Nat.choose (total_students - class_leaders) 4 +
                   Nat.choose class_leaders 2 * Nat.choose (total_students - class_leaders) 3 ∧
  selection_ways = Nat.choose total_students students_to_select - 
                   Nat.choose (total_students - class_leaders) students_to_select ∧
  selection_ways = Nat.choose class_leaders 1 * Nat.choose (total_students - 1) 4 - 
                   Nat.choose (total_students - class_leaders) 3 ∧
  selection_ways ≠ Nat.choose class_leaders 1 * Nat.choose (total_students - 1) 4 :=
by sorry

end correct_selection_ways_l3965_396518


namespace cut_into_three_similar_rectangles_l3965_396542

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if two rectangles are similar -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- The original rectangle -/
def originalRect : Rectangle :=
  { width := 10, height := 9 }

/-- Theorem stating that the original rectangle can be cut into three unequal but similar rectangles -/
theorem cut_into_three_similar_rectangles :
  ∃ (r1 r2 r3 : Rectangle),
    r1.width + r2.width + r3.width = originalRect.width ∧
    r1.height + r2.height + r3.height = originalRect.height ∧
    r1.width ≠ r2.width ∧ r2.width ≠ r3.width ∧ r1.width ≠ r3.width ∧
    similar r1 r2 ∧ similar r2 r3 ∧ similar r1 r3 ∧
    similar r1 originalRect ∧ similar r2 originalRect ∧ similar r3 originalRect :=
  by sorry

end cut_into_three_similar_rectangles_l3965_396542


namespace angle_BEC_measure_l3965_396549

-- Define the geometric configuration
structure GeometricConfig where
  A : Real
  D : Real
  F : Real
  BEC_exists : Bool
  E_above_C : Bool

-- Define the theorem
theorem angle_BEC_measure (config : GeometricConfig) 
  (h1 : config.A = 45)
  (h2 : config.D = 50)
  (h3 : config.F = 55)
  (h4 : config.BEC_exists = true)
  (h5 : config.E_above_C = true) :
  ∃ (BEC : Real), BEC = 10 := by
  sorry

end angle_BEC_measure_l3965_396549


namespace dog_age_difference_l3965_396511

/-- The ratio of dog years to human years -/
def dogYearRatio : ℕ := 7

/-- The age of Max (the human) in years -/
def maxAge : ℕ := 3

/-- The age of Max's dog in human years -/
def dogAgeHuman : ℕ := 3

/-- Calculates the age of a dog in dog years given its age in human years -/
def dogAgeInDogYears (humanYears : ℕ) : ℕ := humanYears * dogYearRatio

/-- The difference in years between a dog's age in dog years and its owner's age in human years -/
def ageDifference (humanAge : ℕ) (dogAgeHuman : ℕ) : ℕ :=
  dogAgeInDogYears dogAgeHuman - humanAge

theorem dog_age_difference :
  ageDifference maxAge dogAgeHuman = 18 := by
  sorry

end dog_age_difference_l3965_396511


namespace min_coefficient_value_l3965_396573

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + box * x + 15) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ (min_box : ℤ), (min_box = 34 ∧ box ≥ min_box) := by
sorry

end min_coefficient_value_l3965_396573


namespace no_integer_solution_l3965_396513

theorem no_integer_solution : ∀ (x y : ℤ), x^2 + 4*x - 11 ≠ 8*y := by sorry

end no_integer_solution_l3965_396513


namespace difference_is_895_l3965_396525

/-- The smallest positive three-digit integer congruent to 7 (mod 13) -/
def m : ℕ := sorry

/-- The smallest positive four-digit integer congruent to 7 (mod 13) -/
def n : ℕ := sorry

/-- m is a three-digit number -/
axiom m_three_digit : 100 ≤ m ∧ m < 1000

/-- n is a four-digit number -/
axiom n_four_digit : 1000 ≤ n ∧ n < 10000

/-- m is congruent to 7 (mod 13) -/
axiom m_congruence : m % 13 = 7

/-- n is congruent to 7 (mod 13) -/
axiom n_congruence : n % 13 = 7

/-- m is the smallest such number -/
axiom m_smallest : ∀ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ k % 13 = 7 → m ≤ k

/-- n is the smallest such number -/
axiom n_smallest : ∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 13 = 7 → n ≤ k

theorem difference_is_895 : n - m = 895 := by sorry

end difference_is_895_l3965_396525


namespace odd_function_parallelicity_l3965_396523

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = -f x

/-- A function has parallelicity if there exist two distinct points with parallel tangent lines -/
def HasParallelicity (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    DifferentiableAt ℝ f x₁ ∧ DifferentiableAt ℝ f x₂ ∧
    deriv f x₁ = deriv f x₂

/-- Theorem: Any odd function defined on (-∞,0)∪(0,+∞) has parallelicity -/
theorem odd_function_parallelicity (f : ℝ → ℝ) (hf : IsOdd f) : HasParallelicity f := by
  sorry


end odd_function_parallelicity_l3965_396523


namespace abs_fraction_inequality_l3965_396502

theorem abs_fraction_inequality (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end abs_fraction_inequality_l3965_396502


namespace percentage_of_muslim_boys_l3965_396592

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ)
  (other_communities : ℕ) (hindu_percentage_condition : hindu_percentage = 28 / 100)
  (sikh_percentage_condition : sikh_percentage = 10 / 100)
  (total_boys_condition : total_boys = 850)
  (other_communities_condition : other_communities = 187) :
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_communities)) /
  total_boys * 100 = 40 := by
sorry

end percentage_of_muslim_boys_l3965_396592


namespace class_size_problem_l3965_396547

theorem class_size_problem (x : ℕ) : 
  (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := by sorry

end class_size_problem_l3965_396547


namespace spacy_subsets_count_l3965_396589

/-- A set of integers is spacy if it contains no more than one out of any four consecutive integers. -/
def IsSpacy (s : Set ℕ) : Prop :=
  ∀ n : ℕ, (n ∈ s → (n + 1 ∉ s ∧ n + 2 ∉ s ∧ n + 3 ∉ s))

/-- The number of spacy subsets of {1, 2, ..., n} -/
def NumSpacySubsets (n : ℕ) : ℕ :=
  if n ≤ 4 then
    n + 1
  else
    NumSpacySubsets (n - 1) + NumSpacySubsets (n - 4)

theorem spacy_subsets_count :
  NumSpacySubsets 15 = 181 :=
by sorry

end spacy_subsets_count_l3965_396589


namespace kelly_sony_games_left_l3965_396591

/-- Given that Kelly has 132 Sony games and gives away 101 Sony games, 
    prove that she will have 31 Sony games left. -/
theorem kelly_sony_games_left : 
  ∀ (initial_sony_games given_away_sony_games : ℕ),
  initial_sony_games = 132 →
  given_away_sony_games = 101 →
  initial_sony_games - given_away_sony_games = 31 :=
by sorry

end kelly_sony_games_left_l3965_396591


namespace invisible_dots_count_l3965_396586

/-- The total number of dots on a single die -/
def dots_per_die : ℕ := 21

/-- The sum of visible numbers on the stacked dice -/
def visible_sum : ℕ := 2 + 2 + 3 + 4 + 5 + 5 + 6 + 6

/-- The number of dice stacked -/
def num_dice : ℕ := 3

/-- The number of visible faces -/
def visible_faces : ℕ := 8

theorem invisible_dots_count : 
  num_dice * dots_per_die - visible_sum = 30 := by
sorry

end invisible_dots_count_l3965_396586


namespace company_production_l3965_396507

/-- Calculates the total number of bottles produced in one day given the number of cases and bottles per case. -/
def bottles_per_day (cases : ℕ) (bottles_per_case : ℕ) : ℕ :=
  cases * bottles_per_case

/-- Theorem stating that given the specific conditions, the company produces 72,000 bottles per day. -/
theorem company_production : bottles_per_day 7200 10 = 72000 := by
  sorry

end company_production_l3965_396507


namespace remaining_blue_fraction_after_four_changes_l3965_396584

/-- The fraction of a square's area that remains blue after one change -/
def blue_fraction_after_one_change : ℚ := 8 / 9

/-- The number of changes applied to the square -/
def num_changes : ℕ := 4

/-- The fraction of the original area that remains blue after the specified number of changes -/
def remaining_blue_fraction : ℚ := blue_fraction_after_one_change ^ num_changes

/-- Theorem stating that the remaining blue fraction after four changes is 4096/6561 -/
theorem remaining_blue_fraction_after_four_changes :
  remaining_blue_fraction = 4096 / 6561 := by
  sorry

end remaining_blue_fraction_after_four_changes_l3965_396584


namespace expression_simplification_l3965_396517

theorem expression_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + b^2 + x^2) - (x^2 - b^2 - a^2) / Real.sqrt (a^2 + b^2 + x^2)) / (a^2 + b^2 + x^2) =
  2 * (a^2 + b^2) / (a^2 + b^2 + x^2)^(3/2) := by
sorry

end expression_simplification_l3965_396517


namespace total_revenue_is_99_l3965_396548

def cookies_baked : ℕ := 60
def brownies_baked : ℕ := 32
def cookies_eaten_kyle : ℕ := 2
def brownies_eaten_kyle : ℕ := 2
def cookies_eaten_mom : ℕ := 1
def brownies_eaten_mom : ℕ := 2
def cookie_price : ℚ := 1
def brownie_price : ℚ := 3/2

theorem total_revenue_is_99 :
  let cookies_left := cookies_baked - (cookies_eaten_kyle + cookies_eaten_mom)
  let brownies_left := brownies_baked - (brownies_eaten_kyle + brownies_eaten_mom)
  let revenue_cookies := (cookies_left : ℚ) * cookie_price
  let revenue_brownies := (brownies_left : ℚ) * brownie_price
  revenue_cookies + revenue_brownies = 99
  := by sorry

end total_revenue_is_99_l3965_396548


namespace cos_2alpha_minus_2beta_l3965_396574

theorem cos_2alpha_minus_2beta (α β : ℝ) 
  (h1 : Real.sin (α + β) = 2/3)
  (h2 : Real.sin α * Real.cos β = 1/2) : 
  Real.cos (2*α - 2*β) = 7/9 := by
  sorry

end cos_2alpha_minus_2beta_l3965_396574


namespace stating_acid_solution_mixing_l3965_396571

/-- 
Given an initial acid solution and a replacement acid solution,
calculate the final acid concentration after replacing a portion of the initial solution.
-/
def final_acid_concentration (initial_concentration : ℝ) (replacement_concentration : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (initial_concentration * (1 - replaced_fraction) + replacement_concentration * replaced_fraction) * 100

/-- 
Theorem stating that replacing half of a 50% acid solution with a 20% acid solution 
results in a 35% acid solution.
-/
theorem acid_solution_mixing :
  final_acid_concentration 0.5 0.2 0.5 = 35 := by
sorry

#eval final_acid_concentration 0.5 0.2 0.5

end stating_acid_solution_mixing_l3965_396571


namespace solution_set_for_a_equals_one_f_lower_bound_l3965_396524

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x - 1|

-- Theorem for part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 2} = Set.Ici (4/3) ∪ Set.Iic 0 := by sorry

-- Theorem for part II
theorem f_lower_bound (a x : ℝ) :
  f a x ≥ |a - 1/2| := by sorry

end solution_set_for_a_equals_one_f_lower_bound_l3965_396524


namespace num_paths_equals_1960_l3965_396514

/-- The number of paths from A to D passing through C in a 7x9 grid -/
def num_paths_A_to_D_via_C : ℕ :=
  let grid_width := 7
  let grid_height := 9
  let C_right := 4
  let C_down := 3
  let paths_A_to_C := Nat.choose (C_right + C_down) C_right
  let paths_C_to_D := Nat.choose ((grid_width - C_right) + (grid_height - C_down)) (grid_height - C_down)
  paths_A_to_C * paths_C_to_D

/-- Theorem stating that the number of 15-step paths from A to D passing through C is 1960 -/
theorem num_paths_equals_1960 : num_paths_A_to_D_via_C = 1960 := by
  sorry

end num_paths_equals_1960_l3965_396514


namespace car_a_speed_l3965_396504

/-- Proves that Car A's speed is 58 mph given the problem conditions -/
theorem car_a_speed (initial_distance : ℝ) (car_b_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 30 →
  car_b_speed = 50 →
  time = 4.75 →
  final_distance = 8 →
  ∃ (car_a_speed : ℝ),
    car_a_speed * time = car_b_speed * time + initial_distance + final_distance ∧
    car_a_speed = 58 := by
  sorry

end car_a_speed_l3965_396504


namespace cat_stolen_pieces_l3965_396590

/-- Proves the number of pieces the cat stole given the conditions of the problem -/
theorem cat_stolen_pieces (total : ℕ) (boyfriendPieces : ℕ) : 
  total = 60 ∧ 
  boyfriendPieces = 9 ∧ 
  boyfriendPieces = (total - total / 2) / 3 →
  total - (total / 2) - ((total - total / 2) / 3) - boyfriendPieces = 3 :=
by sorry

end cat_stolen_pieces_l3965_396590


namespace right_triangle_medians_count_l3965_396564

/-- A right triangle with legs parallel to the coordinate axes -/
structure RightTriangle where
  /-- The slope of one median -/
  slope1 : ℝ
  /-- The slope of the other median -/
  slope2 : ℝ
  /-- One median lies on the line y = 5x + 1 -/
  median1_eq : slope1 = 5
  /-- The other median lies on the line y = mx + 2 -/
  median2_eq : slope2 = m
  /-- The slopes satisfy the right triangle condition -/
  slope_condition : slope1 = 4 * slope2 ∨ slope2 = 4 * slope1

/-- The theorem stating that there are exactly two values of m for which a right triangle
    with the given conditions can be constructed -/
theorem right_triangle_medians_count :
  ∃ (m1 m2 : ℝ), m1 ≠ m2 ∧
  (∀ m : ℝ, (∃ t : RightTriangle, t.slope2 = m) ↔ (m = m1 ∨ m = m2)) :=
sorry

end right_triangle_medians_count_l3965_396564


namespace intersection_of_A_and_B_l3965_396576

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 2*x + 5)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 2 := by sorry

end intersection_of_A_and_B_l3965_396576


namespace star_emilio_sum_difference_l3965_396505

def star_list := List.range 30

def emilio_list := star_list.map (fun n => 
  if n % 10 = 3 then n - 1
  else if n ≥ 30 then n - 10
  else n)

theorem star_emilio_sum_difference :
  (star_list.sum - emilio_list.sum) = 13 := by sorry

end star_emilio_sum_difference_l3965_396505


namespace c2h6_moles_used_l3965_396578

-- Define the chemical species
structure ChemicalSpecies where
  formula : String
  moles : ℝ

-- Define the balanced chemical equation
def balancedEquation (reactant1 reactant2 product1 product2 : ChemicalSpecies) : Prop :=
  reactant1.formula = "C2H6" ∧
  reactant2.formula = "Cl2" ∧
  product1.formula = "C2Cl6" ∧
  product2.formula = "HCl" ∧
  reactant1.moles = 1 ∧
  reactant2.moles = 6 ∧
  product1.moles = 1 ∧
  product2.moles = 6

-- Define the reaction conditions
def reactionConditions (cl2 c2cl6 : ChemicalSpecies) : Prop :=
  cl2.formula = "Cl2" ∧
  cl2.moles = 6 ∧
  c2cl6.formula = "C2Cl6" ∧
  c2cl6.moles = 1

-- Theorem: The number of moles of C2H6 used in the reaction is 1
theorem c2h6_moles_used
  (reactant1 reactant2 product1 product2 cl2 c2cl6 : ChemicalSpecies)
  (h1 : balancedEquation reactant1 reactant2 product1 product2)
  (h2 : reactionConditions cl2 c2cl6) :
  ∃ c2h6 : ChemicalSpecies, c2h6.formula = "C2H6" ∧ c2h6.moles = 1 :=
sorry

end c2h6_moles_used_l3965_396578


namespace sqrt_expression_equality_l3965_396520

theorem sqrt_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3)^2 = 4 - 2 * Real.sqrt 6 := by
  sorry

end sqrt_expression_equality_l3965_396520


namespace quadratic_equation_positive_root_l3965_396585

theorem quadratic_equation_positive_root (m : ℝ) :
  ∃ x : ℝ, x > 0 ∧ (x - 1) * (x - 2) - m^2 = 0 :=
by sorry

end quadratic_equation_positive_root_l3965_396585


namespace product_remainder_by_ten_l3965_396587

theorem product_remainder_by_ten (a b c : ℕ) (ha : a = 3251) (hb : b = 7462) (hc : c = 93419) :
  (a * b * c) % 10 = 8 := by
  sorry

end product_remainder_by_ten_l3965_396587


namespace geometric_mean_of_4_and_9_l3965_396580

/-- Given line segments a and b, x is their geometric mean if x^2 = ab -/
def is_geometric_mean (a b x : ℝ) : Prop := x^2 = a * b

/-- Proof that for line segments a = 4 and b = 9, their geometric mean x equals 6 -/
theorem geometric_mean_of_4_and_9 :
  ∀ x : ℝ, is_geometric_mean 4 9 x → x = 6 := by
  sorry

end geometric_mean_of_4_and_9_l3965_396580


namespace five_is_integer_l3965_396509

-- Define the set of natural numbers
def NaturalNumber : Type := ℕ

-- Define the set of integers
def Integer : Type := ℤ

-- Define the property that all natural numbers are integers
axiom all_naturals_are_integers : ∀ (n : NaturalNumber), Integer

-- Define that 5 is a natural number
axiom five_is_natural : NaturalNumber

-- Theorem to prove
theorem five_is_integer : Integer :=
sorry

end five_is_integer_l3965_396509


namespace jessica_attended_games_l3965_396546

theorem jessica_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 6)
  (h2 : missed_games = 4) :
  total_games - missed_games = 2 := by
  sorry

end jessica_attended_games_l3965_396546


namespace men_at_first_stop_l3965_396536

/-- Represents the number of people on a subway --/
structure SubwayPopulation where
  women : ℕ
  men : ℕ

/-- The subway population after the first stop --/
def first_stop : SubwayPopulation → Prop
  | ⟨w, m⟩ => m = w - 17

/-- The change in subway population at the second stop --/
def second_stop (pop : SubwayPopulation) : ℕ := 
  pop.women + pop.men + (57 + 18 - 44)

/-- The theorem stating the number of men who got on at the first stop --/
theorem men_at_first_stop (pop : SubwayPopulation) : 
  first_stop pop → second_stop pop = 502 → pop.men = 227 := by
  sorry

end men_at_first_stop_l3965_396536


namespace function_properties_unique_proportional_function_l3965_396554

/-- A proportional function passing through the point (3, 6) -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem stating the properties of the function f -/
theorem function_properties :
  (f 3 = 6) ∧
  (f 4 ≠ -2) ∧
  (f (-1.5) ≠ 3) := by
  sorry

/-- Theorem proving that f is the unique proportional function passing through (3, 6) -/
theorem unique_proportional_function (g : ℝ → ℝ) (h : ∀ x, g x = k * x) :
  g 3 = 6 → g = f := by
  sorry

end function_properties_unique_proportional_function_l3965_396554


namespace cube_root_unity_sum_l3965_396577

theorem cube_root_unity_sum (w : ℂ) 
  (h1 : w^3 - 1 = 0) 
  (h2 : w^2 + w + 1 ≠ 0) : 
  w^105 + w^106 + w^107 + w^108 + w^109 + w^110 = -2 := by
  sorry

end cube_root_unity_sum_l3965_396577


namespace correct_time_to_write_rearrangements_l3965_396538

/-- The number of unique letters in the name --/
def num_letters : ℕ := 8

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 10

/-- The number of minutes in an hour --/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours to write all rearrangements of a name --/
def time_to_write_all_rearrangements : ℚ :=
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute * minutes_per_hour)

theorem correct_time_to_write_rearrangements :
  time_to_write_all_rearrangements = 67.2 := by sorry

end correct_time_to_write_rearrangements_l3965_396538


namespace secant_theorem_l3965_396527

-- Define the basic geometric elements
variable (A B C M A₁ B₁ C₁ : ℝ × ℝ)

-- Define the triangle ABC
def is_triangle (A B C : ℝ × ℝ) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define that M is not on the sides or extensions of ABC
def M_not_on_triangle (A B C M : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, M ≠ A + t • (B - A) ∧ 
           M ≠ B + t • (C - B) ∧ 
           M ≠ C + t • (A - C)

-- Define the secant through M intersecting sides (or extensions) at A₁, B₁, C₁
def secant_intersects (A B C M A₁ B₁ C₁ : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 
    A₁ = A + t₁ • (B - A) ∧
    B₁ = B + t₂ • (C - B) ∧
    C₁ = C + t₃ • (A - C) ∧
    (∃ s₁ s₂ s₃ : ℝ, M = A₁ + s₁ • (B₁ - A₁) ∧
                     M = B₁ + s₂ • (C₁ - B₁) ∧
                     M = C₁ + s₃ • (A₁ - C₁))

-- Define oriented area function
noncomputable def oriented_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define oriented distance function
noncomputable def oriented_distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem secant_theorem (A B C M A₁ B₁ C₁ : ℝ × ℝ) 
  (h_triangle : is_triangle A B C)
  (h_M_not_on : M_not_on_triangle A B C M)
  (h_secant : secant_intersects A B C M A₁ B₁ C₁) :
  (oriented_area A B M) / (oriented_distance M C₁) + 
  (oriented_area B C M) / (oriented_distance M A₁) + 
  (oriented_area C A M) / (oriented_distance M B₁) = 0 := by sorry

end secant_theorem_l3965_396527


namespace largest_divisor_of_n_l3965_396503

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : ∃ k : ℕ, k ∣ n ∧ k = 12 ∧ ∀ m : ℕ, m ∣ n → m ≤ k := by
  sorry

end largest_divisor_of_n_l3965_396503


namespace largest_double_after_digit_removal_l3965_396596

def is_double_after_digit_removal (x : ℚ) : Prop :=
  ∃ (y : ℚ), (y > 0) ∧ (y < 1) ∧ (x = 0.1 * 3 + y) ∧ (2 * x = 0.1 * 0 + y)

theorem largest_double_after_digit_removal :
  ∀ (x : ℚ), (x > 0) → (x < 1) → is_double_after_digit_removal x → x ≤ 0.375 :=
sorry

end largest_double_after_digit_removal_l3965_396596


namespace exponent_rule_equality_l3965_396567

theorem exponent_rule_equality (x : ℝ) (m : ℤ) (h : x ≠ 0) :
  (x^3)^m / (x^m)^2 = x^m :=
sorry

end exponent_rule_equality_l3965_396567


namespace sin_330_degrees_l3965_396529

theorem sin_330_degrees : Real.sin (330 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end sin_330_degrees_l3965_396529


namespace games_in_own_division_l3965_396566

/-- Represents a baseball league with specific game scheduling rules -/
structure BaseballLeague where
  N : ℕ  -- Number of games against each team in own division
  M : ℕ  -- Number of games against each team in other division
  h1 : N > 2 * M
  h2 : M > 4
  h3 : 4 * N + 5 * M = 82

/-- The number of games a team plays within its own division is 52 -/
theorem games_in_own_division (league : BaseballLeague) : 4 * league.N = 52 := by
  sorry

end games_in_own_division_l3965_396566


namespace max_visible_sum_l3965_396533

-- Define a cube type
structure Cube :=
  (faces : Fin 6 → Nat)

-- Define the block type
structure Block :=
  (cubes : Fin 4 → Cube)

-- Function to calculate the sum of visible faces
def sumVisibleFaces (b : Block) : Nat :=
  sorry

-- Theorem statement
theorem max_visible_sum :
  ∃ (b : Block),
    (∀ i : Fin 6, ∀ c : Fin 4, 1 ≤ (b.cubes c).faces i ∧ (b.cubes c).faces i ≤ 6) ∧
    (∀ c1 c2 : Fin 4, c1 ≠ c2 → ∃ i j : Fin 6, (b.cubes c1).faces i = (b.cubes c2).faces j) ∧
    (sumVisibleFaces b = 68) ∧
    (∀ b' : Block, sumVisibleFaces b' ≤ sumVisibleFaces b) :=
  sorry


end max_visible_sum_l3965_396533


namespace increasing_order_x_y_z_l3965_396544

theorem increasing_order_x_y_z (x : ℝ) (hx : 1.1 < x ∧ x < 1.2) :
  x < x^x ∧ x^x < x^(x^x) := by
  sorry

end increasing_order_x_y_z_l3965_396544


namespace monomial_sum_equality_l3965_396595

/-- Given that the sum of two monomials is a monomial, prove the exponents are equal -/
theorem monomial_sum_equality (x y : ℝ) (m n : ℕ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), -x^m * y^(2+3*n) + 5 * x^(2*n-3) * y^8 = a * x^m * y^(2+3*n)) → 
  (m = 1 ∧ n = 2) :=
sorry

end monomial_sum_equality_l3965_396595


namespace arithmetic_progression_property_l3965_396575

/-- An arithmetic progression with n terms and common difference d -/
structure ArithmeticProgression where
  n : ℕ
  d : ℚ
  first_term : ℚ

/-- The sum of absolute values of terms in an arithmetic progression -/
def sum_of_abs_values (ap : ArithmeticProgression) : ℚ :=
  sorry

/-- Theorem stating the relation between n, d, and the sum of absolute values -/
theorem arithmetic_progression_property (ap : ArithmeticProgression) :
  (sum_of_abs_values ap = 100) ∧
  (sum_of_abs_values { ap with first_term := ap.first_term + 1 } = 100) ∧
  (sum_of_abs_values { ap with first_term := ap.first_term + 2 } = 100) →
  ap.n^2 * ap.d = 400 :=
sorry

end arithmetic_progression_property_l3965_396575


namespace integer_solution_system_l3965_396537

theorem integer_solution_system (a b c : ℤ) : 
  a^2 = b*c + 1 ∧ b^2 = a*c + 1 ↔ 
  (a = 1 ∧ b = 0 ∧ c = -1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = -1 ∧ b = 1 ∧ c = 0) := by
sorry

end integer_solution_system_l3965_396537


namespace trajectory_is_line_segment_l3965_396559

/-- The set of points P satisfying |PF₁| + |PF₂| = 10, where F₁ and F₂ are fixed points, forms a line segment. -/
theorem trajectory_is_line_segment (F₁ F₂ : ℝ × ℝ) (h₁ : F₁ = (-5, 0)) (h₂ : F₂ = (5, 0)) :
  {P : ℝ × ℝ | dist P F₁ + dist P F₂ = 10} = {P : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂} :=
sorry

end trajectory_is_line_segment_l3965_396559


namespace parabola_point_ordering_l3965_396562

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, f 0)
def B : ℝ × ℝ := (1, f 1)
def C : ℝ × ℝ := (-2, f (-2))

-- Define y₁, y₂, and y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_point_ordering : y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end parabola_point_ordering_l3965_396562


namespace clinton_shoes_count_l3965_396557

/-- Clinton's wardrobe inventory problem -/
theorem clinton_shoes_count :
  ∀ (shoes belts hats : ℕ),
  shoes = 2 * belts →
  belts = hats + 2 →
  hats = 5 →
  shoes = 14 :=
by sorry

end clinton_shoes_count_l3965_396557


namespace randy_trip_length_l3965_396565

theorem randy_trip_length :
  ∀ (total_length : ℚ),
    (1 / 4 : ℚ) * total_length +  -- gravel road
    30 +                          -- pavement
    (1 / 8 : ℚ) * total_length +  -- scenic route
    (1 / 6 : ℚ) * total_length    -- dirt road
    = total_length →
    total_length = 720 / 11 := by
  sorry

end randy_trip_length_l3965_396565


namespace f_continuous_at_2_l3965_396588

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 * x^2 - 7 else b * (x - 2)^2 + 5

-- State the theorem
theorem f_continuous_at_2 (b : ℝ) : 
  ContinuousAt (f b) 2 := by sorry

end f_continuous_at_2_l3965_396588


namespace basketball_win_percentage_l3965_396531

theorem basketball_win_percentage (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (target_percentage : ℚ) : 
  total_games = 110 →
  first_games = 60 →
  first_wins = 45 →
  target_percentage = 3/4 →
  ∃ (remaining_wins : ℕ), 
    remaining_wins = 38 ∧ 
    (first_wins + remaining_wins : ℚ) / total_games = target_percentage :=
by sorry

end basketball_win_percentage_l3965_396531


namespace half_minus_third_equals_sixth_l3965_396545

theorem half_minus_third_equals_sixth : (1/2 : ℚ) - (1/3 : ℚ) = 1/6 := by sorry

end half_minus_third_equals_sixth_l3965_396545


namespace soda_price_increase_l3965_396528

theorem soda_price_increase (initial_total : ℝ) (new_candy_price new_soda_price : ℝ) 
  (candy_increase : ℝ) :
  initial_total = 16 →
  new_candy_price = 10 →
  new_soda_price = 12 →
  candy_increase = 0.25 →
  (new_soda_price / (initial_total - new_candy_price / (1 + candy_increase)) - 1) * 100 = 50 := by
  sorry

end soda_price_increase_l3965_396528


namespace pool_fill_time_l3965_396515

/-- Proves that a pool with the given specifications takes 25 hours to fill -/
theorem pool_fill_time (pool_volume : ℝ) (hose1_rate : ℝ) (hose2_rate : ℝ) 
  (h_volume : pool_volume = 15000)
  (h_hose1 : hose1_rate = 2)
  (h_hose2 : hose2_rate = 3) : 
  pool_volume / (2 * hose1_rate + 2 * hose2_rate) / 60 = 25 := by
  sorry

end pool_fill_time_l3965_396515


namespace boys_count_l3965_396532

/-- Represents the number of skips in a jump rope competition. -/
structure SkipCompetition where
  boyAvg : ℕ
  girlAvg : ℕ
  totalAvg : ℕ
  boyCount : ℕ
  girlCount : ℕ

/-- Theorem stating the number of boys in the skip competition. -/
theorem boys_count (comp : SkipCompetition) 
  (h1 : comp.boyAvg = 85)
  (h2 : comp.girlAvg = 92)
  (h3 : comp.totalAvg = 88)
  (h4 : comp.boyCount = comp.girlCount + 10)
  (h5 : (comp.boyAvg * comp.boyCount + comp.girlAvg * comp.girlCount) / (comp.boyCount + comp.girlCount) = comp.totalAvg) :
  comp.boyCount = 40 := by
  sorry

end boys_count_l3965_396532


namespace hiking_distance_proof_l3965_396553

theorem hiking_distance_proof (total_distance : ℝ) : 
  (total_distance / 3 + (2 * total_distance / 3) / 3 + (4 * total_distance / 9) / 4 + 24 = total_distance) →
  (total_distance = 72 ∧ 
   total_distance / 3 = 24 ∧ 
   (2 * total_distance / 3) / 3 = 16 ∧ 
   (4 * total_distance / 9) / 4 = 8) :=
by
  sorry

#check hiking_distance_proof

end hiking_distance_proof_l3965_396553


namespace perpendicular_lines_a_value_l3965_396508

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 2 → 2 * x + 4 * y = 5 → (1 : ℝ) * (2 : ℝ) + a * (4 : ℝ) = 0) → 
  a = -1/2 := by
  sorry

end perpendicular_lines_a_value_l3965_396508
