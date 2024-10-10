import Mathlib

namespace a_worked_six_days_l1497_149788

/-- Represents the number of days worked by person a -/
def days_a : ℕ := sorry

/-- Represents the daily wage of person a -/
def wage_a : ℕ := sorry

/-- Represents the daily wage of person b -/
def wage_b : ℕ := sorry

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 100

/-- The total earnings of all three workers -/
def total_earnings : ℕ := 1480

/-- Theorem stating that person a worked for 6 days -/
theorem a_worked_six_days :
  (wage_a = 3 * wage_c / 5) ∧
  (wage_b = 4 * wage_c / 5) ∧
  (days_a * wage_a + 9 * wage_b + 4 * wage_c = total_earnings) →
  days_a = 6 :=
by sorry

end a_worked_six_days_l1497_149788


namespace sum_of_odd_and_multiples_of_five_l1497_149770

/-- The number of five-digit odd numbers -/
def A : ℕ := 45000

/-- The number of five-digit multiples of 5 -/
def B : ℕ := 18000

/-- The sum of five-digit odd numbers and five-digit multiples of 5 is 63000 -/
theorem sum_of_odd_and_multiples_of_five : A + B = 63000 := by
  sorry

end sum_of_odd_and_multiples_of_five_l1497_149770


namespace f_of_x_plus_one_l1497_149723

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x - 3

-- State the theorem
theorem f_of_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 := by
  sorry

end f_of_x_plus_one_l1497_149723


namespace total_pears_picked_l1497_149754

theorem total_pears_picked (jason_pears keith_pears mike_pears sarah_pears : ℝ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12)
  (h4 : sarah_pears = 32.5)
  (emma_pears : ℝ)
  (h5 : emma_pears = 2 / 3 * mike_pears)
  (james_pears : ℝ)
  (h6 : james_pears = 2 * sarah_pears - 3) :
  jason_pears + keith_pears + mike_pears + sarah_pears + emma_pears + james_pears = 207.5 := by
  sorry

#check total_pears_picked

end total_pears_picked_l1497_149754


namespace gold_cube_value_scaling_l1497_149744

/-- Represents the properties of a gold cube -/
structure GoldCube where
  side_length : ℝ
  value : ℝ

/-- Theorem stating the relationship between two gold cubes of different sizes -/
theorem gold_cube_value_scaling (small_cube large_cube : GoldCube) :
  small_cube.side_length = 4 →
  small_cube.value = 800 →
  large_cube.side_length = 6 →
  large_cube.value = 2700 := by
  sorry

#check gold_cube_value_scaling

end gold_cube_value_scaling_l1497_149744


namespace sector_area_special_case_l1497_149753

/-- The area of a sector with central angle 2π/3 and radius √3 is equal to π. -/
theorem sector_area_special_case :
  let central_angle : ℝ := 2 * Real.pi / 3
  let radius : ℝ := Real.sqrt 3
  let sector_area : ℝ := (1 / 2) * radius^2 * central_angle
  sector_area = Real.pi := by
  sorry

end sector_area_special_case_l1497_149753


namespace triangle_sine_product_inequality_l1497_149720

theorem triangle_sine_product_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end triangle_sine_product_inequality_l1497_149720


namespace arithmetic_sequence_common_difference_l1497_149773

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with S_2 = 4 and S_4 = 20, the common difference is 3 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h2 : seq.S 2 = 4)
  (h4 : seq.S 4 = 20) :
  seq.d = 3 := by
  sorry

end arithmetic_sequence_common_difference_l1497_149773


namespace marshas_delivery_problem_l1497_149751

/-- Marsha's delivery problem -/
theorem marshas_delivery_problem (x : ℝ) : 
  (x + 28 + 14) * 2 = 104 → x = 10 := by sorry

end marshas_delivery_problem_l1497_149751


namespace arithmetic_computation_l1497_149757

theorem arithmetic_computation : 12 + 4 * (5 - 2 * 3)^2 = 16 := by sorry

end arithmetic_computation_l1497_149757


namespace cube_sum_divided_by_quadratic_minus_product_plus_square_l1497_149797

theorem cube_sum_divided_by_quadratic_minus_product_plus_square (a b : ℝ) :
  a = 6 ∧ b = 3 → (a^3 + b^3) / (a^2 - a*b + b^2) = 9 := by
  sorry

end cube_sum_divided_by_quadratic_minus_product_plus_square_l1497_149797


namespace prob_two_red_balls_l1497_149776

/-- The probability of selecting 2 red balls from a bag with 5 red, 6 blue, and 4 green balls -/
theorem prob_two_red_balls (red blue green : ℕ) (total : ℕ) (h1 : red = 5) (h2 : blue = 6) (h3 : green = 4) (h4 : total = red + blue + green) :
  (red.choose 2 : ℚ) / (total.choose 2) = 2 / 21 := by
  sorry

end prob_two_red_balls_l1497_149776


namespace silver_division_problem_l1497_149795

/-- 
Given:
- m : ℕ is the number of people
- n : ℕ is the total amount of silver in taels
- Adding 7 taels to each person's share and 7 taels in total equals n
- Subtracting 8 taels from each person's share and subtracting 8 taels in total equals n

Prove that the system of equations 7m + 7 = n and 8m - 8 = n correctly represents the situation
-/
theorem silver_division_problem (m n : ℕ) 
  (h1 : 7 * m + 7 = n) 
  (h2 : 8 * m - 8 = n) : 
  (7 * m + 7 = n) ∧ (8 * m - 8 = n) := by
  sorry

end silver_division_problem_l1497_149795


namespace field_dimensions_l1497_149790

/-- Proves that for a rectangular field with given dimensions, if the area is 92, then m = 4 -/
theorem field_dimensions (m : ℝ) : 
  (3*m + 6) * (m - 3) = 92 → m = 4 := by
sorry

end field_dimensions_l1497_149790


namespace divisibility_problem_l1497_149762

theorem divisibility_problem (x y z : ℕ) (h1 : x = 987654) (h2 : y = 456) (h3 : z = 222) :
  (x + z) % y = 0 := by
  sorry

end divisibility_problem_l1497_149762


namespace rocky_knockout_percentage_l1497_149746

/-- Proves that the percentage of Rocky's knockouts that were in the first round is 20% -/
theorem rocky_knockout_percentage : 
  ∀ (total_fights : ℕ) 
    (knockout_percentage : ℚ) 
    (first_round_knockouts : ℕ),
  total_fights = 190 →
  knockout_percentage = 1/2 →
  first_round_knockouts = 19 →
  (first_round_knockouts : ℚ) / (knockout_percentage * total_fights) = 1/5 := by
sorry

end rocky_knockout_percentage_l1497_149746


namespace negative_two_times_inequality_l1497_149755

theorem negative_two_times_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end negative_two_times_inequality_l1497_149755


namespace bacteria_after_three_hours_l1497_149768

/-- Represents the number of bacteria after a given number of half-hour periods. -/
def bacteria_population (half_hours : ℕ) : ℕ := 2^half_hours

/-- Theorem stating that after 3 hours (6 half-hour periods), the bacteria population will be 64. -/
theorem bacteria_after_three_hours : bacteria_population 6 = 64 := by
  sorry

end bacteria_after_three_hours_l1497_149768


namespace skittles_transfer_l1497_149706

theorem skittles_transfer (bridget_initial : ℕ) (henry_initial : ℕ) : 
  bridget_initial = 4 → henry_initial = 4 → bridget_initial + henry_initial = 8 := by
sorry

end skittles_transfer_l1497_149706


namespace smallest_number_with_digit_product_10_factorial_l1497_149734

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def is_valid_number (n : ℕ) : Prop :=
  digit_product n = factorial 10

theorem smallest_number_with_digit_product_10_factorial :
  ∀ n : ℕ, n < 45578899 → ¬(is_valid_number n) ∧ is_valid_number 45578899 :=
sorry

end smallest_number_with_digit_product_10_factorial_l1497_149734


namespace units_digit_17_cubed_times_24_l1497_149784

theorem units_digit_17_cubed_times_24 : (17^3 * 24) % 10 = 2 := by
  sorry

end units_digit_17_cubed_times_24_l1497_149784


namespace functional_equation_solutions_l1497_149727

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x*y - z*t) + f (x*t + y*z)

/-- The main theorem stating the possible solutions. -/
theorem functional_equation_solutions (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = 1/2) ∨ (∀ x, f x = x^2) := by
  sorry

end functional_equation_solutions_l1497_149727


namespace quadratic_function_property_l1497_149717

theorem quadratic_function_property (a m : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  (f m < 0) → (f (m - 1) > 0) := by
sorry

end quadratic_function_property_l1497_149717


namespace erased_digit_is_four_l1497_149700

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the property that a number is divisible by 9
def divisibleBy9 (n : ℕ) : Prop := n % 9 = 0

-- Main theorem
theorem erased_digit_is_four (N : ℕ) (D : ℕ) (x : ℕ) :
  D = N - sumOfDigits N →
  divisibleBy9 D →
  sumOfDigits D = 131 + x →
  x = 4 := by sorry

end erased_digit_is_four_l1497_149700


namespace vector_simplification_l1497_149779

-- Define the Euclidean space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- Define points in the Euclidean space
variable (A B C D : E)

-- Define vectors as differences between points
def vector (P Q : E) : E := Q - P

-- State the theorem
theorem vector_simplification (A B C D : E) :
  vector A B + vector B C - vector A D = vector D C := by sorry

end vector_simplification_l1497_149779


namespace rational_criterion_l1497_149725

/-- The number of different digit sequences of length n in the decimal expansion of a real number -/
def num_digit_sequences (a : ℝ) (n : ℕ) : ℕ := sorry

/-- A real number is rational if there exists a natural number n such that 
    the number of different digit sequences of length n in its decimal expansion 
    is less than or equal to n + 8 -/
theorem rational_criterion (a : ℝ) : 
  (∃ n : ℕ, num_digit_sequences a n ≤ n + 8) → ∃ q : ℚ, a = ↑q := by sorry

end rational_criterion_l1497_149725


namespace right_triangle_arc_segment_l1497_149722

theorem right_triangle_arc_segment (AC CB : ℝ) (h_AC : AC = 15) (h_CB : CB = 8) :
  let AB := Real.sqrt (AC^2 + CB^2)
  let CP := (AC * CB) / AB
  let PB := Real.sqrt (CB^2 - CP^2)
  let BD := 2 * PB
  BD = 128 / 17 := by sorry

end right_triangle_arc_segment_l1497_149722


namespace solve_flower_problem_l1497_149791

def flower_problem (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (bouquets_after_wilting : ℕ) : Prop :=
  let remaining_flowers := bouquets_after_wilting * flowers_per_bouquet
  let wilted_flowers := initial_flowers - remaining_flowers
  wilted_flowers = 35

theorem solve_flower_problem :
  flower_problem 45 5 2 :=
by
  sorry

end solve_flower_problem_l1497_149791


namespace money_difference_l1497_149765

theorem money_difference (eric ben jack : ℕ) 
  (h1 : eric = ben - 10)
  (h2 : ben < 26)
  (h3 : eric + ben + 26 = 50)
  : jack - ben = 9 := by
  sorry

end money_difference_l1497_149765


namespace concrete_mixture_problem_l1497_149743

theorem concrete_mixture_problem (total_weight : Real) (final_cement_percentage : Real) 
  (amount_each_type : Real) (h1 : total_weight = 4500) 
  (h2 : final_cement_percentage = 10.8) (h3 : amount_each_type = 1125) :
  ∃ (first_type_percentage : Real),
    first_type_percentage = 2 ∧
    (amount_each_type * first_type_percentage / 100 + 
     amount_each_type * (2 * final_cement_percentage - first_type_percentage) / 100) = 
    (total_weight * final_cement_percentage / 100) :=
by sorry

end concrete_mixture_problem_l1497_149743


namespace simplify_sqrt_expression_l1497_149783

theorem simplify_sqrt_expression : 
  Real.sqrt 80 - 3 * Real.sqrt 10 + (2 * Real.sqrt 500) / Real.sqrt 5 = Real.sqrt 2205 := by
  sorry

end simplify_sqrt_expression_l1497_149783


namespace stick_cutting_l1497_149766

theorem stick_cutting (n : ℕ) (a₁ a₂ a₃ : ℕ) 
  (h_pos : n > 0)
  (h_min : a₁ ≥ n ∧ a₂ ≥ n ∧ a₃ ≥ n)
  (h_sum : a₁ + a₂ + a₃ = n * (n + 1) / 2) :
  ∃ (segments : List ℕ), 
    segments.length = n ∧ 
    segments.sum = a₁ + a₂ + a₃ ∧
    (∀ i ∈ Finset.range n, (i + 1) ∈ segments) :=
by sorry

end stick_cutting_l1497_149766


namespace seating_arrangements_l1497_149726

/-- Represents a row of seats -/
structure Row :=
  (total : ℕ)
  (available : ℕ)

/-- Calculates the number of seating arrangements for two people in a single row -/
def arrangementsInRow (row : Row) : ℕ :=
  row.available * (row.available - 1)

/-- Calculates the number of seating arrangements for two people in different rows -/
def arrangementsAcrossRows (row1 row2 : Row) : ℕ :=
  row1.available * row2.available * 2

/-- The main theorem stating the total number of seating arrangements -/
theorem seating_arrangements :
  let frontRow : Row := ⟨11, 8⟩
  let backRow : Row := ⟨12, 12⟩
  arrangementsInRow frontRow + arrangementsInRow backRow + arrangementsAcrossRows frontRow backRow = 334 :=
by sorry

end seating_arrangements_l1497_149726


namespace functional_equation_solution_l1497_149781

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f x + f y = 2 * f ((x + y) / 2)) →
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
by sorry

end functional_equation_solution_l1497_149781


namespace atlantis_population_growth_l1497_149738

def initial_year : Nat := 2000
def initial_population : Nat := 400
def island_capacity : Nat := 15000
def years_to_check : Nat := 200

def population_after_n_cycles (n : Nat) : Nat :=
  initial_population * 2^n

theorem atlantis_population_growth :
  ∃ (y : Nat), y ≤ years_to_check ∧ 
  population_after_n_cycles (y / 40) ≥ island_capacity :=
sorry

end atlantis_population_growth_l1497_149738


namespace sequence_problem_l1497_149767

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_problem (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_a_sum : a 1 + a 5 + a 9 = 9)
    (h_b_prod : b 2 * b 5 * b 8 = 3 * Real.sqrt 3) :
    (a 2 + a 8) / (1 + b 2 * b 8) = 3/2 := by
  sorry

end sequence_problem_l1497_149767


namespace g_of_6_l1497_149731

def g (x : ℝ) : ℝ := 2*x^4 - 13*x^3 + 28*x^2 - 32*x - 48

theorem g_of_6 : g 6 = 552 := by
  sorry

end g_of_6_l1497_149731


namespace cone_sphere_radius_theorem_l1497_149771

/-- Represents a right cone with a sphere inscribed within it. -/
structure ConeWithSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ

/-- Checks if the sphere radius can be expressed in the form b√d - b. -/
def has_valid_sphere_radius (cone : ConeWithSphere) (b d : ℕ) : Prop :=
  cone.sphere_radius = b * Real.sqrt d - b

/-- The main theorem stating the relationship between b and d for the given cone. -/
theorem cone_sphere_radius_theorem (cone : ConeWithSphere) (b d : ℕ) :
  cone.base_radius = 15 →
  cone.height = 20 →
  has_valid_sphere_radius cone b d →
  b + d = 17 := by
  sorry

#check cone_sphere_radius_theorem

end cone_sphere_radius_theorem_l1497_149771


namespace angle_inequality_l1497_149782

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.cos θ - x * (1 - x) * Real.tan θ + (1 - x)^2 * Real.sin θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end angle_inequality_l1497_149782


namespace final_price_after_discounts_l1497_149792

def original_price : Float := 49.99
def first_discount_rate : Float := 0.10
def second_discount_rate : Float := 0.20

theorem final_price_after_discounts :
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 36.00 := by
  sorry

end final_price_after_discounts_l1497_149792


namespace sin_alpha_value_l1497_149740

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/6) = 1/3) :
  Real.sin α = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

end sin_alpha_value_l1497_149740


namespace inequality_range_l1497_149798

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) → 
  -6 ≤ m ∧ m ≤ 2 := by
sorry

end inequality_range_l1497_149798


namespace shopkeeper_card_decks_l1497_149739

theorem shopkeeper_card_decks 
  (total_cards : ℕ) 
  (additional_cards : ℕ) 
  (cards_per_deck : ℕ) 
  (h1 : total_cards = 160)
  (h2 : additional_cards = 4)
  (h3 : cards_per_deck = 52) :
  (total_cards - additional_cards) / cards_per_deck = 3 := by
  sorry

end shopkeeper_card_decks_l1497_149739


namespace determinant_of_specific_matrix_l1497_149710

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]
  Matrix.det A = 48 := by sorry

end determinant_of_specific_matrix_l1497_149710


namespace jack_emails_l1497_149709

theorem jack_emails (morning_emails : ℕ) (afternoon_emails : ℕ) 
  (h1 : morning_emails = 6)
  (h2 : afternoon_emails = morning_emails + 2) : 
  afternoon_emails = 8 := by
sorry

end jack_emails_l1497_149709


namespace semicircle_perimeter_l1497_149703

/-- The perimeter of a semi-circle with radius r is πr + 2r -/
theorem semicircle_perimeter (r : ℝ) (hr : r > 0) : 
  let P := π * r + 2 * r
  P = π * r + 2 * r := by
  sorry

end semicircle_perimeter_l1497_149703


namespace rain_in_first_hour_l1497_149785

theorem rain_in_first_hour (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by
sorry

end rain_in_first_hour_l1497_149785


namespace frog_jump_distance_l1497_149761

/-- The jumping contest problem -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (frog_extra_jump : ℕ) :
  grasshopper_jump = 36 →
  frog_extra_jump = 17 →
  grasshopper_jump + frog_extra_jump = 53 :=
by
  sorry

#check frog_jump_distance

end frog_jump_distance_l1497_149761


namespace fraction_equality_l1497_149752

theorem fraction_equality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) :
  (2 * a + b) / b = 4 := by
  sorry

end fraction_equality_l1497_149752


namespace right_triangle_cos_c_l1497_149713

theorem right_triangle_cos_c (A B C : Real) (sinB : Real) :
  -- Triangle ABC exists
  -- Angle A is a right angle (90 degrees)
  A + B + C = Real.pi →
  A = Real.pi / 2 →
  -- sin B is given as 3/5
  sinB = 3 / 5 →
  -- Prove that cos C = 3/5
  Real.cos C = 3 / 5 := by
  sorry

end right_triangle_cos_c_l1497_149713


namespace expression_equality_l1497_149789

theorem expression_equality : 
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 := by
  sorry

end expression_equality_l1497_149789


namespace complement_of_M_in_U_l1497_149714

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 6*x + 5 ≤ 0}

-- State the theorem
theorem complement_of_M_in_U :
  (U \ M) = {6, 7} := by
  sorry

end complement_of_M_in_U_l1497_149714


namespace solid_with_isosceles_triangle_views_is_tetrahedron_l1497_149732

/-- A solid object in 3D space -/
structure Solid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a view (projection) of a solid -/
inductive View
  | Front
  | Top
  | Side

/-- Represents the shape of a view -/
inductive Shape
  | IsoscelesTriangle
  | Other

/-- Function to get the shape of a view for a given solid -/
def viewShape (s : Solid) (v : View) : Shape :=
  sorry -- Implementation details

/-- Predicate to check if a solid is a tetrahedron -/
def isTetrahedron (s : Solid) : Prop :=
  sorry -- Definition of a tetrahedron

/-- Theorem: If all three views of a solid are isosceles triangles, then it's a tetrahedron -/
theorem solid_with_isosceles_triangle_views_is_tetrahedron (s : Solid) :
  (∀ v : View, viewShape s v = Shape.IsoscelesTriangle) →
  isTetrahedron s :=
sorry

end solid_with_isosceles_triangle_views_is_tetrahedron_l1497_149732


namespace remainder_nineteen_power_plus_nineteen_mod_twenty_l1497_149760

theorem remainder_nineteen_power_plus_nineteen_mod_twenty : (19^19 + 19) % 20 = 18 := by
  sorry

end remainder_nineteen_power_plus_nineteen_mod_twenty_l1497_149760


namespace quadratic_inequality_range_l1497_149745

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m + 1) * x^2 - (m + 1) * x + 1 ≤ 0) ↔ 
  (m ≥ -1 ∧ m < 3) :=
sorry

end quadratic_inequality_range_l1497_149745


namespace f_max_min_range_l1497_149747

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Condition for f to have both a maximum and a minimum -/
def has_max_and_min (a : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0

theorem f_max_min_range (a : ℝ) : has_max_and_min a → a < -3 ∨ a > 6 := by sorry

end f_max_min_range_l1497_149747


namespace fourth_coaster_speed_l1497_149721

/-- Given 5 rollercoasters with known speeds for 4 of them and a known average speed for all 5,
    prove that the speed of the unknown coaster is equal to the total speed (based on the average)
    minus the sum of the known speeds. -/
theorem fourth_coaster_speed
  (speed1 speed2 speed3 speed5 : ℝ)
  (average_speed : ℝ)
  (h1 : speed1 = 50)
  (h2 : speed2 = 62)
  (h3 : speed3 = 73)
  (h5 : speed5 = 40)
  (h_avg : average_speed = 59)
  : ∃ speed4 : ℝ,
    speed4 = 5 * average_speed - (speed1 + speed2 + speed3 + speed5) :=
by sorry

end fourth_coaster_speed_l1497_149721


namespace dice_roll_probability_l1497_149711

def total_outcomes : ℕ := 6^6

def ways_to_choose_numbers : ℕ := Nat.choose 6 2

def ways_to_arrange_dice : ℕ := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1)

def successful_outcomes : ℕ := ways_to_choose_numbers * ways_to_arrange_dice

theorem dice_roll_probability :
  (successful_outcomes : ℚ) / total_outcomes = 25 / 1296 := by
  sorry

end dice_roll_probability_l1497_149711


namespace azalea_sheep_count_l1497_149712

/-- The number of sheep Azalea sheared -/
def num_sheep : ℕ := 200

/-- The amount paid to the shearer -/
def shearer_payment : ℕ := 2000

/-- The amount of wool produced by each sheep in pounds -/
def wool_per_sheep : ℕ := 10

/-- The price of wool per pound -/
def wool_price : ℕ := 20

/-- The profit made by Azalea -/
def profit : ℕ := 38000

/-- Theorem stating that the number of sheep Azalea sheared is 200 -/
theorem azalea_sheep_count :
  num_sheep = (profit + shearer_payment) / (wool_per_sheep * wool_price) :=
by sorry

end azalea_sheep_count_l1497_149712


namespace min_value_theorem_l1497_149777

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2*a + b = 6) :
  (1/a + 2/b) ≥ 4/3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + b₀ = 6 ∧ 1/a₀ + 2/b₀ = 4/3 := by
sorry

end min_value_theorem_l1497_149777


namespace problem_statement_l1497_149704

theorem problem_statement : (3.14 - Real.pi) ^ 0 - 2 ^ (-1 : ℤ) = (1 : ℝ) / 2 := by
  sorry

end problem_statement_l1497_149704


namespace particular_solutions_l1497_149702

/-- The differential equation -/
def diff_eq (x y y' : ℝ) : Prop :=
  x * y'^2 - 2 * y * y' + 4 * x = 0

/-- The general integral -/
def general_integral (x y C : ℝ) : Prop :=
  x^2 = C * (y - C)

/-- Theorem stating that y = 2x and y = -2x are particular solutions -/
theorem particular_solutions (x : ℝ) (hx : x > 0) :
  (diff_eq x (2*x) 2 ∧ diff_eq x (-2*x) (-2)) ∧
  (∃ C, general_integral x (2*x) C) ∧
  (∃ C, general_integral x (-2*x) C) :=
sorry

end particular_solutions_l1497_149702


namespace mary_money_left_l1497_149759

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 2 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem stating that Mary has 50 - 8q dollars left after her purchases -/
theorem mary_money_left (q : ℝ) : money_left q = 50 - 8 * q := by
  sorry

end mary_money_left_l1497_149759


namespace exponents_in_30_factorial_l1497_149748

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def exponent_in_factorial (p n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / p) 0

theorem exponents_in_30_factorial :
  exponent_in_factorial 2 30 = 26 ∧ exponent_in_factorial 5 30 = 7 := by
  sorry

end exponents_in_30_factorial_l1497_149748


namespace sqrt_400_div_2_l1497_149730

theorem sqrt_400_div_2 : Real.sqrt 400 / 2 = 10 := by
  sorry

end sqrt_400_div_2_l1497_149730


namespace difference_squared_equals_negative_sixteen_l1497_149718

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : a^2 + 8 > 0)  -- Ensure a^2 + 8 is positive to avoid division by zero
variable (h2 : a * b = 12)

-- State the theorem
theorem difference_squared_equals_negative_sixteen : (a - b)^2 = -16 := by
  sorry

end difference_squared_equals_negative_sixteen_l1497_149718


namespace specific_plot_fencing_cost_l1497_149778

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  perimeter : ℝ

/-- The total cost of fencing for a rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem specific_plot_fencing_cost :
  ∃ (plot : RectangularPlot),
    plot.length = plot.width + 10 ∧
    plot.perimeter = 220 ∧
    plot.fencing_cost_per_meter = 6.5 ∧
    total_fencing_cost plot = 1430 := by
  sorry

end specific_plot_fencing_cost_l1497_149778


namespace not_perfect_square_l1497_149764

theorem not_perfect_square (n : ℤ) : ¬ ∃ m : ℤ, m^2 = 4*n + 3 := by
  sorry

end not_perfect_square_l1497_149764


namespace coefficient_of_x_is_negative_one_l1497_149737

-- Define the expression as a polynomial
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 3 * (8 - 3 * x^2 + 7 * x) - 9 * (3 * x - 2)

-- Theorem stating that the coefficient of x in the expression is -1
theorem coefficient_of_x_is_negative_one :
  ∃ (a b c : ℝ), expression = fun x => a * x^2 + (-1) * x + c :=
by
  sorry

end coefficient_of_x_is_negative_one_l1497_149737


namespace P_positive_P_surjective_l1497_149794

/-- A polynomial in two real variables that takes only positive values and achieves all positive values -/
def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

/-- The polynomial P is always positive for any real x and y -/
theorem P_positive (x y : ℝ) : P x y > 0 := by sorry

/-- For any positive real t, there exist real x and y such that P(x,y) = t -/
theorem P_surjective (t : ℝ) (ht : t > 0) : ∃ x y : ℝ, P x y = t := by sorry

end P_positive_P_surjective_l1497_149794


namespace inverse_proportion_ordering_l1497_149774

theorem inverse_proportion_ordering (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : y₃ = 2 / x₃) 
  (h4 : x₁ < x₂) 
  (h5 : x₂ < 0) 
  (h6 : 0 < x₃) : 
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end inverse_proportion_ordering_l1497_149774


namespace problem_solution_l1497_149769

theorem problem_solution (x : ℝ) (h : x + 1/x = 5) :
  (x - 3)^2 + 36/((x - 3)^2) = 15 := by
  sorry

end problem_solution_l1497_149769


namespace monthly_payment_calculation_l1497_149796

def original_price : ℝ := 480
def discount_percentage : ℝ := 5
def first_installment : ℝ := 150
def num_installments : ℕ := 3

theorem monthly_payment_calculation :
  let discounted_price := original_price * (1 - discount_percentage / 100)
  let remaining_balance := discounted_price - first_installment
  let monthly_payment := remaining_balance / num_installments
  monthly_payment = 102 := by
  sorry

end monthly_payment_calculation_l1497_149796


namespace tetrahedrons_from_cube_l1497_149719

/-- A cube has 8 vertices -/
def cube_vertices : ℕ := 8

/-- The number of tetrahedrons that can be formed using the vertices of a cube -/
def num_tetrahedrons : ℕ := 58

/-- Theorem: The number of tetrahedrons that can be formed using the vertices of a cube is 58 -/
theorem tetrahedrons_from_cube : num_tetrahedrons = 58 := by
  sorry

end tetrahedrons_from_cube_l1497_149719


namespace range_of_a_l1497_149750

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x, p x → q x a) →
  (∃ x, ¬p x ∧ q x a) →
  a ≥ 9 :=
sorry

end range_of_a_l1497_149750


namespace rubber_band_area_l1497_149780

/-- Represents a nail on the board -/
structure Nail where
  x : ℝ
  y : ℝ

/-- Represents the quadrilateral formed by the rubber band -/
structure Quadrilateral where
  nails : Fin 4 → Nail

/-- The area of a quadrilateral formed by a rubber band looped around four nails arranged in a 2x2 grid with 1 unit spacing -/
def quadrilateralArea (q : Quadrilateral) : ℝ :=
  sorry

/-- The theorem stating that the area of the quadrilateral is 6 square units -/
theorem rubber_band_area (q : Quadrilateral) 
  (h1 : q.nails 0 = ⟨0, 0⟩)
  (h2 : q.nails 1 = ⟨1, 0⟩)
  (h3 : q.nails 2 = ⟨0, 1⟩)
  (h4 : q.nails 3 = ⟨1, 1⟩) :
  quadrilateralArea q = 6 :=
sorry

end rubber_band_area_l1497_149780


namespace jack_morning_emails_l1497_149735

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 7

theorem jack_morning_emails :
  morning_emails = 10 :=
by
  sorry

end jack_morning_emails_l1497_149735


namespace inverse_congruence_solution_l1497_149708

theorem inverse_congruence_solution (p : ℕ) (a : ℤ) (hp : Nat.Prime p) :
  (∃ x : ℤ, (a * x) % p = 1 ∧ x % p = a % p) ↔ (a % p = 1 ∨ a % p = p - 1) := by
  sorry

end inverse_congruence_solution_l1497_149708


namespace min_value_expression_l1497_149705

theorem min_value_expression (a b c : ℝ) 
  (sum_condition : a + b + c = -1) 
  (product_condition : a * b * c ≤ -3) :
  (a * b + 1) / (a + b) + (b * c + 1) / (b + c) + (c * a + 1) / (c + a) ≥ 3 :=
by sorry

end min_value_expression_l1497_149705


namespace opposite_of_fraction_l1497_149701

theorem opposite_of_fraction : 
  -(11 / 2022 : ℚ) = -11 / 2022 := by sorry

end opposite_of_fraction_l1497_149701


namespace function_eventually_constant_l1497_149742

def is_eventually_constant (f : ℕ+ → ℕ+) : Prop :=
  ∃ m : ℕ+, ∀ x ≥ m, f x = f m

theorem function_eventually_constant
  (f : ℕ+ → ℕ+)
  (h1 : ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1))
  (h2 : ∀ x : ℕ+, f x < 2000) :
  is_eventually_constant f :=
sorry

end function_eventually_constant_l1497_149742


namespace window_width_theorem_l1497_149728

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the configuration of a window -/
structure Window where
  pane : Pane
  grid_width : Nat
  grid_height : Nat
  border_width : ℝ

/-- Calculates the total width of a window -/
def total_window_width (w : Window) : ℝ :=
  w.grid_width * w.pane.width + (w.grid_width + 1) * w.border_width

/-- Theorem stating the total width of the window -/
theorem window_width_theorem (w : Window) 
  (h1 : w.grid_width = 3)
  (h2 : w.grid_height = 3)
  (h3 : w.border_width = 3)
  : total_window_width w = 3 * w.pane.width + 12 := by
  sorry

#check window_width_theorem

end window_width_theorem_l1497_149728


namespace jacob_michael_age_difference_l1497_149733

theorem jacob_michael_age_difference :
  ∀ (jacob_age michael_age : ℕ),
    (jacob_age + 4 = 11) →
    (michael_age + 5 = 2 * (jacob_age + 5)) →
    (michael_age - jacob_age = 12) :=
by sorry

end jacob_michael_age_difference_l1497_149733


namespace sum_of_fourth_powers_of_primes_mod_240_l1497_149758

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the fourth powers of the first n prime numbers -/
def sumOfFourthPowersOfPrimes (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => (nthPrime (i + 1)) ^ 4)

/-- The main theorem -/
theorem sum_of_fourth_powers_of_primes_mod_240 :
  sumOfFourthPowersOfPrimes 2014 % 240 = 168 := by sorry

end sum_of_fourth_powers_of_primes_mod_240_l1497_149758


namespace cube_root_unity_sum_l1497_149715

/-- Define ω as a complex number satisfying the properties of a cube root of unity -/
def ω : ℂ := sorry

/-- ω is a cube root of unity -/
axiom ω_cubed : ω^3 = 1

/-- ω satisfies the equation ω^2 + ω + 1 = 0 -/
axiom ω_sum : ω^2 + ω + 1 = 0

/-- Theorem: ω^9 + (ω^2)^9 = 2 -/
theorem cube_root_unity_sum : ω^9 + (ω^2)^9 = 2 := by sorry

end cube_root_unity_sum_l1497_149715


namespace china_first_negative_numbers_l1497_149787

-- Define an enumeration for the countries
inductive Country
  | France
  | China
  | England
  | UnitedStates

-- Define a function that represents the property of being the first country to recognize and use negative numbers
def firstToUseNegativeNumbers : Country → Prop :=
  fun c => c = Country.China

-- Theorem statement
theorem china_first_negative_numbers :
  ∃ c : Country, firstToUseNegativeNumbers c ∧
  (c = Country.France ∨ c = Country.China ∨ c = Country.England ∨ c = Country.UnitedStates) :=
by
  sorry


end china_first_negative_numbers_l1497_149787


namespace z_power_2000_eq_one_l1497_149707

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (3 + 4i) / (4 - 3i) -/
noncomputable def z : ℂ := (3 + 4 * i) / (4 - 3 * i)

/-- Theorem stating that z^2000 = 1 -/
theorem z_power_2000_eq_one : z ^ 2000 = 1 := by sorry

end z_power_2000_eq_one_l1497_149707


namespace arithmetic_sequence_sum_30_l1497_149786

/-- An arithmetic sequence and its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with given partial sums, S_30 can be determined -/
theorem arithmetic_sequence_sum_30 (seq : ArithmeticSequence) 
  (h10 : seq.S 10 = 10) (h20 : seq.S 20 = 30) : seq.S 30 = 60 := by
  sorry

end arithmetic_sequence_sum_30_l1497_149786


namespace triangle_max_height_l1497_149793

/-- In a triangle ABC with sides a, b, c corresponding to angles A, B, C respectively,
    given that c = 1 and a*cos(B) + b*cos(A) = 2*cos(C),
    the maximum value of the height h on side AB is √3/2 -/
theorem triangle_max_height (a b c : ℝ) (A B C : ℝ) (h : ℝ) :
  c = 1 →
  a * Real.cos B + b * Real.cos A = 2 * Real.cos C →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  h ≤ Real.sqrt 3 / 2 ∧ ∃ (a' b' : ℝ), h = Real.sqrt 3 / 2 :=
by sorry

end triangle_max_height_l1497_149793


namespace exists_finite_harmonic_progression_no_infinite_harmonic_progression_l1497_149772

-- Define a harmonic progression
def IsHarmonicProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℚ, ∀ k : ℕ, k > 0 → (1 : ℚ) / a (k + 1) - (1 : ℚ) / a k = d

-- Part (a)
theorem exists_finite_harmonic_progression (N : ℕ) :
  ∃ a : ℕ → ℕ, (∀ k : ℕ, k < N → a k < a (k + 1)) ∧ IsHarmonicProgression a :=
sorry

-- Part (b)
theorem no_infinite_harmonic_progression :
  ¬ ∃ a : ℕ → ℕ, (∀ k : ℕ, a k < a (k + 1)) ∧ IsHarmonicProgression a :=
sorry

end exists_finite_harmonic_progression_no_infinite_harmonic_progression_l1497_149772


namespace partition_positive_integers_l1497_149799

theorem partition_positive_integers : ∃ (A B : Set ℕ), 
  (∀ n : ℕ, n > 0 → (n ∈ A ∨ n ∈ B)) ∧ 
  (A ∩ B = ∅) ∧
  (∀ a b c : ℕ, a ∈ A → b ∈ A → c ∈ A → a < b → b < c → b - a ≠ c - b) ∧
  (∀ f : ℕ → ℕ, (∀ n : ℕ, f n ∈ B) → 
    ∃ i j k : ℕ, i < j ∧ j < k ∧ f j - f i ≠ f k - f j) :=
sorry

end partition_positive_integers_l1497_149799


namespace percentage_difference_l1497_149756

theorem percentage_difference : (62 / 100 * 150) - (20 / 100 * 250) = 43 := by
  sorry

end percentage_difference_l1497_149756


namespace tan_sum_pi_24_and_7pi_24_l1497_149741

theorem tan_sum_pi_24_and_7pi_24 : 
  Real.tan (π / 24) + Real.tan (7 * π / 24) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 := by
  sorry

end tan_sum_pi_24_and_7pi_24_l1497_149741


namespace right_triangle_legs_l1497_149736

theorem right_triangle_legs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 37^2 → 
  a * b = (a + 7) * (b - 2) →
  (a = 35 ∧ b = 12) ∨ (a = 12 ∧ b = 35) := by
sorry

end right_triangle_legs_l1497_149736


namespace lisa_works_32_hours_l1497_149724

/-- Given Greta's work hours, Greta's hourly wage, and Lisa's hourly wage,
    calculate the number of hours Lisa needs to work to equal Greta's earnings. -/
def lisa_equal_hours (greta_hours : ℕ) (greta_wage : ℚ) (lisa_wage : ℚ) : ℚ :=
  (greta_hours : ℚ) * greta_wage / lisa_wage

/-- Prove that Lisa needs to work 32 hours to equal Greta's earnings. -/
theorem lisa_works_32_hours :
  lisa_equal_hours 40 12 15 = 32 := by
  sorry

end lisa_works_32_hours_l1497_149724


namespace sum_of_four_primes_divisible_by_60_l1497_149775

theorem sum_of_four_primes_divisible_by_60 (p q r s : ℕ) :
  Prime p → Prime q → Prime r → Prime s →
  5 < p → p < q → q < r → r < s → s < p + 10 →
  60 ∣ (p + q + r + s) := by
sorry

end sum_of_four_primes_divisible_by_60_l1497_149775


namespace square_dissection_ratio_l1497_149749

/-- A square dissection problem -/
theorem square_dissection_ratio (A B E F G X Y W Z : ℝ × ℝ) : 
  let square_side : ℝ := 4
  let AE : ℝ := 1
  let BF : ℝ := 4
  let EF : ℝ := 2
  let AG : ℝ := 4
  let BG : ℝ := Real.sqrt 17
  -- AG perpendicular to BF
  (AG * BF = 0) →
  -- Area preservation
  (square_side * square_side = XY * WZ) →
  -- XY equals AG
  (XY = AG) →
  -- Ratio calculation
  (XY / WZ = 1) := by
  sorry

end square_dissection_ratio_l1497_149749


namespace truck_distance_problem_l1497_149729

/-- Proves that the initial distance between two trucks is 1025 km given the problem conditions --/
theorem truck_distance_problem (speed_A speed_B : ℝ) (extra_distance : ℝ) :
  speed_A = 90 →
  speed_B = 80 →
  extra_distance = 145 →
  ∃ (time : ℝ), 
    time > 0 ∧
    speed_A * (time + 1) = speed_B * time + extra_distance ∧
    speed_A * (time + 1) + speed_B * time = 1025 :=
by sorry

end truck_distance_problem_l1497_149729


namespace derivative_f_at_one_l1497_149763

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_f_at_one :
  deriv f 1 = 1 := by
  sorry

end derivative_f_at_one_l1497_149763


namespace complex_number_imaginary_part_l1497_149716

theorem complex_number_imaginary_part (i : ℂ) (a : ℝ) :
  i * i = -1 →
  let z := (1 - a * i) / (1 + i)
  Complex.im z = -3 →
  a = 5 := by sorry

end complex_number_imaginary_part_l1497_149716
