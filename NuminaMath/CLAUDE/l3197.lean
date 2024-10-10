import Mathlib

namespace least_subtrahend_for_divisibility_specific_case_l3197_319712

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : Prime d) :
  let r := n % d
  r = (n - (n - r)) % d ∧ 
  ∀ m : Nat, m < r → (n - m) % d ≠ 0 :=
by
  sorry

#eval 2376819 % 139  -- This should evaluate to 135

theorem specific_case : 
  let n := 2376819
  let d := 139
  Prime d ∧ 
  (n - 135) % d = 0 ∧
  ∀ m : Nat, m < 135 → (n - m) % d ≠ 0 :=
by
  sorry

end least_subtrahend_for_divisibility_specific_case_l3197_319712


namespace third_side_length_l3197_319768

theorem third_side_length (a b c : ℝ) : 
  a = 4 → b = 10 → c = 12 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end third_side_length_l3197_319768


namespace relations_correctness_l3197_319710

-- Define the relations
def relation1 (a b c : ℝ) : Prop := (a > b) ↔ (a * c^2 > b * c^2)
def relation2 (a b : ℝ) : Prop := (a > b) → (1/a < 1/b)
def relation3 (a b c d : ℝ) : Prop := (a > b ∧ b > 0 ∧ c > d) → (a/d > b/c)
def relation4 (a b c : ℝ) : Prop := (a > b ∧ b > 1 ∧ c < 0) → (a^c < b^c)

-- State the theorem
theorem relations_correctness :
  (∃ a b c : ℝ, ¬(relation1 a b c)) ∧
  (∃ a b : ℝ, ¬(relation2 a b)) ∧
  (∃ a b c d : ℝ, ¬(relation3 a b c d)) ∧
  (∀ a b c : ℝ, relation4 a b c) :=
sorry

end relations_correctness_l3197_319710


namespace fourth_vertex_coordinates_l3197_319786

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

/-- The theorem stating the possible coordinates of the fourth vertex of the parallelogram -/
theorem fourth_vertex_coordinates :
  let A : Point := ⟨0, -9⟩
  let B : Point := ⟨2, 6⟩
  let C : Point := ⟨4, 5⟩
  ∃ D : Point, (D = ⟨2, -10⟩ ∨ D = ⟨-2, -8⟩ ∨ D = ⟨6, 20⟩) ∧ 
    is_parallelogram A B C D :=
by
  sorry


end fourth_vertex_coordinates_l3197_319786


namespace expression_evaluation_l3197_319762

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ((x^2 - 1)^2 * (x^3 - x^2 + 1)^2 / (x^5 - 1)^2)^2 * 
  ((x^2 + 1)^2 * (x^3 + x^2 + 1)^2 / (x^5 + 1)^2)^2 = 1 := by
sorry

end expression_evaluation_l3197_319762


namespace average_shift_l3197_319702

theorem average_shift (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 40) :
  ((x₁ + 40) + (x₂ + 40) + (x₃ + 40)) / 3 = 80 := by
  sorry

end average_shift_l3197_319702


namespace trigonometric_identity_l3197_319763

theorem trigonometric_identity (x y : Real) 
  (h : Real.cos (x + y) = 2 / 3) : 
  Real.sin (x - 3 * Real.pi / 10) * Real.cos (y - Real.pi / 5) - 
  Real.sin (x + Real.pi / 5) * Real.cos (y + 3 * Real.pi / 10) = 
  -2 / 3 := by sorry

end trigonometric_identity_l3197_319763


namespace hyperbola_equation_l3197_319727

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 16) →
  (b / a = Real.sqrt 55 / 11) →
  (∀ x y : ℝ, x^2 / 11 - y^2 / 5 = 1) :=
by sorry

end hyperbola_equation_l3197_319727


namespace triangle_existence_l3197_319770

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the theorem
theorem triangle_existence (m : ℝ) : 
  (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c →
   f m a + f m b > f m c ∧
   f m a + f m c > f m b ∧
   f m b + f m c > f m a) ↔
  m > 6 := by sorry

end triangle_existence_l3197_319770


namespace two_hour_charge_is_174_l3197_319719

/-- Represents the pricing model for therapy sessions -/
structure TherapyPricing where
  first_hour : ℕ
  additional_hour : ℕ
  first_hour_premium : first_hour = additional_hour + 40

/-- Calculates the total charge for a given number of hours -/
def total_charge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.first_hour + (hours - 1) * pricing.additional_hour

/-- Theorem stating the correct charge for 2 hours given the conditions -/
theorem two_hour_charge_is_174 (pricing : TherapyPricing) 
  (h1 : total_charge pricing 5 = 375) : 
  total_charge pricing 2 = 174 := by
  sorry

end two_hour_charge_is_174_l3197_319719


namespace root_sum_reciprocal_l3197_319721

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (1/(a+2) + 1/(b+2) + 1/(c+2) = 3/2) := by
sorry

end root_sum_reciprocal_l3197_319721


namespace smallest_valid_number_l3197_319715

def is_valid (n : ℕ) : Prop :=
  n > 9 ∧
  ¬(n % 7 = 0) ∧
  ∀ (i : ℕ), i < (String.length (toString n)) →
    ((n.div (10^i) % 10) ≠ 7) ∧
    (((n - (n.div (10^i) % 10) * 10^i + 7 * 10^i) % 7 = 0))

theorem smallest_valid_number :
  is_valid 13264513 ∧ ∀ (m : ℕ), m < 13264513 → ¬(is_valid m) := by sorry

end smallest_valid_number_l3197_319715


namespace new_average_age_with_teacher_l3197_319718

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℕ) 
  (h1 : num_students = 50) 
  (h2 : student_avg_age = 14) 
  (h3 : teacher_age = 65) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 15 := by
sorry

end new_average_age_with_teacher_l3197_319718


namespace thirteen_percent_problem_l3197_319793

theorem thirteen_percent_problem : ∃ x : ℝ, 
  (13 / 100) * x = 85 ∧ 
  Int.floor (x + 0.5) = 654 := by
  sorry

end thirteen_percent_problem_l3197_319793


namespace colten_chickens_l3197_319725

theorem colten_chickens (total : ℕ) (q s c : ℕ) : 
  total = 383 →
  q = 2 * s + 25 →
  s = 3 * c - 4 →
  q + s + c = total →
  c = 37 := by
sorry

end colten_chickens_l3197_319725


namespace extreme_point_implies_zero_derivative_zero_derivative_not_always_extreme_point_l3197_319765

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define differentiability for f
variable (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extreme point
def IsExtremePoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem extreme_point_implies_zero_derivative
  (x₀ : ℝ) (h_extreme : IsExtremePoint f x₀) :
  deriv f x₀ = 0 :=
sorry

-- State that the converse is not always true
theorem zero_derivative_not_always_extreme_point :
  ¬ (∀ (g : ℝ → ℝ) (hg : Differentiable ℝ g) (x₀ : ℝ),
    deriv g x₀ = 0 → IsExtremePoint g x₀) :=
sorry

end extreme_point_implies_zero_derivative_zero_derivative_not_always_extreme_point_l3197_319765


namespace complex_modulus_problem_l3197_319741

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l3197_319741


namespace jeremy_earnings_l3197_319754

theorem jeremy_earnings (steven_rate mark_rate steven_rooms mark_rooms : ℚ) 
  (h1 : steven_rate = 12 / 3)
  (h2 : mark_rate = 10 / 4)
  (h3 : steven_rooms = 8 / 3)
  (h4 : mark_rooms = 9 / 4) :
  steven_rate * steven_rooms + mark_rate * mark_rooms = 391 / 24 := by
  sorry

end jeremy_earnings_l3197_319754


namespace cone_surface_area_l3197_319785

/-- 
Given a cone with slant height 2 and lateral surface that unfolds into a semicircle,
prove that its surface area is 3π.
-/
theorem cone_surface_area (h : ℝ) (r : ℝ) : 
  h = 2 ∧ 2 * π * r = 2 * π → π * r * h + π * r^2 = 3 * π :=
by sorry

end cone_surface_area_l3197_319785


namespace bounded_difference_l3197_319739

theorem bounded_difference (x y z : ℝ) :
  x - z < y ∧ x + z > y → -z < x - y ∧ x - y < z := by
  sorry

end bounded_difference_l3197_319739


namespace base_7_sum_theorem_l3197_319744

def base_7_to_decimal (a b c : Nat) : Nat :=
  7^2 * a + 7 * b + c

theorem base_7_sum_theorem (A B C : Nat) :
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 7 ∧ B < 7 ∧ C < 7 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  base_7_to_decimal A B C + base_7_to_decimal B C A + base_7_to_decimal C A B = base_7_to_decimal A A A + 1 →
  B + C = 6 := by
sorry

end base_7_sum_theorem_l3197_319744


namespace smallest_three_digit_multiple_of_13_l3197_319723

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ 13 ∣ n → n ≥ 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l3197_319723


namespace opposite_face_points_are_diametrically_opposite_l3197_319738

-- Define a cube
structure Cube where
  side_length : ℝ
  center : ℝ × ℝ × ℝ

-- Define a point on the surface of a cube
structure CubePoint where
  coordinates : ℝ × ℝ × ℝ
  cube : Cube
  on_surface : Bool

-- Define the concept of diametrically opposite points
def diametrically_opposite (p1 p2 : CubePoint) (c : Cube) : Prop :=
  ∃ (t : ℝ), 
    p1.coordinates = (1 - t) • c.center + t • p2.coordinates ∧ 
    0 ≤ t ∧ t ≤ 1

-- Define opposite faces
def opposite_faces (f1 f2 : CubePoint → Prop) (c : Cube) : Prop :=
  ∀ (p1 p2 : CubePoint), f1 p1 → f2 p2 → diametrically_opposite p1 p2 c

-- Theorem statement
theorem opposite_face_points_are_diametrically_opposite 
  (c : Cube) (p s : CubePoint) (f1 f2 : CubePoint → Prop) :
  opposite_faces f1 f2 c →
  f1 p →
  f2 s →
  diametrically_opposite p s c :=
sorry

end opposite_face_points_are_diametrically_opposite_l3197_319738


namespace incorrect_statement_about_immunity_l3197_319731

-- Define the three lines of defense
inductive LineOfDefense
| First
| Second
| Third

-- Define the types of immunity
inductive ImmunityType
| NonSpecific
| Specific

-- Define the components of each line of defense
def componentsOfDefense (line : LineOfDefense) : String :=
  match line with
  | .First => "skin and mucous membranes"
  | .Second => "antimicrobial substances and phagocytic cells in body fluids"
  | .Third => "immune organs and immune cells"

-- Define the type of immunity for each line of defense
def immunityTypeOfDefense (line : LineOfDefense) : ImmunityType :=
  match line with
  | .First => .NonSpecific
  | .Second => .NonSpecific
  | .Third => .Specific

-- Theorem to prove
theorem incorrect_statement_about_immunity :
  ¬(∀ (line : LineOfDefense), immunityTypeOfDefense line = .NonSpecific) :=
by sorry

end incorrect_statement_about_immunity_l3197_319731


namespace expand_expression_l3197_319704

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 := by
  sorry

end expand_expression_l3197_319704


namespace complex_power_modulus_l3197_319729

theorem complex_power_modulus : Complex.abs ((4 + 2*Complex.I)^5) = 160 * Real.sqrt 5 := by
  sorry

end complex_power_modulus_l3197_319729


namespace cube_sum_given_sum_and_product_l3197_319792

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end cube_sum_given_sum_and_product_l3197_319792


namespace xyz_congruence_l3197_319761

theorem xyz_congruence (x y z : ℕ) : 
  x < 8 → y < 8 → z < 8 →
  (x + 3*y + 2*z) % 8 = 1 →
  (2*x + y + 3*z) % 8 = 5 →
  (3*x + 2*y + z) % 8 = 3 →
  (x*y*z) % 8 = 0 := by
sorry

end xyz_congruence_l3197_319761


namespace point_distance_constraint_l3197_319711

/-- Given points A(1, 0) and B(4, 0), and a point P on the line x + my - 1 = 0
    such that |PA| = 2|PB|, prove that the range of values for m is m ≥ √3 or m ≤ -√3. -/
theorem point_distance_constraint (m : ℝ) : 
  (∃ (x y : ℝ), x + m * y - 1 = 0 ∧ 
   (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)) ↔ 
  (m ≥ Real.sqrt 3 ∨ m ≤ -Real.sqrt 3) :=
sorry

end point_distance_constraint_l3197_319711


namespace paragraphs_per_page_is_twenty_l3197_319790

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 50

/-- Represents the total time taken to read the book in hours -/
def total_reading_time : ℕ := 50

/-- Calculates the number of paragraphs per page in the book -/
def paragraphs_per_page : ℕ :=
  (reading_speed * total_reading_time) / (sentences_per_paragraph * total_pages)

/-- Theorem stating that the number of paragraphs per page is 20 -/
theorem paragraphs_per_page_is_twenty :
  paragraphs_per_page = 20 := by
  sorry

end paragraphs_per_page_is_twenty_l3197_319790


namespace thirty_five_only_math_l3197_319781

/-- Represents the number of students in various class combinations -/
structure ClassCounts where
  total : ℕ
  math : ℕ
  foreign : ℕ
  sport : ℕ
  all_three : ℕ

/-- Calculates the number of students taking only math class -/
def only_math (counts : ClassCounts) : ℕ :=
  counts.math - (counts.total - (counts.math + counts.foreign + counts.sport - counts.all_three))

/-- Theorem stating that 35 students take only math class given the specific class counts -/
theorem thirty_five_only_math (counts : ClassCounts) 
  (h_total : counts.total = 120)
  (h_math : counts.math = 85)
  (h_foreign : counts.foreign = 65)
  (h_sport : counts.sport = 50)
  (h_all_three : counts.all_three = 10) :
  only_math counts = 35 := by
  sorry

end thirty_five_only_math_l3197_319781


namespace quadratic_discriminant_l3197_319703

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  |x₂ - x₁| = 2 →
  b^2 - 4*a*c = 4 := by
sorry

end quadratic_discriminant_l3197_319703


namespace unique_positive_root_l3197_319799

/-- The polynomial function f(x) = x^12 + 5x^11 - 3x^10 + 2000x^9 - 1500x^8 -/
def f (x : ℝ) : ℝ := x^12 + 5*x^11 - 3*x^10 + 2000*x^9 - 1500*x^8

/-- The theorem stating that f(x) has exactly one positive real root -/
theorem unique_positive_root : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

end unique_positive_root_l3197_319799


namespace college_students_count_l3197_319751

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 135) :
  boys + girls = 351 :=
sorry

end college_students_count_l3197_319751


namespace work_completion_larger_group_size_l3197_319779

theorem work_completion (work_days : ℕ) (small_group : ℕ) (large_group_days : ℕ) : ℕ :=
  let total_man_days := work_days * small_group
  total_man_days / large_group_days

theorem larger_group_size : work_completion 25 12 15 = 20 := by
  sorry

end work_completion_larger_group_size_l3197_319779


namespace hannah_leah_study_difference_l3197_319747

theorem hannah_leah_study_difference (daily_differences : List Int) 
  (h1 : daily_differences = [15, -5, 25, -15, 35, 0, 20]) 
  (days_in_week : Nat) (h2 : days_in_week = 7) : 
  Int.floor ((daily_differences.sum : ℚ) / days_in_week) = 10 := by
  sorry

end hannah_leah_study_difference_l3197_319747


namespace building_entrances_l3197_319746

/-- Represents a multi-story building with apartments --/
structure Building where
  floors : ℕ
  apartments_per_floor : ℕ
  total_apartments : ℕ

/-- Calculates the number of entrances in a building --/
def number_of_entrances (b : Building) : ℕ :=
  b.total_apartments / (b.floors * b.apartments_per_floor)

/-- Theorem: A building with 9 floors, 4 apartments per floor, and 180 total apartments has 5 entrances --/
theorem building_entrances :
  let b : Building := ⟨9, 4, 180⟩
  number_of_entrances b = 5 := by
sorry

end building_entrances_l3197_319746


namespace addition_puzzle_l3197_319764

theorem addition_puzzle (x y : ℕ) : 
  x ≠ y →
  x < 10 →
  y < 10 →
  307 + 700 + x = 1010 →
  y - x = 7 :=
by sorry

end addition_puzzle_l3197_319764


namespace factorial_of_factorial_divided_by_factorial_l3197_319774

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_of_factorial_divided_by_factorial_l3197_319774


namespace line_tangent_to_parabola_l3197_319782

/-- The value of c for which the line y = 3x + c is tangent to the parabola y² = 12x -/
def tangent_line_c : ℝ := 1

/-- The line equation: y = 3x + c -/
def line_equation (x y c : ℝ) : Prop := y = 3 * x + c

/-- The parabola equation: y² = 12x -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 12 * x

/-- The line y = 3x + c is tangent to the parabola y² = 12x when c = tangent_line_c -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), line_equation x y tangent_line_c ∧ parabola_equation x y :=
sorry

end line_tangent_to_parabola_l3197_319782


namespace sum_of_special_primes_is_prime_l3197_319732

theorem sum_of_special_primes_is_prime (C D : ℕ+) : 
  Prime C.val → Prime D.val → Prime (C.val - D.val) → Prime (C.val + D.val) →
  Prime (C.val + D.val + (C.val - D.val) + C.val + D.val) := by
sorry

end sum_of_special_primes_is_prime_l3197_319732


namespace monomial_coefficient_and_degree_l3197_319788

/-- Represents a monomial in two variables -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  x_exp : ℕ
  y_exp : ℕ

/-- The degree of a monomial is the sum of its exponents -/
def Monomial.degree {α : Type*} [Ring α] (m : Monomial α) : ℕ :=
  m.x_exp + m.y_exp

theorem monomial_coefficient_and_degree :
  let m : Monomial ℤ := { coeff := -2, x_exp := 1, y_exp := 3 }
  (m.coeff = -2) ∧ (m.degree = 4) := by sorry

end monomial_coefficient_and_degree_l3197_319788


namespace group_formation_count_l3197_319776

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of groups that can be formed with one boy and two girls -/
def oneBoytwoGirls (boys girls : ℕ) : ℕ := 
  binomial boys 1 * binomial girls 2

/-- The number of groups that can be formed with two boys and one girl -/
def twoBoyoneGirl (boys girls : ℕ) : ℕ := 
  binomial boys 2 * binomial girls 1

/-- The total number of valid groups that can be formed -/
def totalGroups (boys girls : ℕ) : ℕ := 
  oneBoytwoGirls boys girls + twoBoyoneGirl boys girls

theorem group_formation_count :
  totalGroups 9 12 = 1026 := by sorry

end group_formation_count_l3197_319776


namespace vase_original_price_l3197_319784

/-- Proves that given a vase with an original price P, which is discounted by 25% 
    and then has a 10% sales tax applied, if the total price paid is $165, 
    then the original price P must be $200. -/
theorem vase_original_price (P : ℝ) : 
  (P * (1 - 0.25) * (1 + 0.1) = 165) → P = 200 := by
  sorry

end vase_original_price_l3197_319784


namespace number_puzzle_l3197_319700

theorem number_puzzle : ∃ x : ℝ, x / 3 = x - 36 ∧ x = 54 := by
  sorry

end number_puzzle_l3197_319700


namespace n_times_n_plus_one_div_by_three_l3197_319707

theorem n_times_n_plus_one_div_by_three (n : ℕ) (h : 1 ≤ n ∧ n ≤ 99) : 
  3 ∣ (n * (n + 1)) := by
sorry

end n_times_n_plus_one_div_by_three_l3197_319707


namespace unique_number_between_cube_roots_l3197_319756

theorem unique_number_between_cube_roots : ∃! (n : ℕ),
  n > 0 ∧ 24 ∣ n ∧ (9 : ℝ) < n ^ (1/3) ∧ n ^ (1/3) < (9.1 : ℝ) :=
by sorry

end unique_number_between_cube_roots_l3197_319756


namespace positive_solution_of_equation_l3197_319773

theorem positive_solution_of_equation : ∃ x : ℝ, x > 0 ∧ 
  (1/2) * (4 * x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 35*x + 7) ∧
  x = (75 + Real.sqrt 5681) / 2 := by
  sorry

end positive_solution_of_equation_l3197_319773


namespace smallest_d_value_l3197_319705

theorem smallest_d_value : ∃ (d : ℝ), d ≥ 0 ∧ 
  (3 * Real.sqrt 5)^2 + (d + 3)^2 = (3 * d)^2 ∧
  (∀ (x : ℝ), x ≥ 0 ∧ (3 * Real.sqrt 5)^2 + (x + 3)^2 = (3 * x)^2 → d ≤ x) ∧
  d = 3 := by
  sorry

end smallest_d_value_l3197_319705


namespace ellipse_sum_property_l3197_319709

/-- Properties of an ellipse -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- length of semi-major axis
  b : ℝ  -- length of semi-minor axis

/-- Theorem about the sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_property (E : Ellipse) 
  (center_x : E.h = 3) 
  (center_y : E.k = -5) 
  (major_axis : E.a = 6) 
  (minor_axis : E.b = 2) : 
  E.h + E.k + E.a + E.b = 6 := by
  sorry

#check ellipse_sum_property

end ellipse_sum_property_l3197_319709


namespace max_candy_leftover_l3197_319797

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end max_candy_leftover_l3197_319797


namespace abc_sum_sqrt_l3197_319767

theorem abc_sum_sqrt (a b c : ℝ) 
  (eq1 : b + c = 20) 
  (eq2 : c + a = 22) 
  (eq3 : a + b = 24) : 
  Real.sqrt (a * b * c * (a + b + c)) = 357 := by
  sorry

end abc_sum_sqrt_l3197_319767


namespace variance_transformation_l3197_319750

def data_variance (data : List ℝ) : ℝ := sorry

theorem variance_transformation (data : List ℝ) (h : data.length = 2010) 
  (h_var : data_variance data = 2) :
  data_variance (data.map (λ x => -3 * x + 1)) = 18 := by sorry

end variance_transformation_l3197_319750


namespace min_balls_for_three_colors_l3197_319743

theorem min_balls_for_three_colors (num_colors : Nat) (balls_per_color : Nat) 
  (h1 : num_colors = 4) (h2 : balls_per_color = 13) :
  (2 * balls_per_color + 1) = 27 := by
  sorry

end min_balls_for_three_colors_l3197_319743


namespace assignment_statement_valid_l3197_319726

-- Define what constitutes a valid variable name
def IsValidVariableName (name : String) : Prop := name.length > 0 ∧ name.all Char.isAlpha

-- Define what constitutes a valid arithmetic expression
inductive ArithmeticExpression
  | Var : String → ArithmeticExpression
  | Num : Int → ArithmeticExpression
  | Add : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression
  | Mul : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression
  | Sub : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression

-- Define what constitutes a valid assignment statement
structure AssignmentStatement where
  lhs : String
  rhs : ArithmeticExpression
  valid : IsValidVariableName lhs

-- The statement we want to prove
theorem assignment_statement_valid :
  ∃ (stmt : AssignmentStatement),
    stmt.lhs = "A" ∧
    stmt.rhs = ArithmeticExpression.Sub
      (ArithmeticExpression.Add
        (ArithmeticExpression.Mul
          (ArithmeticExpression.Var "A")
          (ArithmeticExpression.Var "A"))
        (ArithmeticExpression.Var "A"))
      (ArithmeticExpression.Num 3) :=
by sorry


end assignment_statement_valid_l3197_319726


namespace contrapositive_equivalence_l3197_319757

theorem contrapositive_equivalence (a b x : ℝ) :
  (x ≥ a^2 + b^2 → x ≥ 2*a*b) ↔ (x < 2*a*b → x < a^2 + b^2) := by sorry

end contrapositive_equivalence_l3197_319757


namespace only_parallelogram_coincides_l3197_319777

-- Define the shapes
inductive Shape
  | Parallelogram
  | EquilateralTriangle
  | IsoscelesRightTriangle
  | RegularPentagon

-- Define a function to check if a shape coincides with itself after 180° rotation
def coincides_after_180_rotation (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | _ => False

-- Theorem statement
theorem only_parallelogram_coincides :
  ∀ (s : Shape), coincides_after_180_rotation s ↔ s = Shape.Parallelogram :=
by sorry

end only_parallelogram_coincides_l3197_319777


namespace perpendicular_line_through_point_l3197_319733

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -1/2x - 2 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -1/2 * x - 2
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = 1/2 * x - 3/2) →  -- Slope of L1 is 1/2
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (x₂ - x₁) * (-1/2) = -(y₂ - y₁) / (x₂ - x₁)) →  -- L1 and L2 are perpendicular
  L2 x y
  := by sorry

end perpendicular_line_through_point_l3197_319733


namespace average_of_a_and_b_l3197_319742

theorem average_of_a_and_b (a b c : ℝ) : 
  (b + c) / 2 = 50 → 
  c - a = 10 → 
  (a + b) / 2 = 45 := by
sorry

end average_of_a_and_b_l3197_319742


namespace abs_neg_three_eq_three_l3197_319730

theorem abs_neg_three_eq_three : abs (-3 : ℤ) = 3 := by sorry

end abs_neg_three_eq_three_l3197_319730


namespace ratio_problem_l3197_319791

theorem ratio_problem (x y : ℝ) (h : (2*x - 3*y) / (x + 2*y) = 5/4) :
  x / y = 22/3 := by sorry

end ratio_problem_l3197_319791


namespace fraction_of_trunks_l3197_319789

/-- Given that 38% of garments are bikinis and 63% are either bikinis or trunks,
    prove that 25% of garments are trunks. -/
theorem fraction_of_trunks
  (bikinis : Real)
  (bikinis_or_trunks : Real)
  (h1 : bikinis = 0.38)
  (h2 : bikinis_or_trunks = 0.63) :
  bikinis_or_trunks - bikinis = 0.25 := by
  sorry

end fraction_of_trunks_l3197_319789


namespace min_value_of_expression_l3197_319759

theorem min_value_of_expression (x y z : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10 ≥ -7/2 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 3 * x₀^2 + 3 * x₀ * y₀ + y₀^2 - 3 * x₀ + 3 * y₀ + z₀^2 + 6 * z₀ + 10 = -7/2 ∧
    x₀ = 3/2 ∧ y₀ = -3/2 ∧ z₀ = -3 :=
by sorry

end min_value_of_expression_l3197_319759


namespace square_nailing_theorem_l3197_319717

/-- Represents a paper square on the table -/
structure Square where
  color : Nat
  position : Real × Real

/-- Represents the arrangement of squares on the table -/
def Arrangement := List Square

/-- Checks if two squares can be nailed with one nail -/
def can_nail_together (s1 s2 : Square) : Prop := sorry

/-- The main theorem to be proved -/
theorem square_nailing_theorem (k : Nat) (arrangement : Arrangement) :
  (∀ (distinct_squares : List Square),
    distinct_squares.length = k →
    distinct_squares.Pairwise (λ s1 s2 => s1.color ≠ s2.color) →
    distinct_squares.Sublist arrangement →
    ∃ (s1 s2 : Square), s1 ∈ distinct_squares ∧ s2 ∈ distinct_squares ∧ can_nail_together s1 s2) →
  ∃ (color : Nat),
    let squares_of_color := arrangement.filter (λ s => s.color = color)
    ∃ (nails : List (Real × Real)), nails.length ≤ 2 * k - 2 ∧
      ∀ (s : Square), s ∈ squares_of_color →
        ∃ (nail : Real × Real), nail ∈ nails ∧ s.position = nail :=
sorry

end square_nailing_theorem_l3197_319717


namespace sum_of_digits_of_greatest_prime_divisor_l3197_319796

-- Define the number we're working with
def n : Nat := 32767

-- Define a function to get the greatest prime divisor
def greatestPrimeDivisor (m : Nat) : Nat :=
  sorry

-- Define a function to sum the digits of a number
def sumOfDigits (m : Nat) : Nat :=
  sorry

-- The theorem to prove
theorem sum_of_digits_of_greatest_prime_divisor :
  sumOfDigits (greatestPrimeDivisor n) = 13 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_l3197_319796


namespace imaginary_part_of_z_l3197_319740

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 2) : 
  z.im = -2 := by sorry

end imaginary_part_of_z_l3197_319740


namespace inscribed_circle_area_ratio_l3197_319798

theorem inscribed_circle_area_ratio (hexagon_side : Real) (hexagon_side_positive : hexagon_side > 0) :
  let hexagon_area := 3 * Real.sqrt 3 * hexagon_side^2 / 2
  let inscribed_circle_radius := hexagon_side * Real.sqrt 3 / 2
  let inscribed_circle_area := Real.pi * inscribed_circle_radius^2
  inscribed_circle_area / hexagon_area > 0.9 := by
  sorry

end inscribed_circle_area_ratio_l3197_319798


namespace quadratic_integer_roots_l3197_319745

theorem quadratic_integer_roots (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q →
  (∃ x y : ℤ, x^2 + 5*p*x + 7*q = 0 ∧ y^2 + 5*p*y + 7*q = 0) ↔ 
  ((p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) := by
sorry

end quadratic_integer_roots_l3197_319745


namespace cookie_recipe_ratio_l3197_319772

-- Define the total amount of sugar needed for the recipe
def total_sugar : ℚ := 3

-- Define the amount of sugar Katie still needs to add
def sugar_to_add : ℚ := 2.5

-- Define the amount of sugar Katie has already added
def sugar_already_added : ℚ := total_sugar - sugar_to_add

-- Define the ratio of sugar already added to total sugar needed
def sugar_ratio : ℚ × ℚ := (sugar_already_added, total_sugar)

-- Theorem to prove
theorem cookie_recipe_ratio :
  sugar_ratio = (1, 6) := by sorry

end cookie_recipe_ratio_l3197_319772


namespace distance_between_A_and_B_implies_a_eq_neg_one_l3197_319787

/-- Given two points A and B in 3D space, returns the square of the distance between them. -/
def distance_squared (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := A
  let (x₂, y₂, z₂) := B
  (x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2

theorem distance_between_A_and_B_implies_a_eq_neg_one :
  ∀ a : ℝ, 
  let A : ℝ × ℝ × ℝ := (-1, 1, -a)
  let B : ℝ × ℝ × ℝ := (-a, 3, -1)
  distance_squared A B = 4 → a = -1 := by
  sorry

end distance_between_A_and_B_implies_a_eq_neg_one_l3197_319787


namespace power_of_two_pairs_l3197_319737

theorem power_of_two_pairs (m n : ℕ+) :
  (∃ a : ℕ, m + n = 2^(a+1)) ∧
  (∃ b : ℕ, m * n + 1 = 2^b) →
  (∃ a : ℕ, (m = 2^(a+1) - 1 ∧ n = 1) ∨ (m = 2^a + 1 ∧ n = 2^a - 1) ∨
             (m = 1 ∧ n = 2^(a+1) - 1) ∨ (m = 2^a - 1 ∧ n = 2^a + 1)) :=
by sorry

end power_of_two_pairs_l3197_319737


namespace butter_profit_percentage_l3197_319701

/-- Calculates the profit percentage for a butter mixture sale --/
theorem butter_profit_percentage
  (butter1_weight : ℝ)
  (butter1_price : ℝ)
  (butter2_weight : ℝ)
  (butter2_price : ℝ)
  (selling_price : ℝ)
  (h1 : butter1_weight = 44)
  (h2 : butter1_price = 150)
  (h3 : butter2_weight = 36)
  (h4 : butter2_price = 125)
  (h5 : selling_price = 194.25) :
  let total_cost := butter1_weight * butter1_price + butter2_weight * butter2_price
  let total_weight := butter1_weight + butter2_weight
  let total_selling_price := total_weight * selling_price
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 40 := by
sorry

end butter_profit_percentage_l3197_319701


namespace three_doors_two_colors_l3197_319794

/-- The number of ways to paint a given number of doors with a given number of colors -/
def paintingWays (doors : ℕ) (colors : ℕ) : ℕ := colors ^ doors

/-- Theorem: The number of ways to paint 3 doors with 2 colors is 8 -/
theorem three_doors_two_colors : paintingWays 3 2 = 8 := by
  sorry

end three_doors_two_colors_l3197_319794


namespace h_one_value_l3197_319714

/-- A polynomial of degree 3 with constant coefficients -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ
  h_order : p < q ∧ q < r

/-- The function f(x) = x^3 + px^2 + qx + r -/
def f (c : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + c.p * x^2 + c.q * x + c.r

/-- A polynomial h(x) whose roots are the squares of the reciprocals of the roots of f(x) -/
def h (c : CubicPolynomial) (x : ℝ) : ℝ :=
  sorry  -- Definition of h(x) is not explicitly given in the problem

/-- Theorem stating the value of h(1) in terms of p, q, and r -/
theorem h_one_value (c : CubicPolynomial) :
  h c 1 = (1 - c.p + c.q - c.r) * (1 - c.q + c.p - c.r) * (1 - c.r + c.p - c.q) / c.r^2 :=
sorry

end h_one_value_l3197_319714


namespace one_less_than_three_times_l3197_319760

/-- The number that is 1 less than 3 times a real number a can be expressed as 3a - 1. -/
theorem one_less_than_three_times (a : ℝ) : ∃ x : ℝ, x = 3 * a - 1 ∧ x + 1 = 3 * a := by
  sorry

end one_less_than_three_times_l3197_319760


namespace emily_seeds_count_l3197_319783

/-- The number of seeds Emily planted in the big garden -/
def big_garden_seeds : ℕ := 29

/-- The number of small gardens Emily has -/
def num_small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden -/
def seeds_per_small_garden : ℕ := 4

/-- The total number of seeds Emily planted -/
def total_seeds : ℕ := big_garden_seeds + num_small_gardens * seeds_per_small_garden

theorem emily_seeds_count : total_seeds = 41 := by
  sorry

end emily_seeds_count_l3197_319783


namespace production_scale_l3197_319769

/-- Production function that calculates the number of items produced given the number of workers, hours per day, number of days, and production rate. -/
def production (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours_per_day : ℚ) * (days : ℚ) * rate

/-- Theorem stating that if 8 workers produce 512 items in 8 hours a day for 8 days, 
    then 10 workers working 10 hours a day for 10 days will produce 1000 items, 
    assuming a constant production rate. -/
theorem production_scale (rate : ℚ) : 
  production 8 8 8 rate = 512 → production 10 10 10 rate = 1000 := by
  sorry

#check production_scale

end production_scale_l3197_319769


namespace count_prime_differences_l3197_319706

def is_in_set (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k - 3 ∧ k ≥ 1

def is_prime_difference (n : ℕ) : Prop := ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p - q

theorem count_prime_differences : 
  (∃! (s : Finset ℕ), (∀ n ∈ s, is_in_set n ∧ is_prime_difference n) ∧ s.card = 2) :=
sorry

end count_prime_differences_l3197_319706


namespace sqrt_product_equals_two_l3197_319724

theorem sqrt_product_equals_two : Real.sqrt (2/3) * Real.sqrt 6 = 2 := by
  sorry

end sqrt_product_equals_two_l3197_319724


namespace total_team_score_l3197_319720

def team_score (connor_score amy_score jason_score emily_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score + emily_score

theorem total_team_score :
  ∀ (connor_score amy_score jason_score emily_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    emily_score = 3 * (connor_score + amy_score + jason_score) →
    team_score connor_score amy_score jason_score emily_score = 80 :=
by
  sorry

end total_team_score_l3197_319720


namespace min_four_dollar_frisbees_l3197_319753

theorem min_four_dollar_frisbees 
  (total_frisbees : ℕ) 
  (total_receipts : ℕ) 
  (h_total : total_frisbees = 60) 
  (h_receipts : total_receipts = 204) : 
  ∃ (three_dollar : ℕ) (four_dollar : ℕ), 
    three_dollar + four_dollar = total_frisbees ∧ 
    3 * three_dollar + 4 * four_dollar = total_receipts ∧ 
    four_dollar ≥ 24 := by
  sorry

end min_four_dollar_frisbees_l3197_319753


namespace pens_after_sale_l3197_319758

theorem pens_after_sale (initial_pens : ℕ) (sold_pens : ℕ) (h1 : initial_pens = 106) (h2 : sold_pens = 92) :
  initial_pens - sold_pens = 14 := by
  sorry

end pens_after_sale_l3197_319758


namespace inequality_solution_set_l3197_319766

theorem inequality_solution_set (x : ℝ) : x + 2 > 3 ↔ x > 1 := by sorry

end inequality_solution_set_l3197_319766


namespace derivative_at_one_l3197_319749

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x + x^3) :
  deriv f 1 = 5 := by
  sorry

end derivative_at_one_l3197_319749


namespace tax_discount_order_invariance_l3197_319734

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_order_invariance 
  (original_price tax_rate discount_rate : ℝ) 
  (hp : 0 < original_price) 
  (ht : 0 ≤ tax_rate) 
  (hd : 0 ≤ discount_rate) 
  (hd1 : discount_rate ≤ 1) :
  original_price * (1 + tax_rate) * (1 - discount_rate) = 
  original_price * (1 - discount_rate) * (1 + tax_rate) :=
sorry

end tax_discount_order_invariance_l3197_319734


namespace largest_multiple_12_negation_gt_neg150_l3197_319771

theorem largest_multiple_12_negation_gt_neg150 :
  ∀ n : ℤ, (12 ∣ n) → -n > -150 → n ≤ 144 :=
by sorry

end largest_multiple_12_negation_gt_neg150_l3197_319771


namespace seven_times_prime_divisors_l3197_319735

theorem seven_times_prime_divisors (p : ℕ) (h_prime : Nat.Prime p) :
  (Nat.divisors (7 * p)).card = 4 := by
  sorry

end seven_times_prime_divisors_l3197_319735


namespace circle_tangent_to_x_axis_l3197_319748

/-- A circle with center (-1, 2) that is tangent to the x-axis has the equation (x + 1)^2 + (y - 2)^2 = 4 -/
theorem circle_tangent_to_x_axis :
  ∃ (r : ℝ),
    (∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 ↔ ((x + 1)^2 + (y - 2)^2 = r^2)) ∧
    (∀ (x : ℝ), ∃ (y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 → y = 0) :=
by sorry

end circle_tangent_to_x_axis_l3197_319748


namespace power_multiplication_l3197_319778

theorem power_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end power_multiplication_l3197_319778


namespace optimus_prime_distance_l3197_319795

/-- Prove that the distance between points A and B is 750 km given the conditions in the problem --/
theorem optimus_prime_distance : ∀ (D S : ℝ),
  (D / S - D / (S * (1 + 1/4)) = 1) →
  (150 / S + (D - 150) / S - (150 / S + (D - 150) / (S * (1 + 1/5))) = 2/3) →
  D = 750 := by
  sorry

end optimus_prime_distance_l3197_319795


namespace cube_planes_divide_space_into_27_parts_l3197_319775

/-- Represents a cube in 3D space -/
structure Cube where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Function to generate planes through each face of a cube -/
def planes_through_cube_faces (c : Cube) : List Plane :=
  sorry

/-- Function to count the number of parts the space is divided into by the planes -/
def count_divided_parts (planes : List Plane) : Nat :=
  sorry

/-- Theorem stating that planes through each face of a cube divide space into 27 parts -/
theorem cube_planes_divide_space_into_27_parts (c : Cube) :
  count_divided_parts (planes_through_cube_faces c) = 27 := by
  sorry

end cube_planes_divide_space_into_27_parts_l3197_319775


namespace fraction_chain_l3197_319708

theorem fraction_chain (x y z w : ℚ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 4)
  (h3 : z / w = 7) :
  w / x = 4 / 35 := by
  sorry

end fraction_chain_l3197_319708


namespace arithmetic_mean_implies_arithmetic_progression_geometric_mean_implies_geometric_progression_l3197_319752

/-- A sequence is an arithmetic progression if the difference between consecutive terms is constant. -/
def IsArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is a geometric progression if the ratio between consecutive terms is constant. -/
def IsGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

/-- Theorem: If each term (from the second to the second-to-last) in a sequence is the arithmetic mean
    of its neighboring terms, then the sequence is an arithmetic progression. -/
theorem arithmetic_mean_implies_arithmetic_progression (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3)
    (h_arithmetic_mean : ∀ k, 2 ≤ k ∧ k < n → a k = (a (k - 1) + a (k + 1)) / 2) :
    IsArithmeticProgression a := by sorry

/-- Theorem: If each term (from the second to the second-to-last) in a sequence is the geometric mean
    of its neighboring terms, then the sequence is a geometric progression. -/
theorem geometric_mean_implies_geometric_progression (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3)
    (h_geometric_mean : ∀ k, 2 ≤ k ∧ k < n → a k = Real.sqrt (a (k - 1) * a (k + 1))) :
    IsGeometricProgression a := by sorry

end arithmetic_mean_implies_arithmetic_progression_geometric_mean_implies_geometric_progression_l3197_319752


namespace angle_between_vectors_l3197_319755

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (-1, 3) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = π / 4 := by
  sorry

end angle_between_vectors_l3197_319755


namespace polynomial_simplification_simplify_and_evaluate_l3197_319736

-- Part 1: Polynomial simplification
theorem polynomial_simplification (m n : ℝ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m := by sorry

-- Part 2: Simplify and evaluate
theorem simplify_and_evaluate :
  let a : ℝ := 1/2
  let b : ℝ := 3
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -53/2 := by sorry

end polynomial_simplification_simplify_and_evaluate_l3197_319736


namespace max_removable_edges_l3197_319722

/-- Represents a volleyball net grid with internal divisions -/
structure VolleyballNet where
  rows : Nat
  cols : Nat
  internalDivisions : Nat

/-- Calculates the total number of nodes in the volleyball net -/
def totalNodes (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1) + net.rows * net.cols

/-- Calculates the total number of edges in the volleyball net -/
def totalEdges (net : VolleyballNet) : Nat :=
  net.rows * (net.cols + 1) + net.cols * (net.rows + 1) + net.internalDivisions * net.rows * net.cols

/-- Theorem stating the maximum number of removable edges -/
theorem max_removable_edges (net : VolleyballNet) :
  net.rows = 10 → net.cols = 20 → net.internalDivisions = 4 →
  totalEdges net - (totalNodes net - 1) = 800 := by
  sorry


end max_removable_edges_l3197_319722


namespace max_distance_between_C1_and_C2_l3197_319780

-- Define the curves C1 and C2
def C1 (ρ θ : ℝ) : Prop := ρ + 6 * Real.sin θ + 8 / ρ = 0
def C2 (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define a point on C1
def point_on_C1 (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), C1 ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define a point on C2
def point_on_C2 (x y : ℝ) : Prop := C2 x y

-- State the theorem
theorem max_distance_between_C1_and_C2 :
  ∃ (max_dist : ℝ),
    max_dist = Real.sqrt 65 / 2 + 1 ∧
    (∀ (x1 y1 x2 y2 : ℝ),
      point_on_C1 x1 y1 → point_on_C2 x2 y2 →
      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≤ max_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      point_on_C1 x1 y1 ∧ point_on_C2 x2 y2 ∧
      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = max_dist) :=
by sorry

end max_distance_between_C1_and_C2_l3197_319780


namespace bananas_left_l3197_319716

/-- The number of bananas originally in the jar -/
def original_bananas : ℕ := 46

/-- The number of bananas Denise removes from the jar -/
def removed_bananas : ℕ := 5

/-- Theorem stating the number of bananas left in the jar after Denise removes some -/
theorem bananas_left : original_bananas - removed_bananas = 41 := by
  sorry

end bananas_left_l3197_319716


namespace program_output_correct_verify_output_l3197_319713

/-- Represents the result of the program execution -/
structure ProgramResult where
  x : Int
  y : Int

/-- Executes the program logic based on initial values -/
def executeProgram (initialX initialY : Int) : ProgramResult :=
  if initialX < 0 then
    { x := initialY - 4, y := initialY }
  else
    { x := initialX, y := initialY + 4 }

/-- Theorem stating the program output for given initial values -/
theorem program_output_correct :
  let result := executeProgram 2 (-30)
  result.x - result.y = 28 ∧ result.y - result.x = -28 := by
  sorry

/-- Verifies that the program output matches the expected result -/
theorem verify_output :
  let result := executeProgram 2 (-30)
  (result.x - result.y, result.y - result.x) = (28, -28) := by
  sorry

end program_output_correct_verify_output_l3197_319713


namespace julia_monday_playmates_l3197_319728

/-- The number of kids Julia played with on different days -/
structure JuliaPlaymates where
  wednesday : ℕ
  monday : ℕ

/-- Given conditions about Julia's playmates -/
def julia_conditions (j : JuliaPlaymates) : Prop :=
  j.wednesday = 4 ∧ j.monday = j.wednesday + 2

/-- Theorem: Julia played with 6 kids on Monday -/
theorem julia_monday_playmates (j : JuliaPlaymates) (h : julia_conditions j) : j.monday = 6 := by
  sorry

end julia_monday_playmates_l3197_319728
