import Mathlib

namespace log_inequality_equiv_solution_set_l1191_119158

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the set of solutions
def solution_set : Set ℝ := {x | x < -1 ∨ x > 3}

-- State the theorem
theorem log_inequality_equiv_solution_set :
  ∀ x : ℝ, lg (x^2 - 2*x - 3) ≥ 0 ↔ x ∈ solution_set :=
sorry

end log_inequality_equiv_solution_set_l1191_119158


namespace parallel_transitive_perpendicular_from_line_l1191_119182

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Axioms for parallel and perpendicular relations
axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_symm {a b : Plane} : perpendicular a b → perpendicular b a

-- Theorem 1
theorem parallel_transitive {α β γ : Plane} :
  parallel α β → parallel α γ → parallel β γ := by sorry

-- Theorem 2
theorem perpendicular_from_line {m : Line} {α β : Plane} :
  line_perpendicular m α → line_parallel m β → perpendicular α β := by sorry

end parallel_transitive_perpendicular_from_line_l1191_119182


namespace billion_to_scientific_notation_l1191_119140

theorem billion_to_scientific_notation :
  ∀ (x : ℝ), x = 508 → (x * (10^9 : ℝ)) = 5.08 * (10^11 : ℝ) := by
  sorry

end billion_to_scientific_notation_l1191_119140


namespace discretionary_income_ratio_l1191_119148

/-- Represents Jill's financial situation --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFundPercentage : ℝ
  savingsPercentage : ℝ
  socializingPercentage : ℝ
  remainingAmount : ℝ

/-- Theorem stating the ratio of discretionary income to net salary --/
theorem discretionary_income_ratio (j : JillFinances) 
  (h1 : j.netSalary = 3700)
  (h2 : j.vacationFundPercentage = 0.3)
  (h3 : j.savingsPercentage = 0.2)
  (h4 : j.socializingPercentage = 0.35)
  (h5 : j.remainingAmount = 111)
  (h6 : j.discretionaryIncome * (1 - (j.vacationFundPercentage + j.savingsPercentage + j.socializingPercentage)) = j.remainingAmount) :
  j.discretionaryIncome / j.netSalary = 1 / 5 := by
  sorry

end discretionary_income_ratio_l1191_119148


namespace cube_edge_length_l1191_119161

theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) :
  surface_area = 96 ∧ surface_area = 6 * edge_length^2 → edge_length = 4 := by
  sorry

end cube_edge_length_l1191_119161


namespace smallest_value_l1191_119169

theorem smallest_value (a b : ℝ) (h : b < 0) : (a + b < a) ∧ (a + b < a - b) := by
  sorry

end smallest_value_l1191_119169


namespace union_of_A_and_B_l1191_119136

def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 4} := by sorry

end union_of_A_and_B_l1191_119136


namespace unique_solution_quadratic_product_l1191_119110

theorem unique_solution_quadratic_product (k : ℝ) : 
  (∃! x : ℝ, k * x^2 + (k + 5) * x + 5 = 0) → k = 5 :=
by sorry

end unique_solution_quadratic_product_l1191_119110


namespace solution_to_equation_sum_of_fourth_powers_l1191_119198

-- Define the equation for part 1
def equation (x : ℝ) : Prop := x^4 - x^2 - 6 = 0

-- Theorem for part 1
theorem solution_to_equation :
  ∀ x : ℝ, equation x ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 :=
sorry

-- Define the conditions for part 2
def condition (a b : ℝ) : Prop :=
  a^4 - 3*a^2 + 1 = 0 ∧ b^4 - 3*b^2 + 1 = 0 ∧ a ≠ b

-- Theorem for part 2
theorem sum_of_fourth_powers (a b : ℝ) :
  condition a b → a^4 + b^4 = 7 :=
sorry

end solution_to_equation_sum_of_fourth_powers_l1191_119198


namespace square_difference_40_39_l1191_119168

theorem square_difference_40_39 : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by
  sorry

end square_difference_40_39_l1191_119168


namespace shirts_produced_l1191_119111

theorem shirts_produced (shirts_per_minute : ℕ) (minutes_worked : ℕ) : 
  shirts_per_minute = 2 → minutes_worked = 4 → shirts_per_minute * minutes_worked = 8 := by
  sorry

end shirts_produced_l1191_119111


namespace angle_set_inclusion_l1191_119152

def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

theorem angle_set_inclusion : N ⊆ M ∧ M ⊆ P := by sorry

end angle_set_inclusion_l1191_119152


namespace find_C_l1191_119108

theorem find_C (A B C : ℕ) : A = 348 → B = A + 173 → C = B + 299 → C = 820 := by
  sorry

end find_C_l1191_119108


namespace tim_laundry_cycle_l1191_119135

/-- Ronald's laundry cycle in days -/
def ronald_cycle : ℕ := 6

/-- Number of days until they both do laundry on the same day again -/
def next_common_day : ℕ := 18

/-- Tim's laundry cycle in days -/
def tim_cycle : ℕ := 3

theorem tim_laundry_cycle :
  (ronald_cycle ∣ next_common_day) ∧
  (tim_cycle ∣ next_common_day) ∧
  (tim_cycle < ronald_cycle) ∧
  (∀ x : ℕ, x < tim_cycle → ¬(x ∣ next_common_day ∧ x ∣ ronald_cycle)) :=
sorry

end tim_laundry_cycle_l1191_119135


namespace quadratic_two_distinct_roots_l1191_119123

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end quadratic_two_distinct_roots_l1191_119123


namespace log_sum_property_l1191_119146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_sum_property (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∃ (f : ℝ → ℝ) (f_inv : ℝ → ℝ),
    (∀ x > 0, f x = Real.log x / Real.log a) ∧
    (∀ x, f (f_inv x) = x) ∧
    (f_inv 2 = 9) →
    f 9 + f 6 = 1 := by
  sorry

end log_sum_property_l1191_119146


namespace library_books_pages_l1191_119109

theorem library_books_pages (num_books : ℕ) (total_pages : ℕ) (h1 : num_books = 8) (h2 : total_pages = 3824) :
  total_pages / num_books = 478 := by
sorry

end library_books_pages_l1191_119109


namespace cereal_box_ratio_l1191_119191

/-- Theorem: Cereal Box Ratio
Given 3 boxes of cereal where:
- The first box contains 14 ounces
- The total amount in all boxes is 33 ounces
- The second box contains 5 ounces less than the third box
Then the ratio of cereal in the second box to the first box is 1:2
-/
theorem cereal_box_ratio (box1 box2 box3 : ℝ) : 
  box1 = 14 →
  box1 + box2 + box3 = 33 →
  box2 = box3 - 5 →
  box2 / box1 = 1 / 2 := by
sorry

end cereal_box_ratio_l1191_119191


namespace parabola_focus_coordinates_l1191_119196

/-- Given a parabola with equation y² = 4ax where a < 0, 
    prove that the coordinates of its focus are (a, 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a < 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*a*x}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (a, 0) := by
  sorry

end parabola_focus_coordinates_l1191_119196


namespace marble_distribution_l1191_119189

theorem marble_distribution (sets : Nat) (marbles_per_set : Nat) (marbles_per_student : Nat) :
  sets = 3 →
  marbles_per_set = 32 →
  marbles_per_student = 4 →
  (sets * marbles_per_set) % marbles_per_student = 0 →
  (sets * marbles_per_set) / marbles_per_student = 24 := by
  sorry

end marble_distribution_l1191_119189


namespace problem_solution_l1191_119144

structure Problem where
  -- Define the parabola
  p : ℝ
  parabola : ℝ → ℝ → Prop
  parabola_def : ∀ x y, parabola x y ↔ y^2 = 2*p*x

  -- Define points O, P, and Q
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ

  -- Define line l
  l : ℝ → ℝ → Prop

  -- Conditions
  O_is_origin : O = (0, 0)
  P_coordinates : P = (2, 1)
  p_positive : p > 0
  l_perpendicular_to_OP : (l 2 1) ∧ (∀ x y, l x y → (x - 2) = (y - 1) * (2 / 1))
  Q_on_l : l Q.1 Q.2
  Q_on_parabola : parabola Q.1 Q.2
  OPQ_right_isosceles : (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = (P.1 - O.1)^2 + (P.2 - O.2)^2

theorem problem_solution (prob : Problem) : prob.p = 2 := by
  sorry

end problem_solution_l1191_119144


namespace matrix_rank_two_l1191_119147

/-- Given an n×n matrix A where A_ij = i + j, prove that the rank of A is 2 -/
theorem matrix_rank_two (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
  (h : ∀ (i j : Fin n), A i j = (i.val + 1 : ℝ) + (j.val + 1 : ℝ)) :
  Matrix.rank A = 2 := by
  sorry

end matrix_rank_two_l1191_119147


namespace correct_quotient_proof_l1191_119166

theorem correct_quotient_proof (dividend : ℕ) (wrong_divisor correct_divisor wrong_quotient : ℕ) 
  (h1 : dividend = wrong_divisor * wrong_quotient)
  (h2 : wrong_divisor = 121)
  (h3 : correct_divisor = 215)
  (h4 : wrong_quotient = 432) :
  dividend / correct_divisor = 243 := by
sorry

end correct_quotient_proof_l1191_119166


namespace trapezoid_diagonal_length_l1191_119162

/-- Represents a trapezoid ABCD with specific side lengths and angle -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  cos_BCD : ℝ
  h_AB : AB = 27
  h_CD : CD = 28
  h_BC : BC = 5
  h_cos_BCD : cos_BCD = -2/7

/-- The length of the diagonal AC in the trapezoid -/
def diagonal_AC (t : Trapezoid) : Set ℝ :=
  {28, 2 * Real.sqrt 181}

/-- Theorem stating that the diagonal AC of the trapezoid is either 28 or 2√181 -/
theorem trapezoid_diagonal_length (t : Trapezoid) :
  ∃ x ∈ diagonal_AC t, x = (Real.sqrt ((t.AB - t.BC)^2 + (t.CD * Real.sqrt (1 - t.cos_BCD^2))^2)) :=
sorry

end trapezoid_diagonal_length_l1191_119162


namespace bryans_books_l1191_119104

theorem bryans_books (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 9) 
  (h2 : books_per_shelf = 56) : 
  num_shelves * books_per_shelf = 504 := by
  sorry

end bryans_books_l1191_119104


namespace min_number_after_operations_l1191_119197

def board_operation (S : Finset ℕ) : Finset ℕ :=
  sorry

def min_after_operations (n : ℕ) : ℕ :=
  sorry

theorem min_number_after_operations :
  (min_after_operations 111 = 0) ∧ (min_after_operations 110 = 1) :=
sorry

end min_number_after_operations_l1191_119197


namespace intersection_M_N_l1191_119188

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | -5 < x ∧ x < 5 }

theorem intersection_M_N : M ∩ N = { x | -3 < x ∧ x < 5 } := by sorry

end intersection_M_N_l1191_119188


namespace arithmetic_evaluation_l1191_119163

theorem arithmetic_evaluation : 2 * (5 - 2) - 5^2 = -19 := by
  sorry

end arithmetic_evaluation_l1191_119163


namespace even_periodic_function_value_l1191_119183

/-- A function that is even and has a period of 2 -/
def EvenPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ (∀ x, f (x + 2) = f x)

theorem even_periodic_function_value 
  (f : ℝ → ℝ) 
  (h_even_periodic : EvenPeriodicFunction f)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = x + 1) :
  ∀ x ∈ Set.Ioo 1 2, f x = 3 - x := by
sorry

end even_periodic_function_value_l1191_119183


namespace symmetric_point_wrt_x_axis_l1191_119187

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The x-axis symmetry operation -/
def xAxisSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetric_point_wrt_x_axis :
  let original := Point3D.mk (-2) 1 9
  xAxisSymmetry original = Point3D.mk (-2) (-1) (-9) := by
  sorry

end symmetric_point_wrt_x_axis_l1191_119187


namespace hyperbola_asymptotes_l1191_119125

/-- The asymptotes of the hyperbola x²/16 - y²/9 = -1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/16 - y^2/9 = -1
  ∀ x y : ℝ, (∃ (ε : ℝ), ε > 0 ∧ ∀ δ : ℝ, δ > ε → h (δ * x) (δ * y)) →
    y = (3/4) * x ∨ y = -(3/4) * x :=
by sorry

end hyperbola_asymptotes_l1191_119125


namespace book_cost_in_cny_l1191_119100

/-- Exchange rate from US dollar to Namibian dollar -/
def usd_to_nad : ℚ := 7

/-- Exchange rate from US dollar to Chinese yuan -/
def usd_to_cny : ℚ := 6

/-- Cost of the book in Namibian dollars -/
def book_cost_nad : ℚ := 168

/-- Calculate the cost of the book in Chinese yuan -/
def book_cost_cny : ℚ := book_cost_nad * (usd_to_cny / usd_to_nad)

/-- Theorem stating that the book costs 144 Chinese yuan -/
theorem book_cost_in_cny : book_cost_cny = 144 := by
  sorry

end book_cost_in_cny_l1191_119100


namespace taxi_fare_distance_l1191_119167

/-- Represents the taxi fare structure and proves the distance for each fare segment -/
theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (total_distance : ℝ) (total_fare : ℝ) :
  initial_fare = 8 →
  subsequent_fare = 0.8 →
  total_distance = 8 →
  total_fare = 39.2 →
  ∃ (d : ℝ), d > 0 ∧ d = 1/5 ∧
    total_fare = initial_fare + subsequent_fare * ((total_distance - d) / d) :=
by
  sorry


end taxi_fare_distance_l1191_119167


namespace quadratic_root_existence_l1191_119164

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_root_existence (a b c : ℝ) (ha : a ≠ 0) :
  quadratic_function a b c (-3) = -11 →
  quadratic_function a b c (-2) = -5 →
  quadratic_function a b c (-1) = -1 →
  quadratic_function a b c 0 = 1 →
  quadratic_function a b c 1 = 1 →
  ∃ x₁ : ℝ, quadratic_function a b c x₁ = 0 ∧ -1 < x₁ ∧ x₁ < 0 :=
by sorry

end quadratic_root_existence_l1191_119164


namespace prism_coloring_iff_divisible_by_three_l1191_119113

/-- Represents a prism with an n-gon base -/
structure Prism :=
  (n : ℕ)

/-- Represents a coloring of the prism vertices -/
def Coloring (p : Prism) := Fin (2 * p.n) → Fin 3

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (p : Prism) (c : Coloring p) : Prop :=
  ∀ v : Fin (2 * p.n), 
    ∃ (c1 c2 : Fin 3), c1 ≠ c v ∧ c2 ≠ c v ∧ c1 ≠ c2 ∧
    ∃ (v1 v2 : Fin (2 * p.n)), v1 ≠ v ∧ v2 ≠ v ∧ v1 ≠ v2 ∧
    c v1 = c1 ∧ c v2 = c2

theorem prism_coloring_iff_divisible_by_three (p : Prism) :
  (∃ c : Coloring p, is_valid_coloring p c) ↔ p.n % 3 = 0 :=
sorry

end prism_coloring_iff_divisible_by_three_l1191_119113


namespace parabola_focus_distance_l1191_119101

/-- Proves that for a parabola y^2 = 2px with p > 0, if a point P(2,m) on the parabola
    is at a distance of 4 from its focus, then p = 4. -/
theorem parabola_focus_distance (p : ℝ) (m : ℝ) (h1 : p > 0) :
  m^2 = 2*p*2 →  -- Point P(2,m) is on the parabola y^2 = 2px
  (2 - p/2)^2 + m^2 = 4^2 →  -- Distance from P to focus F(p/2, 0) is 4
  p = 4 := by
sorry

end parabola_focus_distance_l1191_119101


namespace tangent_line_at_one_l1191_119119

def f (x : ℝ) := x^3 - x + 3

theorem tangent_line_at_one : 
  ∃ (a b : ℝ), ∀ (x y : ℝ), 
    (y = f x ∧ x = 1) → (y = a * x + b) :=
sorry

end tangent_line_at_one_l1191_119119


namespace population_increase_l1191_119174

theorem population_increase (birth_rate : ℚ) (death_rate : ℚ) (seconds_per_day : ℕ) :
  birth_rate = 7 / 2 →
  death_rate = 3 / 2 →
  seconds_per_day = 24 * 3600 →
  (birth_rate - death_rate) * seconds_per_day = 172800 := by
  sorry

end population_increase_l1191_119174


namespace line_intersection_theorem_l1191_119107

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (skew : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- State the theorem
theorem line_intersection_theorem 
  (m n l : Line) (α β : Plane) 
  (h1 : skew m n)
  (h2 : contains α m)
  (h3 : contains β n)
  (h4 : plane_intersection α β = l) :
  intersects l m ∨ intersects l n :=
sorry

end line_intersection_theorem_l1191_119107


namespace average_speed_is_25_l1191_119199

-- Define the given conditions
def workdays : ℕ := 5
def work_distance : ℝ := 20
def weekend_ride : ℝ := 200
def total_time : ℝ := 16

-- Define the total distance
def total_distance : ℝ := 2 * workdays * work_distance + weekend_ride

-- Theorem to prove
theorem average_speed_is_25 : 
  total_distance / total_time = 25 := by sorry

end average_speed_is_25_l1191_119199


namespace star_example_l1191_119194

-- Define the ★ operation
def star (m n p q : ℚ) : ℚ := (m + 1) * (p + 1) * ((q + 1) / (n + 1))

-- Theorem statement
theorem star_example : star (5/11) (11/1) (7/2) (2/1) = 12 := by
  sorry

end star_example_l1191_119194


namespace schedule_theorem_l1191_119129

-- Define the number of periods in a day
def periods : ℕ := 7

-- Define the number of courses to be scheduled
def courses : ℕ := 4

-- Define a function to calculate the number of ways to schedule courses
def schedule_ways (p : ℕ) (c : ℕ) : ℕ := sorry

-- Theorem statement
theorem schedule_theorem : 
  schedule_ways periods courses = 120 := by sorry

end schedule_theorem_l1191_119129


namespace reciprocal_inequality_for_negative_numbers_l1191_119121

theorem reciprocal_inequality_for_negative_numbers (a b : ℝ) 
  (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end reciprocal_inequality_for_negative_numbers_l1191_119121


namespace pumpkin_ravioli_weight_l1191_119143

theorem pumpkin_ravioli_weight (brother_ravioli_count : ℕ) (total_weight : ℝ) : 
  brother_ravioli_count = 12 → total_weight = 15 → 
  (total_weight / brother_ravioli_count : ℝ) = 1.25 := by
  sorry

end pumpkin_ravioli_weight_l1191_119143


namespace cos_250_over_sin_200_equals_1_l1191_119172

theorem cos_250_over_sin_200_equals_1 :
  (Real.cos (250 * π / 180)) / (Real.sin (200 * π / 180)) = 1 := by
  sorry

end cos_250_over_sin_200_equals_1_l1191_119172


namespace y_derivative_l1191_119176

noncomputable section

open Real

def y (x : ℝ) : ℝ := (1/6) * log ((1 - sinh (2*x)) / (2 + sinh (2*x)))

theorem y_derivative (x : ℝ) : 
  deriv y x = cosh (2*x) / (sinh (2*x)^2 + sinh (2*x) - 2) :=
by sorry

end y_derivative_l1191_119176


namespace h_properties_l1191_119138

-- Define the functions
noncomputable def g (x : ℝ) : ℝ := 2^x

-- f is symmetric to g with respect to y = x
def f_symmetric_to_g (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define h in terms of f
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - |x|)

-- Main theorem
theorem h_properties (f : ℝ → ℝ) (hf : f_symmetric_to_g f) :
  (∀ x, h f x = h f (-x)) ∧ 
  (∀ x, h f x ≤ 0 ∧ h f 0 = 0) :=
sorry

end h_properties_l1191_119138


namespace circumcircle_radius_l1191_119151

/-- Given a triangle ABC with side length a = 2 and sin A = 1/3, 
    the radius R of its circumcircle is 3. -/
theorem circumcircle_radius (A B C : ℝ × ℝ) (a : ℝ) (sin_A : ℝ) :
  a = 2 →
  sin_A = 1/3 →
  let R := (a / 2) / sin_A
  R = 3 := by sorry

end circumcircle_radius_l1191_119151


namespace solution_l1191_119134

-- Define the set of points satisfying the equation
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + p.2)^2 = p.1^2 + p.2^2}

-- Define the x-axis and y-axis
def X_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def Y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Theorem stating that S is equivalent to the union of X_axis and Y_axis
theorem solution : S = X_axis ∪ Y_axis := by
  sorry

end solution_l1191_119134


namespace woodworker_legs_count_l1191_119195

/-- The number of furniture legs made by a woodworker -/
def total_furniture_legs (chairs tables : ℕ) : ℕ :=
  4 * chairs + 4 * tables

/-- Theorem: A woodworker who has built 6 chairs and 4 tables has made 40 furniture legs in total -/
theorem woodworker_legs_count : total_furniture_legs 6 4 = 40 := by
  sorry

end woodworker_legs_count_l1191_119195


namespace cycling_speed_problem_l1191_119130

/-- Proves that given the conditions of the problem, B's cycling speed is 20 kmph -/
theorem cycling_speed_problem (a_speed b_speed : ℝ) (delay meeting_distance : ℝ) : 
  a_speed = 10 →
  delay = 7 →
  meeting_distance = 140 →
  b_speed * delay = meeting_distance →
  b_speed = 20 := by
  sorry

end cycling_speed_problem_l1191_119130


namespace area_after_shortening_other_side_l1191_119120

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side by 2 -/
def shortened : Rectangle := { length := 3, width := 7 }

/-- The rectangle after shortening the other side by 2 -/
def other_shortened : Rectangle := { length := 5, width := 5 }

theorem area_after_shortening_other_side :
  shortened.area = 21 → other_shortened.area = 25 := by sorry

end area_after_shortening_other_side_l1191_119120


namespace perfect_square_sum_of_powers_l1191_119114

theorem perfect_square_sum_of_powers (x y z : ℕ+) :
  ∃ (k : ℕ), (4:ℕ)^(x:ℕ) + (4:ℕ)^(y:ℕ) + (4:ℕ)^(z:ℕ) = k^2 ↔
  ∃ (b z' : ℕ+), x = 2*b - 1 + z' ∧ y = b + z' ∧ z = z' :=
sorry

end perfect_square_sum_of_powers_l1191_119114


namespace walking_speed_difference_l1191_119149

theorem walking_speed_difference (child_distance child_time elderly_distance elderly_time : ℝ) :
  child_distance = 15 ∧ 
  child_time = 3.5 ∧ 
  elderly_distance = 10 ∧ 
  elderly_time = 4 → 
  (elderly_time * 60 / elderly_distance) - (child_time * 60 / child_distance) = 10 :=
by sorry

end walking_speed_difference_l1191_119149


namespace least_five_digit_square_and_cube_l1191_119193

theorem least_five_digit_square_and_cube : 
  (∀ n : ℕ, n < 15625 → ¬(∃ a b : ℕ, n = a^2 ∧ n = b^3 ∧ n ≥ 10000)) ∧ 
  (∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3) := by
  sorry

end least_five_digit_square_and_cube_l1191_119193


namespace carts_needed_is_15_l1191_119157

-- Define the total volume of goods
def total_volume : ℚ := 1

-- Define the daily capacity of each vehicle type
def large_truck_capacity : ℚ := total_volume / (3 * 4)
def small_truck_capacity : ℚ := total_volume / (4 * 5)
def cart_capacity : ℚ := total_volume / (20 * 6)

-- Define the work done in the first 2 days
def work_done_2_days : ℚ := 2 * (2 * large_truck_capacity + 3 * small_truck_capacity + 7 * cart_capacity)

-- Define the remaining work
def remaining_work : ℚ := total_volume - work_done_2_days

-- Define the number of carts needed for the last 2 days
def carts_needed : ℕ := (remaining_work / (2 * cart_capacity)).ceil.toNat

-- Theorem statement
theorem carts_needed_is_15 : carts_needed = 15 := by
  sorry

end carts_needed_is_15_l1191_119157


namespace min_sum_squared_distances_l1191_119192

/-- Given five collinear points A, B, C, D, E in that order, with specific distances between them,
    prove that the minimum sum of squared distances from these points to any point P on AD is 237. -/
theorem min_sum_squared_distances (A B C D E P : ℝ) : 
  (A < B) → (B < C) → (C < D) → (D < E) →  -- Points are collinear and in order
  (B - A = 1) → (C - B = 1) → (D - C = 3) → (E - D = 12) →  -- Given distances
  (A ≤ P) → (P ≤ D) →  -- P is on segment AD
  ∃ (m : ℝ), ∀ (Q : ℝ), (A ≤ Q) → (Q ≤ D) → 
    (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 ≥ m ∧ 
    m = 237 :=
by sorry

end min_sum_squared_distances_l1191_119192


namespace parabola_y_order_l1191_119156

/-- Given that (-3, y₁), (1, y₂), and (-1/2, y₃) are points on the graph of y = x² - 2x + 3,
    prove that y₂ < y₃ < y₁ -/
theorem parabola_y_order (y₁ y₂ y₃ : ℝ) 
    (h₁ : y₁ = (-3)^2 - 2*(-3) + 3)
    (h₂ : y₂ = 1^2 - 2*1 + 3)
    (h₃ : y₃ = (-1/2)^2 - 2*(-1/2) + 3) :
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end parabola_y_order_l1191_119156


namespace perfect_square_condition_l1191_119160

theorem perfect_square_condition (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 18*x + k = (a*x + b)^2) ↔ k = 81 :=
by sorry

end perfect_square_condition_l1191_119160


namespace only_yellow_river_certain_l1191_119190

-- Define the type for events
inductive Event
  | MoonlightInFrontOfBed
  | LonelySmokeInDesert
  | ReachForStarsWithHand
  | YellowRiverFlowsIntoSea

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.YellowRiverFlowsIntoSea => True
  | _ => False

-- Theorem stating that only the Yellow River flowing into the sea is certain
theorem only_yellow_river_certain :
  ∀ (e : Event), isCertain e ↔ e = Event.YellowRiverFlowsIntoSea :=
by
  sorry

#check only_yellow_river_certain

end only_yellow_river_certain_l1191_119190


namespace painted_square_ratio_exists_l1191_119112

/-- Represents a square with a painted pattern -/
structure PaintedSquare where
  s : ℝ  -- side length of the square
  w : ℝ  -- width of the brush
  h_positive_s : 0 < s
  h_positive_w : 0 < w
  h_painted_area : w^2 + 2 * Real.sqrt 2 * ((s - w * Real.sqrt 2) / 2)^2 = s^2 / 3

/-- There exists a ratio between the side length and brush width for a painted square -/
theorem painted_square_ratio_exists (ps : PaintedSquare) : 
  ∃ r : ℝ, ps.s = r * ps.w :=
sorry

end painted_square_ratio_exists_l1191_119112


namespace inequality_relationship_l1191_119117

theorem inequality_relationship (a b : ℝ) : 
  (∀ a b, a > b → a + 1 > b - 2) ∧ 
  (∃ a b, a + 1 > b - 2 ∧ ¬(a > b)) := by
sorry

end inequality_relationship_l1191_119117


namespace geometric_sequence_sum_l1191_119177

/-- A geometric sequence with positive terms -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l1191_119177


namespace median_salary_is_worker_salary_l1191_119179

/-- Represents a position in the company -/
inductive Position
  | CEO
  | GeneralManager
  | Manager
  | Supervisor
  | Worker

/-- Information about a position: number of employees and salary -/
structure PositionInfo where
  count : Nat
  salary : Nat

/-- Company salary data -/
def companySalaries : List (Position × PositionInfo) :=
  [(Position.CEO, ⟨1, 150000⟩),
   (Position.GeneralManager, ⟨3, 100000⟩),
   (Position.Manager, ⟨12, 80000⟩),
   (Position.Supervisor, ⟨8, 55000⟩),
   (Position.Worker, ⟨35, 30000⟩)]

/-- Total number of employees -/
def totalEmployees : Nat :=
  companySalaries.foldr (fun (_, info) acc => acc + info.count) 0

/-- Theorem: The median salary of the company is $30,000 -/
theorem median_salary_is_worker_salary :
  let salaries := companySalaries.map (fun (_, info) => info.salary)
  let counts := companySalaries.map (fun (_, info) => info.count)
  let medianIndex := (totalEmployees + 1) / 2
  ∃ (i : Nat), i < salaries.length ∧
    (counts.take i).sum < medianIndex ∧
    medianIndex ≤ (counts.take (i + 1)).sum ∧
    salaries[i]! = 30000 := by
  sorry

#eval totalEmployees -- Should output 59

end median_salary_is_worker_salary_l1191_119179


namespace presidency_meeting_ways_l1191_119105

def num_schools : Nat := 4
def members_per_school : Nat := 6
def host_representatives : Nat := 3
def non_host_representatives : Nat := 2
def seniors_per_school : Nat := 3

theorem presidency_meeting_ways :
  let choose_host := num_schools
  let host_rep_ways := Nat.choose members_per_school host_representatives
  let non_host_school_ways := Nat.choose seniors_per_school 1 * Nat.choose (members_per_school - seniors_per_school) 1
  let non_host_schools_ways := non_host_school_ways ^ (num_schools - 1)
  choose_host * host_rep_ways * non_host_schools_ways = 58320 := by
  sorry

end presidency_meeting_ways_l1191_119105


namespace square_difference_540_460_l1191_119184

theorem square_difference_540_460 : 540^2 - 460^2 = 80000 := by sorry

end square_difference_540_460_l1191_119184


namespace train_length_l1191_119102

/-- Given a train that crosses a platform in a certain time and a signal pole in another time,
    this theorem proves the length of the train. -/
theorem train_length
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 24)
  (h3 : platform_length = 187.5) :
  (platform_crossing_time * platform_length) / (platform_crossing_time - pole_crossing_time) = 300 :=
by sorry

#check train_length

end train_length_l1191_119102


namespace sphere_cross_section_distance_l1191_119171

theorem sphere_cross_section_distance (V : ℝ) (A : ℝ) (r : ℝ) (r_cross : ℝ) (d : ℝ) :
  V = 4 * Real.sqrt 3 * Real.pi →
  (4 / 3) * Real.pi * r^3 = V →
  A = Real.pi →
  Real.pi * r_cross^2 = A →
  d^2 + r_cross^2 = r^2 →
  d = Real.sqrt 2 := by
  sorry

#check sphere_cross_section_distance

end sphere_cross_section_distance_l1191_119171


namespace milk_problem_l1191_119126

theorem milk_problem (initial_milk : ℚ) (tim_fraction : ℚ) (kim_fraction : ℚ) : 
  initial_milk = 3/4 →
  tim_fraction = 1/3 →
  kim_fraction = 1/2 →
  kim_fraction * (initial_milk - tim_fraction * initial_milk) = 1/4 := by
  sorry

end milk_problem_l1191_119126


namespace one_greater_one_less_than_one_l1191_119159

theorem one_greater_one_less_than_one (a b : ℝ) (h : ((1 + a * b) / (a + b))^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (-1 < a ∧ a < 1 ∧ b > 1) :=
sorry

end one_greater_one_less_than_one_l1191_119159


namespace problem_solution_l1191_119180

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) ∧ 
   (∀ m' : ℝ, m' < m → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a * b > m') ∧
   m = 1/4) ∧
  (∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -2 ≤ x ∧ x ≤ 6) :=
by sorry

end problem_solution_l1191_119180


namespace square_of_119_l1191_119153

theorem square_of_119 : 119^2 = 14161 := by
  sorry

end square_of_119_l1191_119153


namespace dogs_not_liking_either_l1191_119186

theorem dogs_not_liking_either (total : ℕ) (watermelon : ℕ) (salmon : ℕ) (both : ℕ)
  (h1 : total = 75)
  (h2 : watermelon = 12)
  (h3 : salmon = 55)
  (h4 : both = 7) :
  total - (watermelon + salmon - both) = 15 := by
  sorry

end dogs_not_liking_either_l1191_119186


namespace opposite_of_sqrt_three_l1191_119145

theorem opposite_of_sqrt_three : 
  -(Real.sqrt 3) = -Real.sqrt 3 := by sorry

end opposite_of_sqrt_three_l1191_119145


namespace negation_equivalence_l1191_119118

theorem negation_equivalence :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, (n : ℝ) < x) :=
by sorry

end negation_equivalence_l1191_119118


namespace consecutive_odd_numbers_equation_l1191_119173

/-- Given three consecutive odd numbers where the first is 7, 
    prove that the multiple of the third number that satisfies 
    the equation is 3 --/
theorem consecutive_odd_numbers_equation (n : ℕ) : 
  let first := 7
  let second := first + 2
  let third := second + 2
  8 * first = n * third + 5 + 2 * second → n = 3 := by
  sorry

end consecutive_odd_numbers_equation_l1191_119173


namespace snooker_tournament_revenue_l1191_119175

theorem snooker_tournament_revenue
  (vip_price : ℚ)
  (general_price : ℚ)
  (total_tickets : ℕ)
  (ticket_difference : ℕ)
  (h1 : vip_price = 45)
  (h2 : general_price = 20)
  (h3 : total_tickets = 320)
  (h4 : ticket_difference = 276)
  : ∃ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = total_tickets ∧
    vip_tickets = general_tickets - ticket_difference ∧
    vip_price * vip_tickets + general_price * general_tickets = 6950 := by
  sorry

#check snooker_tournament_revenue

end snooker_tournament_revenue_l1191_119175


namespace closest_perfect_square_to_350_l1191_119127

def closest_perfect_square (n : ℕ) : ℕ :=
  let root := n.sqrt
  if (root + 1)^2 - n < n - root^2
  then (root + 1)^2
  else root^2

theorem closest_perfect_square_to_350 :
  closest_perfect_square 350 = 361 := by
  sorry

end closest_perfect_square_to_350_l1191_119127


namespace instrument_players_fraction_l1191_119106

theorem instrument_players_fraction (total : ℕ) (two_or_more : ℕ) (prob_one : ℚ) :
  total = 800 →
  two_or_more = 32 →
  prob_one = 1/10 + 3/50 →
  (((prob_one * total) + two_or_more) : ℚ) / total = 1/5 := by
  sorry

end instrument_players_fraction_l1191_119106


namespace inscribed_circle_radius_l1191_119137

theorem inscribed_circle_radius : ∃ (r : ℝ), 
  (1 / r = 1 / 6 + 1 / 10 + 1 / 15 + 3 * Real.sqrt (1 / (6 * 10) + 1 / (6 * 15) + 1 / (10 * 15))) ∧
  r = 30 / (10 * Real.sqrt 26 + 3) := by
  sorry

end inscribed_circle_radius_l1191_119137


namespace largest_minus_smallest_is_52_l1191_119132

def digits : Finset Nat := {8, 3, 4, 6}

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_number (n : Nat) : Prop :=
  is_two_digit n ∧ ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

theorem largest_minus_smallest_is_52 :
  ∃ (max min : Nat),
    valid_number max ∧
    valid_number min ∧
    (∀ n, valid_number n → n ≤ max) ∧
    (∀ n, valid_number n → min ≤ n) ∧
    max - min = 52 := by
  sorry

end largest_minus_smallest_is_52_l1191_119132


namespace factorization_equality_l1191_119165

theorem factorization_equality (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2)^2 := by
  sorry

end factorization_equality_l1191_119165


namespace square_difference_pattern_l1191_119150

theorem square_difference_pattern (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end square_difference_pattern_l1191_119150


namespace c_profit_is_400_l1191_119185

/-- Represents the investment and profit distribution for three individuals --/
structure BusinessInvestment where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ

/-- Calculates C's share of the profit based on the given investments and total profit --/
def c_profit_share (investment : BusinessInvestment) : ℕ :=
  (investment.c_investment * investment.total_profit) / (investment.a_investment + investment.b_investment + investment.c_investment)

/-- Theorem stating that C's share of the profit is 400 given the specific investments and total profit --/
theorem c_profit_is_400 (investment : BusinessInvestment)
  (h1 : investment.a_investment = 800)
  (h2 : investment.b_investment = 1000)
  (h3 : investment.c_investment = 1200)
  (h4 : investment.total_profit = 1000) :
  c_profit_share investment = 400 := by
  sorry

#eval c_profit_share ⟨800, 1000, 1200, 1000⟩

end c_profit_is_400_l1191_119185


namespace count_valid_arrangements_l1191_119181

/-- Represents a valid arrangement of multiples of 2013 in a table -/
def ValidArrangement : Type :=
  { arr : Fin 11 → Fin 11 // Function.Injective arr ∧ 
    ∀ i : Fin 11, (2013 * (arr i + 1)) % (i + 1) = 0 }

/-- The number of valid arrangements -/
def numValidArrangements : ℕ := sorry

theorem count_valid_arrangements : numValidArrangements = 24 := by
  sorry

end count_valid_arrangements_l1191_119181


namespace mary_crayons_left_l1191_119116

/-- The number of crayons Mary has left after giving some away and breaking some -/
def crayons_left (initial_green initial_blue initial_yellow : ℚ)
  (given_green given_blue given_yellow broken_yellow : ℚ) : ℚ :=
  (initial_green - given_green) + (initial_blue - given_blue) + (initial_yellow - given_yellow - broken_yellow)

/-- Theorem stating that Mary has 12 crayons left -/
theorem mary_crayons_left :
  crayons_left 5 8 7 3.5 1.25 2.75 0.5 = 12 := by
  sorry

end mary_crayons_left_l1191_119116


namespace shirt_production_l1191_119115

theorem shirt_production (machines1 machines2 : ℕ) 
  (production1 production2 : ℕ) (time1 time2 : ℕ) : 
  machines1 = 12 → 
  machines2 = 20 → 
  production1 = 24 → 
  production2 = 45 → 
  time1 = 18 → 
  time2 = 22 → 
  production1 * time1 + production2 * time2 = 1422 := by
sorry

end shirt_production_l1191_119115


namespace same_parity_iff_square_sum_l1191_119131

theorem same_parity_iff_square_sum (a b : ℤ) :
  (∃ k : ℤ, a - b = 2 * k) ↔ (∃ c d : ℤ, a^2 + b^2 + c^2 + 1 = d^2) := by sorry

end same_parity_iff_square_sum_l1191_119131


namespace binomial_coefficient_60_2_l1191_119154

theorem binomial_coefficient_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end binomial_coefficient_60_2_l1191_119154


namespace ellipse_and_line_equations_l1191_119141

/-- Given an ellipse with the specified properties, prove its standard equation and the equations of the intersecting line. -/
theorem ellipse_and_line_equations 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e = Real.sqrt 2 / 2) 
  (h_point : a^2 * (1/2)^2 + b^2 * (Real.sqrt 2 / 2)^2 = 1) 
  (k : ℝ) 
  (h_intersection : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / (2 * a^2) + y₁^2 / (2 * b^2) = 1 ∧
    x₂^2 / (2 * a^2) + y₂^2 / (2 * b^2) = 1 ∧
    y₁ = k * (x₁ + 1) ∧
    y₂ = k * (x₂ + 1) ∧
    ((x₁ - 1)^2 + y₁^2 + (x₂ - 1)^2 + y₂^2 + 2 * ((x₁ - 1) * (x₂ - 1) + y₁ * y₂))^(1/2) = 2 * Real.sqrt 26 / 3) :
  (a^2 = 2 ∧ b^2 = 1) ∧ (k = 1 ∨ k = -1) := by
  sorry


end ellipse_and_line_equations_l1191_119141


namespace min_product_xyz_l1191_119128

theorem min_product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq_one : x + y + z = 1) (z_eq_2x : z = 2 * x) (y_eq_3x : y = 3 * x) :
  x * y * z ≥ 1 / 36 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀ + y₀ + z₀ = 1 ∧ z₀ = 2 * x₀ ∧ y₀ = 3 * x₀ ∧ x₀ * y₀ * z₀ = 1 / 36 :=
by sorry

end min_product_xyz_l1191_119128


namespace phi_value_l1191_119142

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is decreasing on an interval [a, b] if for all x, y in [a, b],
    x < y implies f(x) > f(y) -/
def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem phi_value (f : ℝ → ℝ) (φ : ℝ) 
    (h1 : f = λ x => 2 * Real.sin (2 * x + φ + π / 3))
    (h2 : IsOdd f)
    (h3 : IsDecreasingOn f 0 (π / 4)) :
    φ = 2 * π / 3 := by
  sorry

end phi_value_l1191_119142


namespace sqrt_3_times_sqrt_12_l1191_119178

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l1191_119178


namespace fraction_equality_and_sum_l1191_119124

theorem fraction_equality_and_sum : ∃! (α β : ℝ),
  (∀ x : ℝ, x ≠ -β → x ≠ -110.36 →
    (x - α) / (x + β) = (x^2 - 64*x + 1007) / (x^2 + 81*x - 3240)) ∧
  α + β = 146.483 := by
  sorry

end fraction_equality_and_sum_l1191_119124


namespace cafeteria_bill_l1191_119155

/-- The total amount spent by Mell and her friends at the cafeteria -/
theorem cafeteria_bill (coffee_price ice_cream_price cake_price : ℕ) 
  (h1 : coffee_price = 4)
  (h2 : ice_cream_price = 3)
  (h3 : cake_price = 7)
  (mell_coffee mell_cake : ℕ)
  (h4 : mell_coffee = 2)
  (h5 : mell_cake = 1)
  (friend_count : ℕ)
  (h6 : friend_count = 2) :
  (mell_coffee * (friend_count + 1) * coffee_price) + 
  (mell_cake * (friend_count + 1) * cake_price) + 
  (friend_count * ice_cream_price) = 51 :=
by sorry

end cafeteria_bill_l1191_119155


namespace sin_sum_product_l1191_119133

theorem sin_sum_product (x : ℝ) : 
  Real.sin (7 * x) + Real.sin (9 * x) = 2 * Real.sin (8 * x) * Real.cos x := by
  sorry

end sin_sum_product_l1191_119133


namespace conference_handshakes_l1191_119103

theorem conference_handshakes (n : ℕ) (h : n = 7) : 
  (2 * n) * ((2 * n - 1) - 2) / 2 = 77 := by
  sorry

end conference_handshakes_l1191_119103


namespace quadratic_root_difference_l1191_119122

theorem quadratic_root_difference (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 2 ∧ r₂ ≤ 2 ∧ 
   x^2 - 5*x + 6 = (x - r₁) * (x - r₂)) → 
  ∃ r₁ r₂ : ℝ, r₁ - r₂ = 1 ∧ 
   x^2 - 5*x + 6 = (x - r₁) * (x - r₂) :=
by sorry

end quadratic_root_difference_l1191_119122


namespace probability_coprime_pairs_l1191_119139

def S : Finset Nat := Finset.range 8

theorem probability_coprime_pairs (a b : Nat) (h : a ∈ S ∧ b ∈ S ∧ a ≠ b) :
  (Finset.filter (fun p : Nat × Nat => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ Nat.gcd p.1 p.2 = 1) 
    (S.product S)).card / (Finset.filter (fun p : Nat × Nat => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2) 
    (S.product S)).card = 3 / 4 := by
  sorry

end probability_coprime_pairs_l1191_119139


namespace quadratic_root_k_value_l1191_119170

theorem quadratic_root_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x - 6 = 0 ∧ x = -3) → k = -1 := by
  sorry

end quadratic_root_k_value_l1191_119170
