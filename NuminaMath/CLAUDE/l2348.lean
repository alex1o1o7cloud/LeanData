import Mathlib

namespace max_sum_constrained_integers_l2348_234824

theorem max_sum_constrained_integers (a b c d e f g : ℕ) 
  (eq1 : a + b + c = 2)
  (eq2 : b + c + d = 2)
  (eq3 : c + d + e = 2)
  (eq4 : d + e + f = 2)
  (eq5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 :=
by sorry

end max_sum_constrained_integers_l2348_234824


namespace problem_solution_l2348_234819

theorem problem_solution (a b A : ℝ) 
  (h1 : 3^a = A) 
  (h2 : 5^b = A) 
  (h3 : 1/a + 1/b = 2) : 
  A = Real.sqrt 15 := by
  sorry

end problem_solution_l2348_234819


namespace negative_sum_distribution_l2348_234829

theorem negative_sum_distribution (x y : ℝ) : -(x + y) = -x - y := by
  sorry

end negative_sum_distribution_l2348_234829


namespace triangle_problem_l2348_234825

/-- Triangle with side lengths a, b, c and inradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- The main theorem stating the properties of the three triangles -/
theorem triangle_problem (H₁ H₂ H₃ : Triangle) : 
  (∃ (d : ℝ), H₁.b = (H₁.a + H₁.c) / 2 ∧ 
               H₁.a = H₁.b - d ∧ 
               H₁.c = H₁.b + d) →
  (H₂.a = H₁.a - 10 ∧ H₂.b = H₁.b - 10 ∧ H₂.c = H₁.c - 10) →
  (H₃.a = H₁.a + 14 ∧ H₃.b = H₁.b + 14 ∧ H₃.c = H₁.c + 14) →
  (H₂.r = H₁.r - 5) →
  (H₃.r = H₁.r + 5) →
  (H₁.a = 25 ∧ H₁.b = 38 ∧ H₁.c = 51) := by
sorry

end triangle_problem_l2348_234825


namespace unique_root_condition_l2348_234839

theorem unique_root_condition (a : ℝ) : 
  (∃! x, Real.log (x - 2*a) - 3*(x - 2*a)^2 + 2*a = 0) ↔ 
  (a = (Real.log 6 + 1) / 4) := by sorry

end unique_root_condition_l2348_234839


namespace ellipse_set_is_ellipse_l2348_234842

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define points A and B
variable (A B : E)

-- Define the set of points P satisfying the condition
def ellipse_set (A B : E) : Set E :=
  {P : E | dist P A + dist P B = 2 * dist A B}

-- Theorem statement
theorem ellipse_set_is_ellipse (A B : E) (h : A ≠ B) :
  ∃ (C : E) (a b : ℝ), a > b ∧ b > 0 ∧
    ellipse_set A B = {P : E | (dist P C)^2 / a^2 + (dist P (C + (B - A)))^2 / b^2 = 1} :=
sorry

end ellipse_set_is_ellipse_l2348_234842


namespace negation_equivalence_l2348_234885

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by
  sorry

end negation_equivalence_l2348_234885


namespace polymerization_of_tetrafluoroethylene_yields_teflon_l2348_234835

-- Define the monomer tetrafluoroethylene
structure Tetrafluoroethylene : Type :=
  (formula : String)

-- Define the polymer Teflon (PTFE)
structure Teflon : Type :=
  (formula : String)

-- Define the polymerization process
def polymerize (monomer : Tetrafluoroethylene) : Teflon :=
  sorry

-- Theorem statement
theorem polymerization_of_tetrafluoroethylene_yields_teflon 
  (monomer : Tetrafluoroethylene) 
  (h : monomer.formula = "CF2=CF2") :
  (polymerize monomer).formula = "(-CF2-CF2-)n" :=
sorry

end polymerization_of_tetrafluoroethylene_yields_teflon_l2348_234835


namespace cubic_polynomial_root_property_l2348_234897

/-- Given a cubic polynomial x^3 + ax^2 + bx + 16a where a and b are nonzero integers,
    if two of its roots coincide and all three roots are integers,
    then |ab| = 2496 -/
theorem cubic_polynomial_root_property (a b : ℤ) : 
  a ≠ 0 → b ≠ 0 → 
  (∃ r s : ℤ, (X - r)^2 * (X - s) = X^3 + a*X^2 + b*X + 16*a) →
  |a * b| = 2496 := by
  sorry

end cubic_polynomial_root_property_l2348_234897


namespace multiples_of_15_between_21_and_205_l2348_234815

theorem multiples_of_15_between_21_and_205 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 21 ∧ n < 205) (Finset.range 205)).card = 12 := by
  sorry

end multiples_of_15_between_21_and_205_l2348_234815


namespace abs_sum_minimum_l2348_234828

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 5| + |x + 6| + |x + 7| ≥ 5 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| + |y + 7| = 5 := by
  sorry

#check abs_sum_minimum

end abs_sum_minimum_l2348_234828


namespace tom_remaining_seashells_l2348_234805

def initial_seashells : ℕ := 5
def seashells_given_away : ℕ := 2

theorem tom_remaining_seashells : 
  initial_seashells - seashells_given_away = 3 := by sorry

end tom_remaining_seashells_l2348_234805


namespace trig_identity_l2348_234873

theorem trig_identity (α : Real) (h : Real.sin (2 * Real.pi / 3 + α) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - α) = -1 / 3 := by
  sorry

end trig_identity_l2348_234873


namespace function_symmetry_l2348_234868

def is_symmetric_about_one (g : ℝ → ℝ) : Prop :=
  ∀ x, g (1 - x) = g (1 + x)

theorem function_symmetry 
  (f : ℝ → ℝ) 
  (h1 : f 0 = 0)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ t, f (1 - t) - f (1 + t) + 4 * t = 0) :
  is_symmetric_about_one (λ x => f x - 2 * x) := by
sorry

end function_symmetry_l2348_234868


namespace four_digit_integer_transformation_l2348_234813

theorem four_digit_integer_transformation (A : ℕ) (n : ℕ) :
  (A ≥ 1000 ∧ A < 10000) →
  (∃ a b c d : ℕ,
    A = 1000 * a + 100 * b + 10 * c + d ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n) = n * A) →
  A = 1818 :=
by sorry

end four_digit_integer_transformation_l2348_234813


namespace exactly_two_cubic_polynomials_satisfy_l2348_234892

/-- A polynomial function of degree 3 or less -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

/-- The condition that f(x)f(-x) = f(x^3) for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that exactly two cubic polynomials satisfy the condition -/
theorem exactly_two_cubic_polynomials_satisfy :
  ∃! (s : Finset (ℝ → ℝ)),
    (∀ f ∈ s, ∃ a b c d, f = CubicPolynomial a b c d) ∧
    (∀ f ∈ s, SatisfiesCondition f) ∧
    s.card = 2 :=
sorry

end exactly_two_cubic_polynomials_satisfy_l2348_234892


namespace line_equation_proof_l2348_234808

/-- Given two lines in the 2D plane, we define them as parallel if they have the same slope. -/
def parallel_lines (m1 b1 m2 b2 : ℝ) : Prop :=
  m1 = m2

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line given by y = mx + b -/
def point_on_line (p : Point) (m b : ℝ) : Prop :=
  p.y = m * p.x + b

theorem line_equation_proof :
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x - y + 3 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2 * x - y - 8 = 0
  let point_A : Point := ⟨2, -4⟩
  parallel_lines 2 3 2 (-8) ∧ point_on_line point_A 2 (-8) :=
by
  sorry

end line_equation_proof_l2348_234808


namespace extended_triangle_theorem_l2348_234841

-- Define the triangle ABC
variable (A B C : Point) (ABC : Triangle A B C)

-- Define the condition BC = 2AC
variable (h1 : BC = 2 * AC)

-- Define point D such that AD = 1/3 * AB
variable (D : Point)
variable (h2 : AD = (1/3) * AB)

-- Theorem statement
theorem extended_triangle_theorem : CD = 2 * AD := by
  sorry

end extended_triangle_theorem_l2348_234841


namespace garden_expenses_l2348_234814

/-- Calculate the total expenses for flowers in a garden --/
theorem garden_expenses (tulips carnations roses : ℕ) (price : ℚ) : 
  tulips = 250 → 
  carnations = 375 → 
  roses = 320 → 
  price = 2 → 
  (tulips + carnations + roses : ℚ) * price = 1890 := by
sorry

end garden_expenses_l2348_234814


namespace train_length_l2348_234843

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : 
  speed_kmh = 75.6 → time_sec = 21 → speed_kmh * (1000 / 3600) * time_sec = 441 := by
  sorry

#check train_length

end train_length_l2348_234843


namespace city_visit_selection_schemes_l2348_234851

theorem city_visit_selection_schemes :
  let total_people : ℕ := 6
  let selected_people : ℕ := 4
  let total_cities : ℕ := 4
  let restricted_people : ℕ := 2
  let restricted_cities : ℕ := 1

  (total_people - restricted_people) *
  (total_people - 1) *
  (total_people - 2) *
  (total_people - 3) = 240 :=
by
  sorry

end city_visit_selection_schemes_l2348_234851


namespace triangle_with_perimeter_7_l2348_234854

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_with_perimeter_7 :
  ∀ a b c : ℕ,
  a + b + c = 7 →
  is_valid_triangle a b c →
  (a = 1 ∨ a = 2 ∨ a = 3) ∧
  (b = 1 ∨ b = 2 ∨ b = 3) ∧
  (c = 1 ∨ c = 2 ∨ c = 3) :=
by sorry

end triangle_with_perimeter_7_l2348_234854


namespace power_of_point_l2348_234800

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a line passing through two points
structure Line where
  point1 : Point
  point2 : Point

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the intersection of a line and a circle
def intersect (l : Line) (c : Circle) : Option (Point × Point) := sorry

-- Theorem statement
theorem power_of_point (S : Circle) (P A B A1 B1 : Point) 
  (l1 l2 : Line) : 
  l1.point1 = P → l2.point1 = P → 
  intersect l1 S = some (A, B) → 
  intersect l2 S = some (A1, B1) → 
  distance P A * distance P B = distance P A1 * distance P B1 := by
  sorry

end power_of_point_l2348_234800


namespace quadratic_roots_range_l2348_234855

/-- Proves the range of m for which x^2 - 3x - m = 0 has two unequal real roots -/
theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 3*x - m = 0 ∧ y^2 - 3*y - m = 0) ↔ m > -9/4 := by
  sorry

end quadratic_roots_range_l2348_234855


namespace girls_together_arrangements_boy_A_not_end_two_girls_together_arrangements_l2348_234877

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the function for the number of arrangements when three girls must stand together
def arrangements_girls_together : ℕ := sorry

-- Define the function for the number of arrangements when boy A cannot stand at either end and exactly two girls stand together
def arrangements_boy_A_not_end_two_girls_together : ℕ := sorry

-- Theorem for the first question
theorem girls_together_arrangements :
  arrangements_girls_together = 144 := by sorry

-- Theorem for the second question
theorem boy_A_not_end_two_girls_together_arrangements :
  arrangements_boy_A_not_end_two_girls_together = 288 := by sorry

end girls_together_arrangements_boy_A_not_end_two_girls_together_arrangements_l2348_234877


namespace opposite_event_of_hit_at_least_once_l2348_234823

-- Define the sample space for two shots
inductive ShotOutcome
  | Hit
  | Miss

-- Define the event of hitting the target at least once
def hitAtLeastOnce (outcome1 outcome2 : ShotOutcome) : Prop :=
  outcome1 = ShotOutcome.Hit ∨ outcome2 = ShotOutcome.Hit

-- Define the event of both shots missing
def bothShotsMiss (outcome1 outcome2 : ShotOutcome) : Prop :=
  outcome1 = ShotOutcome.Miss ∧ outcome2 = ShotOutcome.Miss

-- Theorem statement
theorem opposite_event_of_hit_at_least_once 
  (outcome1 outcome2 : ShotOutcome) : 
  ¬(hitAtLeastOnce outcome1 outcome2) ↔ bothShotsMiss outcome1 outcome2 := by
  sorry

end opposite_event_of_hit_at_least_once_l2348_234823


namespace balboa_earned_180_l2348_234866

/-- Represents the earnings of students from three middle schools --/
structure SchoolEarnings where
  allen_students : Nat
  allen_days : Nat
  balboa_students : Nat
  balboa_days : Nat
  carver_students : Nat
  carver_days : Nat
  total_paid : Nat

/-- Calculates the total earnings for Balboa school students --/
def balboa_earnings (e : SchoolEarnings) : Nat :=
  let total_student_days := e.allen_students * e.allen_days + 
                            e.balboa_students * e.balboa_days + 
                            e.carver_students * e.carver_days
  let daily_wage := e.total_paid / total_student_days
  daily_wage * e.balboa_students * e.balboa_days

/-- Theorem stating that Balboa school students earned 180 dollars --/
theorem balboa_earned_180 (e : SchoolEarnings) 
  (h1 : e.allen_students = 7)
  (h2 : e.allen_days = 3)
  (h3 : e.balboa_students = 4)
  (h4 : e.balboa_days = 5)
  (h5 : e.carver_students = 5)
  (h6 : e.carver_days = 9)
  (h7 : e.total_paid = 744) :
  balboa_earnings e = 180 := by
  sorry

end balboa_earned_180_l2348_234866


namespace append_digit_square_difference_l2348_234870

theorem append_digit_square_difference (x y : ℕ) : 
  x > 0 → y ≤ 9 → (10 * x + y - x^2 = 8 * x) → 
  ((x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8)) := by
  sorry

end append_digit_square_difference_l2348_234870


namespace root_in_interval_l2348_234869

-- Define the function f(x) = 2x - 3
def f (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end root_in_interval_l2348_234869


namespace positive_expression_l2348_234831

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + 3 * b^2 := by
  sorry

end positive_expression_l2348_234831


namespace elvis_studio_time_l2348_234890

/-- Calculates the total time spent in the studio for Elvis's album production -/
def total_studio_time (num_songs : ℕ) (record_time : ℕ) (edit_time : ℕ) (write_time : ℕ) : ℚ :=
  let total_minutes := num_songs * (record_time + write_time) + edit_time
  total_minutes / 60

/-- Proves that Elvis spent 5 hours in the studio given the specified conditions -/
theorem elvis_studio_time :
  total_studio_time 10 12 30 15 = 5 := by
sorry

end elvis_studio_time_l2348_234890


namespace chinese_remainder_theorem_l2348_234880

theorem chinese_remainder_theorem (x : ℤ) :
  (2 + x) % (2^4) = 3^2 % (2^4) ∧
  (3 + x) % (3^4) = 2^3 % (3^4) ∧
  (5 + x) % (5^4) = 7^2 % (5^4) →
  x % 30 = 14 := by
sorry

end chinese_remainder_theorem_l2348_234880


namespace method_doubles_method_power_of_two_l2348_234891

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a row of coins -/
def CoinRow (N : ℕ) := Fin N → CoinState

/-- Represents a method for the magician to guess the number -/
structure GuessMethod (N : ℕ) :=
(guess : CoinRow N → Fin N)

/-- States that if a method exists for N coins, it exists for 2N coins -/
theorem method_doubles {N : ℕ} (h : GuessMethod N) : GuessMethod (2 * N) :=
sorry

/-- States that the method only works for powers of 2 -/
theorem method_power_of_two {N : ℕ} : GuessMethod N → ∃ k : ℕ, N = 2^k :=
sorry

end method_doubles_method_power_of_two_l2348_234891


namespace gcd_count_for_product_360_l2348_234853

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), S.card = 8 ∧ ∀ d, d ∈ S ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) :=
sorry

end gcd_count_for_product_360_l2348_234853


namespace unique_number_with_properties_l2348_234847

/-- Given a natural number n, returns the sum of its digits. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Given a two-digit number n, returns the number formed by reversing its digits. -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Predicate that checks if a number is two-digit. -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem unique_number_with_properties : 
  ∃! n : ℕ, is_two_digit n ∧ 
    n = 4 * (sum_of_digits n) + 3 ∧ 
    n + 18 = reverse_digits n := by sorry

end unique_number_with_properties_l2348_234847


namespace complementary_event_l2348_234803

/-- The sample space of outcomes when two students purchase a beverage with a chance of winning a prize -/
inductive Outcome
  | BothWin
  | AWinsBLoses
  | ALosesBWins
  | BothLose

/-- The event where both students win a prize -/
def bothWin (o : Outcome) : Prop :=
  o = Outcome.BothWin

/-- The event where at most one student wins a prize -/
def atMostOneWins (o : Outcome) : Prop :=
  o = Outcome.AWinsBLoses ∨ o = Outcome.ALosesBWins ∨ o = Outcome.BothLose

/-- Theorem stating that the complementary event to "both win" is "at most one wins" -/
theorem complementary_event :
  ∀ o : Outcome, ¬(bothWin o) ↔ atMostOneWins o :=
sorry


end complementary_event_l2348_234803


namespace no_real_roots_quadratic_l2348_234845

theorem no_real_roots_quadratic : ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0 := by
  sorry

end no_real_roots_quadratic_l2348_234845


namespace value_of_expression_l2348_234894

theorem value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 3) 
  (h2 : m*n + n^2 = 4) : 
  m^2 + 3*m*n + n^2 = 7 := by
  sorry

end value_of_expression_l2348_234894


namespace circumscribed_quadrilateral_arc_angles_l2348_234860

theorem circumscribed_quadrilateral_arc_angles (a b c d : ℝ) :
  let x := (b + c + d) / 2
  let y := (a + c + d) / 2
  let z := (a + b + d) / 2
  let t := (a + b + c) / 2
  a + b + c + d = 360 →
  x + y + z + t = 540 := by
sorry

end circumscribed_quadrilateral_arc_angles_l2348_234860


namespace product_multiple_of_five_probability_l2348_234822

def N : ℕ := 2020

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def count_multiples_of_five : ℕ := N / 5

def prob_not_multiple_of_five : ℚ := (N - count_multiples_of_five) / N

theorem product_multiple_of_five_probability :
  let p := 1 - (prob_not_multiple_of_five * (prob_not_multiple_of_five - 1 / N) * (prob_not_multiple_of_five - 2 / N))
  ∃ ε > 0, |p - 0.485| < ε :=
sorry

end product_multiple_of_five_probability_l2348_234822


namespace stacy_pages_per_day_l2348_234871

/-- Given a paper with a certain number of pages due in a certain number of days,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℚ :=
  total_pages / total_days

/-- Theorem: Stacy needs to write 1 page per day to finish her paper on time. -/
theorem stacy_pages_per_day :
  pages_per_day 12 12 = 1 := by
  sorry

#eval pages_per_day 12 12

end stacy_pages_per_day_l2348_234871


namespace boys_from_clay_middle_school_l2348_234848

theorem boys_from_clay_middle_school 
  (total_students : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (jonas_students : ℕ)
  (clay_students : ℕ)
  (jonas_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : total_boys = 52)
  (h3 : total_girls = 48)
  (h4 : jonas_students = 40)
  (h5 : clay_students = 60)
  (h6 : jonas_girls = 20)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = jonas_students + clay_students)
  : ∃ (clay_boys : ℕ), clay_boys = 32 ∧ 
    clay_boys + (total_boys - clay_boys) = total_boys ∧
    clay_boys + (clay_students - clay_boys) = clay_students :=
by sorry

end boys_from_clay_middle_school_l2348_234848


namespace three_digit_perfect_cube_divisible_by_25_l2348_234807

theorem three_digit_perfect_cube_divisible_by_25 : 
  ∃! (n : ℕ), 100 ≤ 125 * n^3 ∧ 125 * n^3 ≤ 999 := by
  sorry

end three_digit_perfect_cube_divisible_by_25_l2348_234807


namespace x_squared_minus_y_squared_l2348_234804

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8) (h2 : x - y = 3/8) : x^2 - y^2 = 15/64 := by
  sorry

end x_squared_minus_y_squared_l2348_234804


namespace largest_square_area_l2348_234811

-- Define the right triangle XYZ
structure RightTriangle where
  xy : ℝ  -- length of side XY
  xz : ℝ  -- length of side XZ
  yz : ℝ  -- length of hypotenuse YZ
  right_angle : xy^2 + xz^2 = yz^2  -- Pythagorean theorem

-- Define the theorem
theorem largest_square_area (t : RightTriangle) 
  (sum_of_squares : t.xy^2 + t.xz^2 + t.yz^2 = 450) :
  t.yz^2 = 225 := by
  sorry


end largest_square_area_l2348_234811


namespace real_part_of_complex_number_l2348_234849

theorem real_part_of_complex_number (z : ℂ) (h : z - Complex.abs z = -8 + 12*I) : 
  Complex.re z = 5 := by sorry

end real_part_of_complex_number_l2348_234849


namespace seagrass_study_l2348_234875

/-- Represents the sample statistics for a town -/
structure TownSample where
  size : ℕ
  mean : ℝ
  variance : ℝ

/-- Represents the competition probabilities for town A -/
structure CompetitionProbs where
  win_in_A : ℝ
  win_in_B : ℝ

theorem seagrass_study (town_A : TownSample) (town_B : TownSample) (probs : CompetitionProbs)
  (h_A_size : town_A.size = 12)
  (h_A_mean : town_A.mean = 18)
  (h_A_var : town_A.variance = 19)
  (h_B_size : town_B.size = 18)
  (h_B_mean : town_B.mean = 36)
  (h_B_var : town_B.variance = 70)
  (h_prob_A : probs.win_in_A = 3/5)
  (h_prob_B : probs.win_in_B = 1/2) :
  let total_mean := (town_A.size * town_A.mean + town_B.size * town_B.mean) / (town_A.size + town_B.size)
  let total_variance := (1 / (town_A.size + town_B.size)) *
    (town_A.size * town_A.variance + town_A.size * (town_A.mean - total_mean)^2 +
     town_B.size * town_B.variance + town_B.size * (town_B.mean - total_mean)^2)
  let expected_score := 0 * (1 - probs.win_in_A)^2 + 1 * (2 * probs.win_in_A * (1 - probs.win_in_B) * (1 - probs.win_in_A)) +
    2 * (1 - (1 - probs.win_in_A)^2 - 2 * probs.win_in_A * (1 - probs.win_in_B) * (1 - probs.win_in_A))
  total_mean = 28.8 ∧ total_variance = 127.36 ∧ expected_score = 36/25 := by
  sorry

end seagrass_study_l2348_234875


namespace sqrt_115_between_consecutive_integers_product_l2348_234863

theorem sqrt_115_between_consecutive_integers_product :
  ∃ (n : ℕ), n > 0 ∧ (n : ℝ) < Real.sqrt 115 ∧ Real.sqrt 115 < (n + 1) ∧ n * (n + 1) = 110 := by
  sorry

end sqrt_115_between_consecutive_integers_product_l2348_234863


namespace solution_to_equation_l2348_234820

theorem solution_to_equation : ∃ x y : ℝ, x + 2 * y = 4 ∧ x = 0 ∧ y = 2 := by
  sorry

end solution_to_equation_l2348_234820


namespace common_terms_count_l2348_234876

theorem common_terms_count : 
  (Finset.filter (fun k => 15 * k + 8 ≤ 2018) (Finset.range (2019 / 15 + 1))).card = 135 :=
by sorry

end common_terms_count_l2348_234876


namespace solution_set_t_3_nonnegative_for_all_x_l2348_234882

-- Define the function f
def f (t x : ℝ) : ℝ := x^2 - (t + 1)*x + t

-- Theorem 1: Solution set when t = 3
theorem solution_set_t_3 :
  {x : ℝ | f 3 x > 0} = Set.Iio 1 ∪ Set.Ioi 3 :=
sorry

-- Theorem 2: Condition for f(x) ≥ 0 for all real x
theorem nonnegative_for_all_x :
  (∀ x : ℝ, f t x ≥ 0) ↔ t = 1 :=
sorry

end solution_set_t_3_nonnegative_for_all_x_l2348_234882


namespace fourth_vertex_of_complex_rectangle_l2348_234850

/-- A rectangle in the complex plane --/
structure ComplexRectangle where
  a : ℂ
  b : ℂ
  c : ℂ
  d : ℂ
  is_rectangle : (b - a).arg.cos * (c - b).arg.cos + (b - a).arg.sin * (c - b).arg.sin = 0

/-- The theorem stating that given three vertices of a rectangle in the complex plane,
    we can determine the fourth vertex --/
theorem fourth_vertex_of_complex_rectangle (r : ComplexRectangle)
  (h1 : r.a = 3 + 2*I)
  (h2 : r.b = 1 + I)
  (h3 : r.c = -1 - 2*I) :
  r.d = -3 - 3*I := by
  sorry

#check fourth_vertex_of_complex_rectangle

end fourth_vertex_of_complex_rectangle_l2348_234850


namespace ohara_triple_36_25_l2348_234879

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt a + Real.sqrt b = x

/-- Theorem: If (36,25,x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_36_25 (x : ℕ) :
  is_ohara_triple 36 25 x → x = 11 := by
  sorry

end ohara_triple_36_25_l2348_234879


namespace generalized_inequality_l2348_234812

theorem generalized_inequality (x : ℝ) (n : ℕ) (h : x > 0) :
  x^n + n/x > n + 1 := by sorry

end generalized_inequality_l2348_234812


namespace cube_volume_from_face_perimeter_l2348_234888

/-- Given a cube with face perimeter of 20 cm, prove its volume is 125 cubic centimeters -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 20) : 
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 125 := by sorry

end cube_volume_from_face_perimeter_l2348_234888


namespace quadratic_roots_properties_l2348_234834

theorem quadratic_roots_properties (a b m : ℝ) : 
  m > 0 → 
  2 * a^2 - 8 * a + m = 0 → 
  2 * b^2 - 8 * b + m = 0 → 
  (a^2 + b^2 ≥ 8) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2) ∧ 
  (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * Real.sqrt 2) / 12) := by
  sorry

end quadratic_roots_properties_l2348_234834


namespace pandemic_cut_fifty_percent_l2348_234818

/-- Represents a car factory with its production details -/
structure CarFactory where
  doorsPerCar : ℕ
  initialProduction : ℕ
  metalShortageDecrease : ℕ
  finalDoorProduction : ℕ

/-- Calculates the percentage of production cut due to a pandemic -/
def pandemicProductionCutPercentage (factory : CarFactory) : ℚ :=
  let productionAfterMetalShortage := factory.initialProduction - factory.metalShortageDecrease
  let finalCarProduction := factory.finalDoorProduction / factory.doorsPerCar
  let pandemicCut := productionAfterMetalShortage - finalCarProduction
  (pandemicCut / productionAfterMetalShortage) * 100

/-- Theorem stating that the pandemic production cut percentage is 50% for the given factory conditions -/
theorem pandemic_cut_fifty_percent (factory : CarFactory) 
  (h1 : factory.doorsPerCar = 5)
  (h2 : factory.initialProduction = 200)
  (h3 : factory.metalShortageDecrease = 50)
  (h4 : factory.finalDoorProduction = 375) :
  pandemicProductionCutPercentage factory = 50 := by
  sorry

#eval pandemicProductionCutPercentage ⟨5, 200, 50, 375⟩

end pandemic_cut_fifty_percent_l2348_234818


namespace find_k_l2348_234858

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) (k : ℝ) : ℝ := 5 * n^2 + k * n

/-- The nth term of the sequence {a_n} -/
def a (n : ℕ) (k : ℝ) : ℝ := S n k - S (n-1) k

theorem find_k : ∃ k : ℝ, (∀ n : ℕ, S n k = 5 * n^2 + k * n) ∧ a 2 k = 18 → k = 3 :=
sorry

end find_k_l2348_234858


namespace variance_transformation_l2348_234837

/-- Given three real numbers with variance 1, prove that multiplying each by 3 and adding 2 results in a variance of 9. -/
theorem variance_transformation (a₁ a₂ a₃ : ℝ) (μ : ℝ) : 
  (1 / 3 : ℝ) * ((a₁ - μ)^2 + (a₂ - μ)^2 + (a₃ - μ)^2) = 1 →
  (1 / 3 : ℝ) * (((3 * a₁ + 2) - (3 * μ + 2))^2 + 
                 ((3 * a₂ + 2) - (3 * μ + 2))^2 + 
                 ((3 * a₃ + 2) - (3 * μ + 2))^2) = 9 :=
by sorry

end variance_transformation_l2348_234837


namespace most_accurate_value_for_given_K_l2348_234864

/-- Given a scientific constant K and its error margin, 
    returns the most accurate value with all digits significant -/
def most_accurate_value (K : ℝ) (error : ℝ) : ℝ :=
  sorry

theorem most_accurate_value_for_given_K :
  let K : ℝ := 3.68547
  let error : ℝ := 0.00256
  most_accurate_value K error = 3.7 := by sorry

end most_accurate_value_for_given_K_l2348_234864


namespace smoking_lung_disease_relation_l2348_234889

/-- Represents the Chi-square statistic -/
def K_squared : ℝ := 5.231

/-- The probability that K^2 is greater than or equal to 3.841 -/
def P_3_841 : ℝ := 0.05

/-- The probability that K^2 is greater than or equal to 6.635 -/
def P_6_635 : ℝ := 0.01

/-- The confidence level for the relationship between smoking and lung disease -/
def confidence_level : ℝ := 1 - P_3_841

/-- Theorem stating that there is more than 95% confidence that smoking is related to lung disease -/
theorem smoking_lung_disease_relation :
  K_squared > 3.841 ∧ confidence_level > 0.95 := by sorry

end smoking_lung_disease_relation_l2348_234889


namespace polynomial_non_negative_l2348_234861

theorem polynomial_non_negative (x : ℝ) : x^12 - x^7 - x^5 + 1 ≥ 0 := by
  sorry

end polynomial_non_negative_l2348_234861


namespace positive_solution_x_l2348_234881

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 10 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (x_pos : x > 0) :
  x = 3 := by
sorry

end positive_solution_x_l2348_234881


namespace complex_magnitude_problem_l2348_234899

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l2348_234899


namespace intersects_three_points_iff_m_range_l2348_234852

/-- A quadratic function f(x) = x^2 + 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- Predicate indicating if f intersects the coordinate axes at 3 points -/
def intersects_at_three_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m 0 ≠ 0

/-- Theorem stating the range of m for which f intersects the coordinate axes at 3 points -/
theorem intersects_three_points_iff_m_range (m : ℝ) :
  intersects_at_three_points m ↔ m < 1 ∧ m ≠ 0 := by sorry

end intersects_three_points_iff_m_range_l2348_234852


namespace tangent_slope_at_pi_over_six_l2348_234886

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - 2 * Real.cos x

theorem tangent_slope_at_pi_over_six :
  deriv f (π/6) = 3/2 := by sorry

end tangent_slope_at_pi_over_six_l2348_234886


namespace flower_purchase_analysis_l2348_234821

/-- Represents the number and cost of different flower types --/
structure FlowerPurchase where
  roses : ℕ
  lilies : ℕ
  sunflowers : ℕ
  daisies : ℕ
  rose_cost : ℚ
  lily_cost : ℚ
  sunflower_cost : ℚ
  daisy_cost : ℚ

/-- Calculates the total cost of the flower purchase --/
def total_cost (purchase : FlowerPurchase) : ℚ :=
  purchase.roses * purchase.rose_cost +
  purchase.lilies * purchase.lily_cost +
  purchase.sunflowers * purchase.sunflower_cost +
  purchase.daisies * purchase.daisy_cost

/-- Calculates the total number of flowers --/
def total_flowers (purchase : FlowerPurchase) : ℕ :=
  purchase.roses + purchase.lilies + purchase.sunflowers + purchase.daisies

/-- Calculates the percentage of a specific flower type --/
def flower_percentage (count : ℕ) (total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

/-- Theorem stating the total cost and percentages of flowers --/
theorem flower_purchase_analysis (purchase : FlowerPurchase)
  (h1 : purchase.roses = 50)
  (h2 : purchase.lilies = 40)
  (h3 : purchase.sunflowers = 30)
  (h4 : purchase.daisies = 20)
  (h5 : purchase.rose_cost = 2)
  (h6 : purchase.lily_cost = 3/2)
  (h7 : purchase.sunflower_cost = 1)
  (h8 : purchase.daisy_cost = 3/4) :
  total_cost purchase = 205 ∧
  flower_percentage purchase.roses (total_flowers purchase) = 35.71 ∧
  flower_percentage purchase.lilies (total_flowers purchase) = 28.57 ∧
  flower_percentage purchase.sunflowers (total_flowers purchase) = 21.43 ∧
  flower_percentage purchase.daisies (total_flowers purchase) = 14.29 := by
  sorry

end flower_purchase_analysis_l2348_234821


namespace boat_distance_proof_l2348_234856

theorem boat_distance_proof (boat_speed : ℝ) (stream_speed : ℝ) (time_difference : ℝ) :
  boat_speed = 10 →
  stream_speed = 2 →
  time_difference = 1.5 →
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  ∃ distance : ℝ,
    distance / upstream_speed = distance / downstream_speed + time_difference ∧
    distance = 36 := by
  sorry

end boat_distance_proof_l2348_234856


namespace arithmetic_sequence_sum_l2348_234816

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- Theorem: If S_10 = 12 and S_20 = 17, then S_30 = 22 for an arithmetic sequence -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 10 = 12) (h2 : seq.S 20 = 17) : seq.S 30 = 22 := by
  sorry

end arithmetic_sequence_sum_l2348_234816


namespace consecutive_squares_difference_l2348_234846

theorem consecutive_squares_difference (t : ℕ) : 
  (t + 1)^2 - t^2 = 191 → t^2 = 9025 :=
by
  sorry

end consecutive_squares_difference_l2348_234846


namespace sum_of_ten_and_thousand_cube_equals_1010_scientific_notation_of_1010_sum_equals_scientific_notation_l2348_234833

theorem sum_of_ten_and_thousand_cube_equals_1010 : 10 + 10^3 = 1010 := by
  sorry

theorem scientific_notation_of_1010 : 1010 = 1.01 * 10^3 := by
  sorry

theorem sum_equals_scientific_notation : 10 + 10^3 = 1.01 * 10^3 := by
  sorry

end sum_of_ten_and_thousand_cube_equals_1010_scientific_notation_of_1010_sum_equals_scientific_notation_l2348_234833


namespace geometric_sequence_increasing_iff_q_gt_one_l2348_234830

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = q * a n)

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_increasing_iff_q_gt_one (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q → (IncreasingSequence a ↔ q > 1) :=
by sorry

end geometric_sequence_increasing_iff_q_gt_one_l2348_234830


namespace intersection_M_N_l2348_234865

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

theorem intersection_M_N : M ∩ N = {y : ℝ | y ≥ 1} := by sorry

end intersection_M_N_l2348_234865


namespace f_1988_11_equals_169_l2348_234844

/-- Sum of digits of a positive integer -/
def sumOfDigits (k : ℕ+) : ℕ := sorry

/-- Square of sum of digits -/
def f₁ (k : ℕ+) : ℕ := (sumOfDigits k) ^ 2

/-- Recursive definition of fₙ -/
def f (n : ℕ) (k : ℕ+) : ℕ :=
  match n with
  | 0 => k.val
  | 1 => f₁ k
  | n + 1 => f₁ ⟨f n k, sorry⟩

/-- The main theorem to prove -/
theorem f_1988_11_equals_169 : f 1988 11 = 169 := by sorry

end f_1988_11_equals_169_l2348_234844


namespace odd_periodic_function_value_l2348_234802

-- Define the properties of the function f
def is_odd_and_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = -f x)

-- State the theorem
theorem odd_periodic_function_value (f : ℝ → ℝ) (h : is_odd_and_periodic f) : 
  f 2008 = 0 := by
  sorry

end odd_periodic_function_value_l2348_234802


namespace wire_length_ratio_l2348_234809

/-- The ratio of wire lengths in cube frame construction -/
theorem wire_length_ratio : 
  ∀ (bonnie_wire_length roark_wire_length : ℕ) 
    (bonnie_cube_volume roark_total_volume : ℕ),
  bonnie_wire_length = 12 * 8 →
  bonnie_cube_volume = 8^3 →
  roark_total_volume = bonnie_cube_volume →
  (∃ (num_small_cubes : ℕ), 
    roark_total_volume = num_small_cubes * 2^3 ∧
    roark_wire_length = num_small_cubes * 12 * 2) →
  (bonnie_wire_length : ℚ) / roark_wire_length = 1 / 16 := by
sorry

end wire_length_ratio_l2348_234809


namespace sum_of_decimals_l2348_234838

theorem sum_of_decimals : 5.46 + 4.537 = 9.997 := by
  sorry

end sum_of_decimals_l2348_234838


namespace parking_spot_difference_l2348_234893

/-- Represents the number of open parking spots on each level of a 4-story parking area -/
structure ParkingArea where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating the difference in open spots between second and first levels -/
theorem parking_spot_difference (p : ParkingArea) : 
  p.first = 4 → 
  p.third = p.second + 6 → 
  p.fourth = 14 → 
  p.first + p.second + p.third + p.fourth = 46 → 
  p.second - p.first = 7 := by
  sorry

#check parking_spot_difference

end parking_spot_difference_l2348_234893


namespace average_weight_abc_l2348_234826

theorem average_weight_abc (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 43)
  (h3 : b = 31) :
  (a + b + c) / 3 = 45 := by
sorry

end average_weight_abc_l2348_234826


namespace rational_root_l2348_234896

theorem rational_root (x : ℝ) (hx : x ≠ 0) 
  (h1 : ∃ r : ℚ, x^5 = r) 
  (h2 : ∃ p : ℚ, 20*x + 19/x = p) : 
  ∃ q : ℚ, x = q := by
sorry

end rational_root_l2348_234896


namespace smallest_c_for_three_in_range_l2348_234895

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- Theorem statement
theorem smallest_c_for_three_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = 3) ↔ c ≥ 12 := by sorry

end smallest_c_for_three_in_range_l2348_234895


namespace summer_mowing_count_l2348_234840

/-- The number of times Ned mowed his lawn in the spring -/
def spring_mows : ℕ := 6

/-- The total number of times Ned mowed his lawn -/
def total_mows : ℕ := 11

/-- The number of times Ned mowed his lawn in the summer -/
def summer_mows : ℕ := total_mows - spring_mows

theorem summer_mowing_count : summer_mows = 5 := by
  sorry

end summer_mowing_count_l2348_234840


namespace triangle_side_values_l2348_234801

theorem triangle_side_values (n : ℕ) : 
  (3 * n - 3 > 0) ∧ 
  (2 * n + 12 > 0) ∧ 
  (2 * n + 7 > 0) ∧ 
  (3 * n - 3 + 2 * n + 7 > 2 * n + 12) ∧
  (3 * n - 3 + 2 * n + 12 > 2 * n + 7) ∧
  (2 * n + 7 + 2 * n + 12 > 3 * n - 3) ∧
  (2 * n + 12 > 2 * n + 7) ∧
  (2 * n + 7 > 3 * n - 3) →
  (∃ (count : ℕ), count = 7 ∧ 
    (∀ (m : ℕ), (m ≥ 1 ∧ m ≤ count) ↔ 
      (∃ (k : ℕ), k ≥ 3 ∧ k ≤ 9 ∧
        (3 * k - 3 > 0) ∧ 
        (2 * k + 12 > 0) ∧ 
        (2 * k + 7 > 0) ∧ 
        (3 * k - 3 + 2 * k + 7 > 2 * k + 12) ∧
        (3 * k - 3 + 2 * k + 12 > 2 * k + 7) ∧
        (2 * k + 7 + 2 * k + 12 > 3 * k - 3) ∧
        (2 * k + 12 > 2 * k + 7) ∧
        (2 * k + 7 > 3 * k - 3)))) :=
by sorry

end triangle_side_values_l2348_234801


namespace largest_x_value_l2348_234827

theorem largest_x_value (x : ℝ) :
  (x / 3 + 1 / (7 * x) = 1 / 2) →
  x ≤ (21 + Real.sqrt 105) / 28 ∧
  ∃ y : ℝ, y / 3 + 1 / (7 * y) = 1 / 2 ∧ y = (21 + Real.sqrt 105) / 28 :=
by sorry

end largest_x_value_l2348_234827


namespace tower_house_block_difference_l2348_234862

def blocks_for_tower : ℕ := 50
def blocks_for_house : ℕ := 20

theorem tower_house_block_difference :
  blocks_for_tower - blocks_for_house = 30 :=
by sorry

end tower_house_block_difference_l2348_234862


namespace problem_solution_l2348_234867

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def sum_probability (n : ℕ) : ℚ := (3 : ℚ) / choose n 2

def binomial_coefficient (n k : ℕ) : ℤ := (choose n k : ℤ)

def a (n k : ℕ) : ℤ := binomial_coefficient n k * (-2)^k

theorem problem_solution :
  ∃ (n : ℕ),
    (sum_probability n = 3/28) ∧
    (a 8 3 = -448) ∧
    (((choose 5 2 * choose 4 1 + choose 4 3) : ℚ) / choose 9 3 = 11/21) :=
sorry

end problem_solution_l2348_234867


namespace cupcake_cost_l2348_234887

/-- Proves that the cost of a cupcake is 40 cents given initial amount, juice box cost, and remaining amount --/
theorem cupcake_cost (initial_amount : ℕ) (juice_cost : ℕ) (remaining : ℕ) :
  initial_amount = 75 →
  juice_cost = 27 →
  remaining = 8 →
  initial_amount - juice_cost - remaining = 40 :=
by
  sorry

end cupcake_cost_l2348_234887


namespace cube_sum_of_sqrt_equals_24_l2348_234883

theorem cube_sum_of_sqrt_equals_24 :
  (Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3))^3 = 24 := by
  sorry

end cube_sum_of_sqrt_equals_24_l2348_234883


namespace john_payment_l2348_234810

def hearing_aid_cost : ℝ := 2500
def insurance_coverage_percent : ℝ := 80
def number_of_hearing_aids : ℕ := 2

theorem john_payment (total_cost : ℝ) (insurance_payment : ℝ) (john_payment : ℝ) :
  total_cost = hearing_aid_cost * number_of_hearing_aids →
  insurance_payment = (insurance_coverage_percent / 100) * total_cost →
  john_payment = total_cost - insurance_payment →
  john_payment = 1000 := by sorry

end john_payment_l2348_234810


namespace nonreal_cube_root_of_unity_sum_l2348_234878

theorem nonreal_cube_root_of_unity_sum (ω : ℂ) : 
  ω^3 = 1 ∧ ω ≠ 1 → (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := by sorry

end nonreal_cube_root_of_unity_sum_l2348_234878


namespace rectangle_max_area_l2348_234884

/-- The maximum area of a rectangle with perimeter 40 meters is 100 square meters. -/
theorem rectangle_max_area (x : ℝ) :
  let perimeter := 40
  let width := x
  let length := (perimeter / 2) - x
  let area := width * length
  (∀ y, 0 < y ∧ y < perimeter / 2 → area ≥ y * (perimeter / 2 - y)) →
  area ≤ 100 ∧ ∃ z, 0 < z ∧ z < perimeter / 2 ∧ z * (perimeter / 2 - z) = 100 :=
by sorry

end rectangle_max_area_l2348_234884


namespace P_degree_P_terms_P_descending_y_l2348_234836

/-- The polynomial P(x,y) = 3x²y - xy² - 3xy³ + x⁵ - 1 -/
def P (x y : ℝ) : ℝ := 3*x^2*y - x*y^2 - 3*x*y^3 + x^5 - 1

/-- The degree of polynomial P is 5 -/
theorem P_degree : 
  ∃ (n : ℕ), n = 5 ∧ (∀ m : ℕ, (∃ (x y : ℝ), P x y ≠ 0 → m ≤ n)) ∧ 
  (∃ (x y : ℝ), P x y ≠ 0 ∧ n = 5) :=
sorry

/-- P has 5 terms -/
theorem P_terms : 
  ∃ (t : ℕ), t = 5 ∧ (∀ (x y : ℝ), P x y = 3*x^2*y - x*y^2 - 3*x*y^3 + x^5 - 1) :=
sorry

/-- When arranged in descending order of y, P = -3xy³ - xy² + 3x²y + x⁵ - 1 -/
theorem P_descending_y :
  ∀ (x y : ℝ), P x y = -3*x*y^3 - x*y^2 + 3*x^2*y + x^5 - 1 :=
sorry

end P_degree_P_terms_P_descending_y_l2348_234836


namespace ramanujan_number_l2348_234898

theorem ramanujan_number (h r : ℂ) : 
  h * r = 40 + 24 * I ∧ h = 3 + 7 * I → r = 4 - (104 / 29) * I :=
by sorry

end ramanujan_number_l2348_234898


namespace three_primes_product_sum_l2348_234817

theorem three_primes_product_sum : 
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p < q ∧ q < r ∧
    p * q * r = 5 * (p + q + r) ∧
    p = 2 ∧ q = 5 ∧ r = 7 := by
  sorry

end three_primes_product_sum_l2348_234817


namespace maria_quiz_goal_l2348_234806

theorem maria_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (quizzes_taken : ℕ) (as_earned : ℕ) (remaining_lower_a : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 70 / 100 →
  quizzes_taken = 35 →
  as_earned = 28 →
  remaining_lower_a = 11 →
  (as_earned + (total_quizzes - quizzes_taken - remaining_lower_a) : ℚ) / total_quizzes ≥ goal_percentage := by
  sorry

#check maria_quiz_goal

end maria_quiz_goal_l2348_234806


namespace seven_twentyfour_twentyfive_pythagorean_triple_l2348_234859

/-- A Pythagorean triple consists of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- Prove that 7, 24, and 25 form a Pythagorean triple -/
theorem seven_twentyfour_twentyfive_pythagorean_triple :
  is_pythagorean_triple 7 24 25 := by
sorry

end seven_twentyfour_twentyfive_pythagorean_triple_l2348_234859


namespace max_y_value_l2348_234857

theorem max_y_value (a b y : ℝ) (eq1 : a + b + y = 5) (eq2 : a * b + b * y + a * y = 3) :
  y ≤ 13/3 := by
sorry

end max_y_value_l2348_234857


namespace gcd_of_specific_numbers_l2348_234872

theorem gcd_of_specific_numbers : Nat.gcd 123456789 987654321 = 9 := by
  sorry

end gcd_of_specific_numbers_l2348_234872


namespace probability_diamond_ace_face_card_l2348_234874

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (favorable_cards : ℕ)
  (h_total : total_cards = 54)
  (h_favorable : favorable_cards = 26)

/-- The probability of selecting at least one favorable card in two draws with replacement -/
def probability_favorable_card (d : Deck) : ℚ :=
  1 - (↑(d.total_cards - d.favorable_cards) / ↑d.total_cards) ^ 2

theorem probability_diamond_ace_face_card :
  ∃ d : Deck, probability_favorable_card d = 533 / 729 := by
  sorry

end probability_diamond_ace_face_card_l2348_234874


namespace probability_three_different_suits_l2348_234832

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of selecting three cards of different suits from a standard deck without replacement -/
def probabilityDifferentSuits : ℚ :=
  (CardsPerSuit * (StandardDeck - CardsPerSuit) * (StandardDeck - 2 * CardsPerSuit)) /
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem probability_three_different_suits :
  probabilityDifferentSuits = 169 / 425 := by
  sorry

end probability_three_different_suits_l2348_234832
