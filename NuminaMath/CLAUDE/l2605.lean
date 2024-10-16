import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2605_260585

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2605_260585


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2605_260573

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y+1) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/(b+1) = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2605_260573


namespace NUMINAMATH_CALUDE_angle_sum_360_l2605_260575

theorem angle_sum_360 (k : ℝ) : k + 90 = 360 → k = 270 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_360_l2605_260575


namespace NUMINAMATH_CALUDE_isosceles_triangle_locus_l2605_260518

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

/-- The locus equation for point C -/
def LocusEquation (p : Point) : Prop :=
  p.x^2 + p.y^2 - 3*p.x + p.y = 2

theorem isosceles_triangle_locus :
  ∀ (C : Point),
    let A : Point := ⟨3, -2⟩
    let B : Point := ⟨0, 1⟩
    let M : Point := ⟨3/2, -1/2⟩  -- Midpoint of AB
    let r : ℝ := (3 * Real.sqrt 2) / 2  -- Radius of the circle
    C ≠ A ∧ C ≠ B →  -- Exclude points A and B
    (CircleEquation M r C ↔ LocusEquation C) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_locus_l2605_260518


namespace NUMINAMATH_CALUDE_grants_score_l2605_260558

/-- Given the scores of three students on a math test, prove Grant's score. -/
theorem grants_score (hunter_score john_score grant_score : ℕ) : 
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end NUMINAMATH_CALUDE_grants_score_l2605_260558


namespace NUMINAMATH_CALUDE_line_transformation_l2605_260504

/-- Given a line l: ax + y - 7 = 0 transformed by matrix A to line l': 9x + y - 91 = 0,
    prove that a = 2 and b = 13 -/
theorem line_transformation (a b : ℝ) : 
  (∀ x y : ℝ, a * x + y - 7 = 0 → 
    9 * (3 * x) + (-x + b * y) - 91 = 0) → 
  a = 2 ∧ b = 13 := by
sorry

end NUMINAMATH_CALUDE_line_transformation_l2605_260504


namespace NUMINAMATH_CALUDE_point_distance_on_x_axis_l2605_260593

theorem point_distance_on_x_axis (a : ℝ) : 
  let A : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (-3, 0)
  (‖A - B‖ = 5) → (a = -8 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_point_distance_on_x_axis_l2605_260593


namespace NUMINAMATH_CALUDE_odd_integer_not_divides_power_plus_one_l2605_260587

theorem odd_integer_not_divides_power_plus_one (n m : ℕ) : 
  n > 1 → Odd n → m ≥ 1 → ¬(n ∣ m^(n-1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_not_divides_power_plus_one_l2605_260587


namespace NUMINAMATH_CALUDE_f_inequality_iff_x_gt_one_l2605_260572

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - 1) + Real.exp (x - 1) - Real.exp (1 - x) - x + 1

theorem f_inequality_iff_x_gt_one :
  ∀ x : ℝ, f x + f (3 - 2*x) < 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_iff_x_gt_one_l2605_260572


namespace NUMINAMATH_CALUDE_dave_money_l2605_260598

theorem dave_money (dave_amount : ℝ) : 
  (2 / 3 * (3 * dave_amount - 12) = 84) → dave_amount = 46 := by
  sorry

end NUMINAMATH_CALUDE_dave_money_l2605_260598


namespace NUMINAMATH_CALUDE_simplest_form_fraction_l2605_260510

/-- A fraction is in simplest form if its numerator and denominator have no common factors
    other than 1 and -1, and neither the numerator nor denominator can be factored further. -/
def IsSimplestForm (n d : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, (n x y ≠ 0 ∨ d x y ≠ 0) →
    ∀ f : ℝ → ℝ → ℝ, (f x y ∣ n x y) ∧ (f x y ∣ d x y) → f x y = 1 ∨ f x y = -1

/-- The fraction (x^2 + y^2) / (x + y) is in simplest form. -/
theorem simplest_form_fraction (x y : ℝ) :
    IsSimplestForm (fun x y => x^2 + y^2) (fun x y => x + y) := by
  sorry

#check simplest_form_fraction

end NUMINAMATH_CALUDE_simplest_form_fraction_l2605_260510


namespace NUMINAMATH_CALUDE_percentage_of_360_l2605_260569

theorem percentage_of_360 : (32 / 100) * 360 = 115.2 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_l2605_260569


namespace NUMINAMATH_CALUDE_lisa_spoon_count_l2605_260543

/-- The total number of spoons Lisa has after combining old and new sets -/
def total_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ) 
                 (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * baby_spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Theorem stating that Lisa has 39 spoons in total -/
theorem lisa_spoon_count : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoon_count_l2605_260543


namespace NUMINAMATH_CALUDE_abc_sum_sixteen_l2605_260568

theorem abc_sum_sixteen (a b c : ℤ) 
  (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4)
  (h4 : ¬(a = b ∧ b = c))
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) :
  a + b + c = 16 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_sixteen_l2605_260568


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2605_260547

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 6*x + 2) * q + (22*x^2 + 8*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2605_260547


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l2605_260529

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := sorry

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := 4518

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- Theorem stating that the amount of strawberry jelly is 1792 grams -/
theorem strawberry_jelly_amount : strawberry_jelly = 1792 :=
  by
    sorry

/-- Lemma stating that the sum of strawberry and blueberry jelly equals the total jelly -/
lemma jelly_sum : strawberry_jelly + blueberry_jelly = total_jelly :=
  by
    sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l2605_260529


namespace NUMINAMATH_CALUDE_middle_term_value_l2605_260537

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℤ  -- First term
  b : ℤ  -- Second term
  c : ℤ  -- Third term
  is_arithmetic : b - a = c - b

/-- The problem statement -/
theorem middle_term_value (seq : ArithmeticSequence3) 
  (h1 : seq.a = 2^3)
  (h2 : seq.c = 2^5) : 
  seq.b = 20 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_value_l2605_260537


namespace NUMINAMATH_CALUDE_sin_equality_necessary_not_sufficient_l2605_260557

theorem sin_equality_necessary_not_sufficient :
  (∀ A B : ℝ, A = B → Real.sin A = Real.sin B) ∧
  (∃ A B : ℝ, Real.sin A = Real.sin B ∧ A ≠ B) :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_necessary_not_sufficient_l2605_260557


namespace NUMINAMATH_CALUDE_largest_n_for_product_4021_l2605_260534

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1 : ℤ) * seq.diff

theorem largest_n_for_product_4021 (a b : ArithmeticSequence)
    (h1 : a.first = 1)
    (h2 : b.first = 1)
    (h3 : a.diff ≤ b.diff)
    (h4 : ∃ n : ℕ, a.nthTerm n * b.nthTerm n = 4021) :
    (∀ m : ℕ, a.nthTerm m * b.nthTerm m = 4021 → m ≤ 11) ∧
    (∃ n : ℕ, n = 11 ∧ a.nthTerm n * b.nthTerm n = 4021) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_product_4021_l2605_260534


namespace NUMINAMATH_CALUDE_lcm_problem_l2605_260512

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 945) (h2 : Nat.lcm b c = 525) :
  Nat.lcm a c = 675 ∨ Nat.lcm a c = 4725 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2605_260512


namespace NUMINAMATH_CALUDE_student_A_most_stable_l2605_260545

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : Real

/-- Theorem: Given the variances of four students' scores, prove that student A has the most stable performance -/
theorem student_A_most_stable
  (students : Finset Student)
  (hA : Student.mk "A" 3.8 ∈ students)
  (hB : Student.mk "B" 5.5 ∈ students)
  (hC : Student.mk "C" 10 ∈ students)
  (hD : Student.mk "D" 6 ∈ students)
  (h_count : students.card = 4)
  : ∀ s ∈ students, (Student.mk "A" 3.8).variance ≤ s.variance :=
by sorry


end NUMINAMATH_CALUDE_student_A_most_stable_l2605_260545


namespace NUMINAMATH_CALUDE_circle_center_transformation_l2605_260578

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (3, -4)
  let reflected_x := reflect_x initial_center
  let reflected_y := reflect_y reflected_x
  let final_center := translate reflected_y 5 3
  final_center = (2, 7) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l2605_260578


namespace NUMINAMATH_CALUDE_shelf_filling_theorem_l2605_260570

/-- Represents the number of books that can fill a shelf -/
structure ShelfFilling where
  P : ℕ+  -- Programming books
  B : ℕ+  -- Biology books
  F : ℕ+  -- Physics books
  R : ℕ+  -- Another count of programming books
  C : ℕ+  -- Another count of biology books
  Q : ℕ+  -- The number we want to determine
  distinct : P ≠ B ∧ P ≠ F ∧ P ≠ R ∧ P ≠ C ∧ P ≠ Q ∧
             B ≠ F ∧ B ≠ R ∧ B ≠ C ∧ B ≠ Q ∧
             F ≠ R ∧ F ≠ C ∧ F ≠ Q ∧
             R ≠ C ∧ R ≠ Q ∧
             C ≠ Q

/-- The theorem stating that Q = R + 2C for a valid shelf filling -/
theorem shelf_filling_theorem (sf : ShelfFilling) : sf.Q = sf.R + 2 * sf.C := by
  sorry

end NUMINAMATH_CALUDE_shelf_filling_theorem_l2605_260570


namespace NUMINAMATH_CALUDE_complex_square_l2605_260590

theorem complex_square (z : ℂ) (i : ℂ) : z = 2 - 3 * i → i^2 = -1 → z^2 = -5 - 12 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l2605_260590


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l2605_260542

/-- Represents the daily profit function for a mall's product sales -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

/-- Theorem stating that a price reduction of $12 results in a daily profit of $3572 -/
theorem optimal_price_reduction (initial_sales : ℕ) (initial_profit : ℝ)
    (h1 : initial_sales = 70)
    (h2 : initial_profit = 50) :
    daily_profit initial_sales initial_profit 12 = 3572 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_l2605_260542


namespace NUMINAMATH_CALUDE_julia_short_amount_l2605_260577

/-- Represents the cost and quantity of CDs Julia wants to buy -/
structure CDPurchase where
  rock_price : ℕ
  pop_price : ℕ
  dance_price : ℕ
  country_price : ℕ
  quantity : ℕ

/-- Calculates the amount Julia is short given her CD purchase and available money -/
def amount_short (purchase : CDPurchase) (available_money : ℕ) : ℕ :=
  let total_cost := purchase.quantity * (purchase.rock_price + purchase.pop_price + purchase.dance_price + purchase.country_price)
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Julia is short $25 given the specific CD prices, quantities, and available money -/
theorem julia_short_amount : amount_short ⟨5, 10, 3, 7, 4⟩ 75 = 25 := by
  sorry

end NUMINAMATH_CALUDE_julia_short_amount_l2605_260577


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2605_260500

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250) 
  (h2 : bridge_length = 300) 
  (h3 : crossing_time = 45) : 
  ∃ (speed : ℝ), abs (speed - (train_length + bridge_length) / crossing_time) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2605_260500


namespace NUMINAMATH_CALUDE_min_packages_lcm_l2605_260549

/-- The load capacity of Sarah's trucks -/
def sarah_capacity : ℕ := 18

/-- The load capacity of Ryan's trucks -/
def ryan_capacity : ℕ := 11

/-- The load capacity of Emily's trucks -/
def emily_capacity : ℕ := 15

/-- The minimum number of packages each business must have shipped -/
def min_packages : ℕ := 990

theorem min_packages_lcm :
  Nat.lcm (Nat.lcm sarah_capacity ryan_capacity) emily_capacity = min_packages :=
sorry

end NUMINAMATH_CALUDE_min_packages_lcm_l2605_260549


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2605_260567

/-- Quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k + 1)*x + k^2 + 1

/-- Discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + 1)

/-- Theorem stating the conditions for distinct real roots and the value of k -/
theorem quadratic_roots_theorem (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ k > 3/4 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧
   |x₁| + |x₂| = x₁ * x₂ → k = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2605_260567


namespace NUMINAMATH_CALUDE_cricket_team_handedness_l2605_260502

theorem cricket_team_handedness (total_players : Nat) (throwers : Nat) (right_handed : Nat)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : right_handed = 51)
    (h4 : throwers ≤ right_handed) :
    (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_handedness_l2605_260502


namespace NUMINAMATH_CALUDE_comic_stacking_arrangements_l2605_260540

def spiderman_comics : ℕ := 8
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 3

def total_comics : ℕ := spiderman_comics + archie_comics + garfield_comics

def garfield_group_positions : ℕ := 3

theorem comic_stacking_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * garfield_group_positions) = 8669760 := by
  sorry

end NUMINAMATH_CALUDE_comic_stacking_arrangements_l2605_260540


namespace NUMINAMATH_CALUDE_equality_from_sum_of_squares_l2605_260552

theorem equality_from_sum_of_squares (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a → a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equality_from_sum_of_squares_l2605_260552


namespace NUMINAMATH_CALUDE_slope_greater_than_one_line_passes_through_point_distance_not_sqrt_two_not_four_lines_l2605_260526

-- Define the line equations
def line1 (x y : ℝ) : Prop := 5 * x - 4 * y + 1 = 0
def line2 (m x y : ℝ) : Prop := (2 + m) * x + 4 * y - 2 + m = 0
def line3 (x y : ℝ) : Prop := x + y - 1 = 0
def line4 (x y : ℝ) : Prop := 2 * x + 2 * y + 1 = 0

-- Define points
def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (3, -1)

-- Statement 1
theorem slope_greater_than_one : 
  ∃ m : ℝ, (∀ x y : ℝ, line1 x y → y = m * x + (1/4)) ∧ m > 1 := by sorry

-- Statement 2
theorem line_passes_through_point :
  ∀ m : ℝ, line2 m (-1) 1 := by sorry

-- Statement 3
theorem distance_not_sqrt_two :
  ∃ d : ℝ, (d = (|1 + 2|) / Real.sqrt (2^2 + 2^2)) ∧ d ≠ Real.sqrt 2 := by sorry

-- Statement 4
theorem not_four_lines :
  ¬(∃ (lines : Finset (ℝ → ℝ → Prop)), lines.card = 4 ∧
    (∀ l ∈ lines, ∃ d1 d2 : ℝ, d1 = 1 ∧ d2 = 4 ∧
      (∀ x y : ℝ, l x y → 
        (Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) = d1 ∧
         Real.sqrt ((x - point_B.1)^2 + (y - point_B.2)^2) = d2)))) := by sorry

end NUMINAMATH_CALUDE_slope_greater_than_one_line_passes_through_point_distance_not_sqrt_two_not_four_lines_l2605_260526


namespace NUMINAMATH_CALUDE_sandwiches_prepared_correct_l2605_260571

/-- The number of sandwiches Ruth prepared -/
def sandwiches_prepared : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def sandwiches_ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth gave to her brother -/
def sandwiches_given_to_brother : ℕ := 2

/-- The number of sandwiches eaten by the first cousin -/
def sandwiches_first_cousin : ℕ := 2

/-- The number of sandwiches eaten by each of the other two cousins -/
def sandwiches_per_other_cousin : ℕ := 1

/-- The number of other cousins who ate sandwiches -/
def number_of_other_cousins : ℕ := 2

/-- The number of sandwiches left at the end -/
def sandwiches_left : ℕ := 3

/-- Theorem stating that the number of sandwiches Ruth prepared is correct -/
theorem sandwiches_prepared_correct : 
  sandwiches_prepared = 
    sandwiches_ruth_ate + 
    sandwiches_given_to_brother + 
    sandwiches_first_cousin + 
    (sandwiches_per_other_cousin * number_of_other_cousins) + 
    sandwiches_left :=
by sorry

end NUMINAMATH_CALUDE_sandwiches_prepared_correct_l2605_260571


namespace NUMINAMATH_CALUDE_hypotenuse_product_square_l2605_260506

-- Define the triangles and their properties
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem hypotenuse_product_square (x y h₁ h₂ : ℝ) :
  right_triangle x (2*y) h₁ →  -- T1
  right_triangle x y h₂ →      -- T2
  x * (2*y) / 2 = 8 →          -- Area of T1
  x * y / 2 = 4 →              -- Area of T2
  (h₁ * h₂)^2 = 160 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_product_square_l2605_260506


namespace NUMINAMATH_CALUDE_existence_of_close_points_l2605_260551

theorem existence_of_close_points :
  ∃ (x y : ℝ), y = x^3 ∧ |y - (x^3 + |x| + 1)| ≤ 1/100 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_close_points_l2605_260551


namespace NUMINAMATH_CALUDE_janes_age_l2605_260531

theorem janes_age (agnes_age : ℕ) (future_years : ℕ) (jane_age : ℕ) : 
  agnes_age = 25 → 
  future_years = 13 → 
  agnes_age + future_years = 2 * (jane_age + future_years) → 
  jane_age = 6 := by
sorry

end NUMINAMATH_CALUDE_janes_age_l2605_260531


namespace NUMINAMATH_CALUDE_three_number_problem_l2605_260579

def is_solution (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 406 ∧
  ∃ p : ℕ, Nat.Prime p ∧ p > 2 ∧ p ∣ a ∧ p ∣ b ∧ p ∣ c ∧
    Nat.Prime (a / p) ∧ Nat.Prime (b / p) ∧ Nat.Prime (c / p)

theorem three_number_problem :
  ∃ a b c : ℕ, is_solution a b c ∧
    ((a = 14 ∧ b = 21 ∧ c = 371) ∨
     (a = 14 ∧ b = 91 ∧ c = 301) ∨
     (a = 14 ∧ b = 133 ∧ c = 259) ∨
     (a = 58 ∧ b = 145 ∧ c = 203)) :=
sorry

end NUMINAMATH_CALUDE_three_number_problem_l2605_260579


namespace NUMINAMATH_CALUDE_divides_fk_iff_divides_f_l2605_260554

theorem divides_fk_iff_divides_f (k : ℕ) (f : ℕ → ℕ) (x : ℕ) :
  (∀ n : ℕ, ∃ m : ℕ, f n = m * n) →
  (x ∣ f^[k] x ↔ x ∣ f x) :=
sorry

end NUMINAMATH_CALUDE_divides_fk_iff_divides_f_l2605_260554


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2605_260541

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 1)}
def set_B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 1)}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2605_260541


namespace NUMINAMATH_CALUDE_average_of_ABCD_l2605_260564

theorem average_of_ABCD (A B C D : ℚ) 
  (eq1 : 1001 * C - 2004 * A = 4008)
  (eq2 : 1001 * B + 3005 * A - 1001 * D = 6010) :
  (A + B + C + D) / 4 = (5 + D) / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ABCD_l2605_260564


namespace NUMINAMATH_CALUDE_picture_ratio_proof_l2605_260595

theorem picture_ratio_proof (total pictures : ℕ) (vertical_count : ℕ) (haphazard_count : ℕ) :
  total = 30 →
  vertical_count = 10 →
  haphazard_count = 5 →
  (total - vertical_count - haphazard_count) * 2 = total := by
  sorry

end NUMINAMATH_CALUDE_picture_ratio_proof_l2605_260595


namespace NUMINAMATH_CALUDE_xiao_ming_tasks_minimum_time_l2605_260566

def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

def minimum_time : ℕ := 85

theorem xiao_ming_tasks_minimum_time :
  minimum_time = max review_time (max rest_time homework_time) :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_tasks_minimum_time_l2605_260566


namespace NUMINAMATH_CALUDE_age_of_B_l2605_260561

/-- Given the initial ratio of ages and the ratio after 2 years, prove B's age is 6 years -/
theorem age_of_B (k : ℚ) (x : ℚ) : 
  (5 * k : ℚ) / (3 * k : ℚ) = 5 / 3 →
  (4 * k : ℚ) / (3 * k : ℚ) = 4 / 3 →
  ((5 * k + 2) : ℚ) / ((3 * k + 2) : ℚ) = 3 / 2 →
  ((3 * k + 2) : ℚ) / ((2 * k + 2) : ℚ) = 2 / x →
  (3 * k : ℚ) = 6 := by
  sorry

#check age_of_B

end NUMINAMATH_CALUDE_age_of_B_l2605_260561


namespace NUMINAMATH_CALUDE_system_solution_proof_l2605_260509

theorem system_solution_proof :
  ∃ (x y : ℝ), 
    (4 * x + y = 12) ∧ 
    (3 * x - 2 * y = -2) ∧ 
    (x = 2) ∧ 
    (y = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2605_260509


namespace NUMINAMATH_CALUDE_m_greater_than_n_l2605_260555

theorem m_greater_than_n : ∀ a : ℝ, 2 * a^2 - 4 * a > a^2 - 2 * a - 3 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l2605_260555


namespace NUMINAMATH_CALUDE_chord_bisected_at_P_l2605_260548

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 = 1

/-- A point is inside the ellipse if the left side of the equation is less than 1 -/
def inside_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 < 1

/-- The fixed point P -/
def P : ℝ × ℝ := (1, 1)

/-- A chord is bisected at a point if that point is the midpoint of the chord -/
def is_bisected_at (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := 2 * x + y - 3 = 0

theorem chord_bisected_at_P :
  inside_ellipse P.1 P.2 →
  ∀ A B : ℝ × ℝ,
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    is_bisected_at A B P →
    ∀ x y : ℝ, (x, y) ∈ Set.Icc A B → line_equation x y :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_at_P_l2605_260548


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2605_260550

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x^2 + 5) % (x - 1) = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2605_260550


namespace NUMINAMATH_CALUDE_family_gathering_arrangements_l2605_260586

theorem family_gathering_arrangements (n : ℕ) (h : n = 6) : 
  Nat.choose n (n / 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_arrangements_l2605_260586


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_cubes_reciprocals_l2605_260524

theorem quadratic_roots_sum_of_cubes_reciprocals 
  (a b c r s : ℝ) 
  (h1 : 3 * a * r^2 + 5 * b * r + 7 * c = 0) 
  (h2 : 3 * a * s^2 + 5 * b * s + 7 * c = 0) 
  (h3 : r ≠ 0) 
  (h4 : s ≠ 0) 
  (h5 : c ≠ 0) : 
  1 / r^3 + 1 / s^3 = (-5 * b * (25 * b^2 - 63 * c)) / (343 * c^3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_cubes_reciprocals_l2605_260524


namespace NUMINAMATH_CALUDE_negation_of_implication_l2605_260588

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → 2*a > 2*b - 1) ↔ (a ≤ b → 2*a ≤ 2*b - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2605_260588


namespace NUMINAMATH_CALUDE_line_through_points_l2605_260511

/-- A line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  point1 : (ℝ × ℝ)
  point2 : (ℝ × ℝ)
  eq1 : a * point1.1 + b = point1.2
  eq2 : a * point2.1 + b = point2.2

/-- Theorem: For a line y = ax + b passing through (3, 4) and (7, 16), a - b = 8 -/
theorem line_through_points (l : Line) 
  (h1 : l.point1 = (3, 4))
  (h2 : l.point2 = (7, 16)) : 
  l.a - l.b = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2605_260511


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l2605_260536

/-- Represents the swimming scenario with two swimmers in a pool. -/
structure SwimmingScenario where
  poolLength : ℝ
  swimmer1Speed : ℝ
  swimmer2Speed : ℝ
  totalTime : ℝ

/-- Calculates the number of times the swimmers pass each other. -/
def numberOfPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, the swimmers pass each other 20 times. -/
theorem swimmers_pass_count (scenario : SwimmingScenario) 
  (h1 : scenario.poolLength = 90)
  (h2 : scenario.swimmer1Speed = 3)
  (h3 : scenario.swimmer2Speed = 2)
  (h4 : scenario.totalTime = 12 * 60) : -- 12 minutes in seconds
  numberOfPasses scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l2605_260536


namespace NUMINAMATH_CALUDE_box_volume_correct_l2605_260562

/-- The volume of an open box formed from a rectangular sheet -/
def boxVolume (x : ℝ) : ℝ := 4 * x^3 - 56 * x^2 + 192 * x

/-- The properties of the box construction -/
structure BoxProperties where
  sheet_length : ℝ
  sheet_width : ℝ
  corner_cut : ℝ
  max_height : ℝ
  h_length : sheet_length = 16
  h_width : sheet_width = 12
  h_max_height : max_height = 6
  h_corner_cut_range : 0 < corner_cut ∧ corner_cut ≤ max_height

/-- Theorem stating that the boxVolume function correctly calculates the volume of the box -/
theorem box_volume_correct (props : BoxProperties) (x : ℝ) 
    (h_x : 0 < x ∧ x ≤ props.max_height) : 
  boxVolume x = (props.sheet_length - 2*x) * (props.sheet_width - 2*x) * x := by
  sorry

#check box_volume_correct

end NUMINAMATH_CALUDE_box_volume_correct_l2605_260562


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2605_260520

/-- Given a circle with equation x^2 + y^2 + 2x - 4y - 4 = 0, prove that its center is at (-1, 2) and its radius is 3 -/
theorem circle_center_and_radius :
  let f : ℝ × ℝ → ℝ := λ (x, y) => x^2 + y^2 + 2*x - 4*y - 4
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 3 ∧
    ∀ (p : ℝ × ℝ), f p = 0 ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l2605_260520


namespace NUMINAMATH_CALUDE_special_polynomial_property_l2605_260576

/-- The polynomial type representing (1-z)^b₁ · (1-z²)^b₂ · (1-z³)^b₃ ··· (1-z³²)^b₃₂ -/
def SpecialPolynomial (b : Fin 32 → ℕ+) : Polynomial ℚ := sorry

/-- The property that after multiplying out and removing terms with degree > 32, 
    the polynomial equals 1 - 2z -/
def HasSpecialProperty (p : Polynomial ℚ) : Prop := sorry

theorem special_polynomial_property (b : Fin 32 → ℕ+) :
  HasSpecialProperty (SpecialPolynomial b) → b 31 = 2^27 - 2^11 := by sorry

end NUMINAMATH_CALUDE_special_polynomial_property_l2605_260576


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2605_260535

theorem complex_fraction_sum (x y : ℝ) : 
  (∃ (z : ℂ), z = (1 + y * Complex.I) / (1 + Complex.I) ∧ (z : ℂ).re = x) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2605_260535


namespace NUMINAMATH_CALUDE_min_cosine_sine_fraction_l2605_260505

open Real

theorem min_cosine_sine_fraction (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (cos x)^3 / sin x + (sin x)^3 / cos x ≥ 1 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (cos y)^3 / sin y + (sin y)^3 / cos y = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_cosine_sine_fraction_l2605_260505


namespace NUMINAMATH_CALUDE_prime_power_sum_l2605_260528

theorem prime_power_sum (p : ℕ) (x y z : ℕ) 
  (hp : Prime p) 
  (hxyz : x > 0 ∧ y > 0 ∧ z > 0) 
  (heq : x^p + y^p = p^z) : 
  z = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2605_260528


namespace NUMINAMATH_CALUDE_fraction_simplification_l2605_260560

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 / (a * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2605_260560


namespace NUMINAMATH_CALUDE_total_wheels_is_150_l2605_260532

/-- The total number of wheels Naomi saw at the park -/
def total_wheels : ℕ :=
  let regular_bikes := 7
  let children_bikes := 11
  let tandem_bikes_4 := 5
  let tandem_bikes_6 := 3
  let unicycles := 4
  let tricycles := 6
  let training_wheel_bikes := 8

  let regular_bike_wheels := 2
  let children_bike_wheels := 4
  let tandem_bike_4_wheels := 4
  let tandem_bike_6_wheels := 6
  let unicycle_wheels := 1
  let tricycle_wheels := 3
  let training_wheel_bike_wheels := 4

  regular_bikes * regular_bike_wheels +
  children_bikes * children_bike_wheels +
  tandem_bikes_4 * tandem_bike_4_wheels +
  tandem_bikes_6 * tandem_bike_6_wheels +
  unicycles * unicycle_wheels +
  tricycles * tricycle_wheels +
  training_wheel_bikes * training_wheel_bike_wheels

theorem total_wheels_is_150 : total_wheels = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_150_l2605_260532


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_l2605_260583

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ 4 < x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | -5 ≤ x ∧ x < -2} := by sorry

-- Theorem for the intersection of A and the complement of B
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_B_l2605_260583


namespace NUMINAMATH_CALUDE_buddy_fraction_l2605_260539

theorem buddy_fraction (s₆ : ℕ) (n₉ : ℕ) : 
  s₆ > 0 ∧ n₉ > 0 →  -- Ensure positive numbers of students
  (n₉ : ℚ) / 4 = (s₆ : ℚ) / 3 →  -- 1/4 of ninth graders paired with 1/3 of sixth graders
  (s₆ : ℚ) / 3 / ((4 * s₆ : ℚ) / 3 + s₆) = 1 / 7 :=
by sorry

#check buddy_fraction

end NUMINAMATH_CALUDE_buddy_fraction_l2605_260539


namespace NUMINAMATH_CALUDE_apples_at_first_store_l2605_260521

def first_store_price : ℝ := 3
def second_store_price : ℝ := 4
def second_store_apples : ℝ := 10
def savings_per_apple : ℝ := 0.1

theorem apples_at_first_store :
  let second_store_price_per_apple := second_store_price / second_store_apples
  let first_store_price_per_apple := second_store_price_per_apple + savings_per_apple
  first_store_price / first_store_price_per_apple = 6 := by sorry

end NUMINAMATH_CALUDE_apples_at_first_store_l2605_260521


namespace NUMINAMATH_CALUDE_substitution_elimination_l2605_260580

/-- Given a system of linear equations in two variables x and y,
    prove that the equation obtained by eliminating y using substitution
    is equivalent to the given result. -/
theorem substitution_elimination (x y : ℝ) :
  (y = x + 3 ∧ 2 * x - y = 5) → (2 * x - x - 3 = 5) := by
  sorry

end NUMINAMATH_CALUDE_substitution_elimination_l2605_260580


namespace NUMINAMATH_CALUDE_no_solution_for_system_l2605_260508

theorem no_solution_for_system :
  ¬∃ (x y : ℝ), (2 * x - 3 * y = 7) ∧ (4 * x - 6 * y = 20) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l2605_260508


namespace NUMINAMATH_CALUDE_lisa_flight_distance_l2605_260584

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Lisa's flight distance -/
theorem lisa_flight_distance :
  let speed : ℝ := 32
  let time : ℝ := 8
  distance speed time = 256 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_distance_l2605_260584


namespace NUMINAMATH_CALUDE_mean_value_point_of_cubic_minus_linear_l2605_260503

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) := 3*x^2 - 3

-- Define the mean value point property
def is_mean_value_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b x₀ : ℝ) : Prop :=
  f b - f a = f' x₀ * (b - a)

theorem mean_value_point_of_cubic_minus_linear :
  ∃ x₀ : ℝ, is_mean_value_point f f' (-2) 2 x₀ ∧ x₀^2 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_mean_value_point_of_cubic_minus_linear_l2605_260503


namespace NUMINAMATH_CALUDE_total_time_is_8_days_l2605_260519

-- Define the problem parameters
def plow_rate : ℝ := 10  -- acres per day
def mow_rate : ℝ := 12   -- acres per day
def farmland_area : ℝ := 55  -- acres
def grassland_area : ℝ := 30  -- acres

-- Theorem statement
theorem total_time_is_8_days : 
  (farmland_area / plow_rate) + (grassland_area / mow_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_8_days_l2605_260519


namespace NUMINAMATH_CALUDE_peter_erasers_count_l2605_260544

def initial_erasers : ℕ := 8
def multiplier : ℕ := 3

theorem peter_erasers_count : 
  initial_erasers + multiplier * initial_erasers = 32 :=
by sorry

end NUMINAMATH_CALUDE_peter_erasers_count_l2605_260544


namespace NUMINAMATH_CALUDE_problem_solution_l2605_260574

theorem problem_solution : 
  (1 - 1^2022 - (3 * (2/3)^2 - 8/3 / (-2)^3) = -8/3) ∧ 
  (2^3 / 3 * (-1/4 + 7/12 - 5/6) / (-1/18) = 24) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2605_260574


namespace NUMINAMATH_CALUDE_green_peaches_count_l2605_260592

theorem green_peaches_count (red_peaches : ℕ) (green_peaches : ℕ) : 
  red_peaches = 5 → green_peaches = red_peaches + 6 → green_peaches = 11 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2605_260592


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2605_260522

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 10*x^2 + 15*x > 0 ↔ x ∈ Set.Ioo 0 (5 - Real.sqrt 10) ∪ Set.Ioi (5 + Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2605_260522


namespace NUMINAMATH_CALUDE_problem_statement_l2605_260501

theorem problem_statement (x y m n : ℤ) 
  (hxy : x > y) 
  (hmn : m > n) 
  (hsum_xy : x + y = 7) 
  (hprod_xy : x * y = 12) 
  (hsum_mn : m + n = 13) 
  (hsum_squares : m^2 + n^2 = 97) : 
  (x - y) - (m - n) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2605_260501


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2605_260523

theorem quadratic_root_relation (a c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ y = 3 * x ∧ a * x^2 + 6 * x + c = 0 ∧ a * y^2 + 6 * y + c = 0) →
  c = 27 / (4 * a) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2605_260523


namespace NUMINAMATH_CALUDE_sharon_in_middle_l2605_260553

-- Define the people
inductive Person : Type
| Aaron : Person
| Darren : Person
| Karen : Person
| Maren : Person
| Sharon : Person

-- Define the positions in the train
inductive Position : Type
| First : Position
| Second : Position
| Third : Position
| Fourth : Position
| Fifth : Position

def is_behind (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.Second, Position.Third => True
  | Position.Third, Position.Fourth => True
  | Position.Fourth, Position.Fifth => True
  | _, _ => False

def is_in_front (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.First, Position.Second => True
  | Position.First, Position.Third => True
  | Position.First, Position.Fourth => True
  | Position.Second, Position.Third => True
  | Position.Second, Position.Fourth => True
  | Position.Third, Position.Fourth => True
  | _, _ => False

def at_least_one_between (p1 p2 p3 : Position) : Prop :=
  match p1, p2, p3 with
  | Position.First, Position.Third, Position.Fifth => True
  | Position.First, Position.Fourth, Position.Fifth => True
  | Position.First, Position.Third, Position.Fourth => True
  | Position.Second, Position.Fourth, Position.Fifth => True
  | _, _, _ => False

-- Define the seating arrangement
def seating_arrangement (seat : Person → Position) : Prop :=
  (seat Person.Maren = Position.Fifth) ∧
  (∃ p : Position, is_behind (seat Person.Aaron) p ∧ seat Person.Sharon = p) ∧
  (∃ p : Position, is_in_front (seat Person.Darren) (seat Person.Aaron)) ∧
  (at_least_one_between (seat Person.Karen) (seat Person.Darren) (seat Person.Karen) ∨
   at_least_one_between (seat Person.Darren) (seat Person.Karen) (seat Person.Darren))

theorem sharon_in_middle (seat : Person → Position) :
  seating_arrangement seat → seat Person.Sharon = Position.Third :=
sorry

end NUMINAMATH_CALUDE_sharon_in_middle_l2605_260553


namespace NUMINAMATH_CALUDE_remainder_of_n_l2605_260594

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l2605_260594


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_2501_l2605_260591

theorem largest_prime_factor_of_2501 : ∃ p : ℕ, p.Prime ∧ p ∣ 2501 ∧ ∀ q : ℕ, q.Prime → q ∣ 2501 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_2501_l2605_260591


namespace NUMINAMATH_CALUDE_symmetric_point_about_origin_l2605_260516

/-- Given a point P(-1, 2) in a rectangular coordinate system,
    its symmetric point about the origin has coordinates (1, -2). -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (-1, 2)
  (- P.1, - P.2) = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_about_origin_l2605_260516


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2605_260527

-- Define the conditions
def p (a : ℝ) : Prop := (a - 1)^2 ≤ 1

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 ≥ 0

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2605_260527


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l2605_260556

-- Define the number of each type of fruit
def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 3

-- Define the total number of fruits
def total_fruits : ℕ := num_apples + num_oranges + num_bananas

-- Theorem statement
theorem fruit_arrangement_count : 
  (Nat.factorial total_fruits) / (Nat.factorial num_apples * Nat.factorial num_oranges * Nat.factorial num_bananas) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l2605_260556


namespace NUMINAMATH_CALUDE_triangle_six_nine_equals_eleven_l2605_260563

-- Define the ▽ operation
def triangle (m n : ℚ) (x y : ℚ) : ℚ := m^2 * x + n * y - 1

-- Theorem statement
theorem triangle_six_nine_equals_eleven 
  (m n : ℚ) 
  (h : triangle m n 2 3 = 3) : 
  triangle m n 6 9 = 11 := by
sorry

end NUMINAMATH_CALUDE_triangle_six_nine_equals_eleven_l2605_260563


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2605_260513

theorem shaded_area_percentage (total_squares : ℕ) (shaded_squares : ℕ) : 
  total_squares = 16 → shaded_squares = 3 → 
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2605_260513


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l2605_260582

theorem floor_plus_self_unique_solution : 
  ∃! r : ℝ, (⌊r⌋ : ℝ) + r = 18.75 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l2605_260582


namespace NUMINAMATH_CALUDE_final_sum_after_transformations_l2605_260546

theorem final_sum_after_transformations (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_transformations_l2605_260546


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l2605_260559

theorem binomial_coefficient_seven_three : 
  Nat.choose 7 3 = 35 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l2605_260559


namespace NUMINAMATH_CALUDE_inequality_condition_sum_l2605_260599

theorem inequality_condition_sum (a₁ a₂ : ℝ) : 
  (∀ x : ℝ, (x^2 - a₁*x + 2) / (x^2 - x + 1) < 3) ∧
  (∀ x : ℝ, (x^2 - a₂*x + 2) / (x^2 - x + 1) < 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, (x^2 - a*x + 2) / (x^2 - x + 1) < 3) → a > a₁ ∧ a < a₂) →
  a₁ = 3 - 2*Real.sqrt 2 ∧ a₂ = 3 + 2*Real.sqrt 2 ∧ a₁ + a₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_sum_l2605_260599


namespace NUMINAMATH_CALUDE_no_solution_equation_l2605_260538

theorem no_solution_equation :
  ∀ (x : ℝ), x^2 + x ≠ 0 ∧ x + 1 ≠ 0 →
  (5*x + 2) / (x^2 + x) ≠ 3 / (x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2605_260538


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l2605_260533

theorem geometric_sequence_minimum_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  a 6 = 3 →  -- a₆ = 3
  ∃ m : ℝ, m = 6 ∧ ∀ q, q > 0 → a 4 + a 8 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l2605_260533


namespace NUMINAMATH_CALUDE_fill_time_AB_is_2_4_hours_l2605_260525

-- Define the constants for the fill times
def fill_time_ABC : ℝ := 2
def fill_time_AC : ℝ := 3
def fill_time_BC : ℝ := 4

-- Define the rates of water flow for each valve
def rate_A : ℝ := sorry
def rate_B : ℝ := sorry
def rate_C : ℝ := sorry

-- Define the volume of the tank
def tank_volume : ℝ := sorry

-- Theorem to prove
theorem fill_time_AB_is_2_4_hours : 
  tank_volume / (rate_A + rate_B) = 2.4 := by sorry

end NUMINAMATH_CALUDE_fill_time_AB_is_2_4_hours_l2605_260525


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l2605_260514

theorem geometric_progression_solution :
  ∀ (b₁ q : ℚ),
    b₁ + b₁ * q + b₁ * q^2 = 21 →
    b₁^2 + (b₁ * q)^2 + (b₁ * q^2)^2 = 189 →
    ((b₁ = 12 ∧ q = 1/2) ∨ (b₁ = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l2605_260514


namespace NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l2605_260581

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  base_difference : longer_base = shorter_base + 50
  midpoint_segment : ℝ
  midpoint_area_ratio : midpoint_segment = shorter_base + 25
  equal_area_segment : ℝ
  area_ratio_condition : 
    2 * (height / 3 * (shorter_base + midpoint_segment)) = 
    (2 * height / 3) * (midpoint_segment + longer_base)

/-- The main theorem to be proved -/
theorem trapezoid_equal_area_segment (t : Trapezoid) : 
  Int.floor (t.equal_area_segment ^ 2 / 50) = 78 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l2605_260581


namespace NUMINAMATH_CALUDE_unanswered_questions_l2605_260596

/-- Represents the scoring system and results of a math contest --/
structure ContestScore where
  totalQuestions : ℕ
  oldScore : ℕ
  newScore : ℕ

/-- Proves that the number of unanswered questions is 10 given the contest conditions --/
theorem unanswered_questions (score : ContestScore)
  (h1 : score.totalQuestions = 40)
  (h2 : ∃ c w : ℕ, 25 + 3 * c - w = score.oldScore)
  (h3 : score.oldScore = 95)
  (h4 : ∃ c w u : ℕ, 6 * c - 2 * w + 3 * u = score.newScore)
  (h5 : score.newScore = 120)
  (h6 : ∃ c w u : ℕ, c + w + u = score.totalQuestions) :
  ∃ c w : ℕ, c + w + 10 = score.totalQuestions :=
sorry

end NUMINAMATH_CALUDE_unanswered_questions_l2605_260596


namespace NUMINAMATH_CALUDE_trig_identity_l2605_260517

theorem trig_identity : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2605_260517


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_360_l2605_260507

theorem smallest_k_sum_squares_multiple_360 : 
  ∃ k : ℕ+, (∀ m : ℕ+, m < k → ¬(∃ n : ℕ, m * (m + 1) * (2 * m + 1) = 6 * 360 * n)) ∧ 
  (∃ n : ℕ, k * (k + 1) * (2 * k + 1) = 6 * 360 * n) ∧ 
  k = 432 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_360_l2605_260507


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l2605_260565

/-- Proves that the discount percentage is 15% given the conditions of the dress pricing problem -/
theorem dress_discount_percentage : ∀ (original_price : ℝ) (discount_percentage : ℝ),
  original_price > 0 →
  discount_percentage > 0 →
  discount_percentage < 100 →
  original_price * (1 - discount_percentage / 100) = 68 →
  68 * 1.25 = original_price - 5 →
  discount_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l2605_260565


namespace NUMINAMATH_CALUDE_other_colors_correct_l2605_260530

/-- Represents a school with its student data -/
structure School where
  total_students : ℕ
  blue_percent : ℚ
  red_percent : ℚ
  green_percent : ℚ
  blue_red_percent : ℚ
  blue_green_percent : ℚ
  red_green_percent : ℚ

/-- Calculates the number of students wearing other colors -/
def other_colors (s : School) : ℕ :=
  s.total_students - (s.total_students * (s.blue_percent + s.red_percent + s.green_percent - 
    s.blue_red_percent - s.blue_green_percent - s.red_green_percent)).ceil.toNat

/-- The first school's data -/
def school1 : School := {
  total_students := 800,
  blue_percent := 30/100,
  red_percent := 20/100,
  green_percent := 10/100,
  blue_red_percent := 5/100,
  blue_green_percent := 3/100,
  red_green_percent := 2/100
}

/-- The second school's data -/
def school2 : School := {
  total_students := 700,
  blue_percent := 25/100,
  red_percent := 25/100,
  green_percent := 20/100,
  blue_red_percent := 10/100,
  blue_green_percent := 5/100,
  red_green_percent := 3/100
}

/-- The third school's data -/
def school3 : School := {
  total_students := 500,
  blue_percent := 1/100,
  red_percent := 1/100,
  green_percent := 1/100,
  blue_red_percent := 1/2/100,
  blue_green_percent := 1/2/100,
  red_green_percent := 1/2/100
}

/-- Theorem stating the correct number of students wearing other colors in each school -/
theorem other_colors_correct :
  other_colors school1 = 400 ∧
  other_colors school2 = 336 ∧
  other_colors school3 = 475 := by
  sorry

end NUMINAMATH_CALUDE_other_colors_correct_l2605_260530


namespace NUMINAMATH_CALUDE_blue_ball_weight_l2605_260597

theorem blue_ball_weight (brown_weight total_weight : ℝ) 
  (h1 : brown_weight = 3.12)
  (h2 : total_weight = 9.12) :
  total_weight - brown_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_weight_l2605_260597


namespace NUMINAMATH_CALUDE_fifty_third_number_is_71_l2605_260589

def sequenceValue (n : ℕ) : ℕ := 
  let fullSets := (n - 1) / 3
  let remainder := (n - 1) % 3
  1 + 4 * fullSets + remainder + (if remainder = 2 then 1 else 0)

theorem fifty_third_number_is_71 : sequenceValue 53 = 71 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_number_is_71_l2605_260589


namespace NUMINAMATH_CALUDE_power_division_equality_l2605_260515

theorem power_division_equality (a : ℝ) : a^11 / a^2 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2605_260515
