import Mathlib

namespace intersection_A_complement_B_l1355_135597

-- Define the set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

-- Define the set B
def B : Set ℝ := {x : ℝ | x^2 < 4}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x : ℝ | ¬ (x ∈ B)}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ complement_B = {x : ℝ | 2 ≤ x ∧ x < 6} := by sorry

end intersection_A_complement_B_l1355_135597


namespace linear_inequality_solution_set_l1355_135588

theorem linear_inequality_solution_set 
  (a b : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 1) : 
  {x : ℝ | a * x + b < 0} = {x : ℝ | x > 1} := by
sorry

end linear_inequality_solution_set_l1355_135588


namespace probability_units_digit_less_than_3_l1355_135590

/-- A five-digit even integer -/
def FiveDigitEven : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 ∧ n % 2 = 0 }

/-- The set of possible units digits for even numbers -/
def EvenUnitsDigits : Finset ℕ := {0, 2, 4, 6, 8}

/-- The set of units digits less than 3 -/
def UnitsDigitsLessThan3 : Finset ℕ := {0, 2}

/-- The probability of a randomly chosen five-digit even integer having a units digit less than 3 -/
theorem probability_units_digit_less_than_3 :
  (Finset.card UnitsDigitsLessThan3 : ℚ) / (Finset.card EvenUnitsDigits : ℚ) = 2 / 5 := by
  sorry

end probability_units_digit_less_than_3_l1355_135590


namespace bird_nest_babies_six_babies_in_nest_l1355_135580

/-- The number of babies in a bird's nest given the worm requirements and available worms. -/
theorem bird_nest_babies (worms_per_baby_per_day : ℕ) (papa_worms : ℕ) (mama_worms : ℕ) 
  (stolen_worms : ℕ) (additional_worms_needed : ℕ) (days : ℕ) : ℕ :=
  let total_worms := papa_worms + mama_worms - stolen_worms + additional_worms_needed
  let worms_per_baby := worms_per_baby_per_day * days
  total_worms / worms_per_baby

/-- There are 6 babies in the nest given the specific conditions. -/
theorem six_babies_in_nest : 
  bird_nest_babies 3 9 13 2 34 3 = 6 := by
  sorry

end bird_nest_babies_six_babies_in_nest_l1355_135580


namespace days_until_birthday_l1355_135521

/-- Proof of the number of days until Maria's birthday --/
theorem days_until_birthday (daily_savings : ℕ) (flower_cost : ℕ) (flowers_bought : ℕ) :
  daily_savings = 2 →
  flower_cost = 4 →
  flowers_bought = 11 →
  (flowers_bought * flower_cost) / daily_savings = 22 :=
by sorry

end days_until_birthday_l1355_135521


namespace smallest_integer_ending_in_3_divisible_by_11_l1355_135530

theorem smallest_integer_ending_in_3_divisible_by_11 : ∃ n : ℕ, 
  (n % 10 = 3) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, m < n → m % 10 = 3 → m % 11 ≠ 0) :=
by
  -- The proof goes here
  sorry

end smallest_integer_ending_in_3_divisible_by_11_l1355_135530


namespace min_sum_squares_l1355_135545

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 8 :=
sorry

end min_sum_squares_l1355_135545


namespace largest_integer_solution_l1355_135514

theorem largest_integer_solution : 
  ∃ (x : ℕ), (1/4 : ℚ) + (x/5 : ℚ) < 2 ∧ 
  ∀ (y : ℕ), y > x → (1/4 : ℚ) + (y/5 : ℚ) ≥ 2 :=
by
  use 23
  sorry

end largest_integer_solution_l1355_135514


namespace trigonometric_identity_l1355_135577

theorem trigonometric_identity (α : ℝ) :
  -Real.cos (5 * α) * Real.cos (4 * α) - Real.cos (4 * α) * Real.cos (3 * α) + 2 * (Real.cos (2 * α))^2 * Real.cos α
  = 2 * Real.cos α * Real.sin (2 * α) * Real.sin (6 * α) := by
  sorry

end trigonometric_identity_l1355_135577


namespace proposition_validity_l1355_135598

theorem proposition_validity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 - b^2 = 1) → (a - b < 1)) ∧
   ¬((1/b - 1/a = 1) → (a - b < 1)) ∧
   ((Real.exp a - Real.exp b = 1) → (a - b < 1)) ∧
   ¬((Real.log a - Real.log b = 1) → (a - b < 1))) := by
sorry

end proposition_validity_l1355_135598


namespace shirts_not_washed_l1355_135585

theorem shirts_not_washed 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) : 
  short_sleeve + long_sleeve - washed = 1 := by
sorry

end shirts_not_washed_l1355_135585


namespace linear_quadratic_intersection_l1355_135562

-- Define the functions f and g
def f (k b x : ℝ) : ℝ := k * x + b
def g (x : ℝ) : ℝ := x^2 - x - 6

-- State the theorem
theorem linear_quadratic_intersection (k b : ℝ) :
  (∃ A B : ℝ × ℝ, 
    f k b A.1 = 0 ∧ 
    f k b 0 = B.2 ∧ 
    B.1 - A.1 = 2 ∧ 
    B.2 - A.2 = 2) →
  (k = 1 ∧ b = 2) ∧
  (∀ x : ℝ, f k b x > g x → (g x + 1) / (f k b x) ≥ -3) ∧
  (∃ x : ℝ, f k b x > g x ∧ (g x + 1) / (f k b x) = -3) :=
by sorry

end linear_quadratic_intersection_l1355_135562


namespace hyperbola_equation_l1355_135517

/-- A hyperbola with specific properties -/
structure Hyperbola where
  conjugate_axis_length : ℝ
  eccentricity : ℝ
  focal_length : ℝ
  point_m : ℝ × ℝ
  point_p : ℝ × ℝ
  point_q : ℝ × ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / 25 - x^2 / 75 = 1

/-- Theorem stating the standard equation of the specific hyperbola -/
theorem hyperbola_equation (h : Hyperbola)
  (h_conjugate : h.conjugate_axis_length = 12)
  (h_eccentricity : h.eccentricity = 5/4)
  (h_focal : h.focal_length = 26)
  (h_point_m : h.point_m = (0, 12))
  (h_point_p : h.point_p = (-3, 2 * Real.sqrt 7))
  (h_point_q : h.point_q = (-6 * Real.sqrt 2, -7)) :
  ∀ x y, standard_equation h x y ↔ 
    (x = h.point_m.1 ∧ y = h.point_m.2) ∨
    (x = h.point_p.1 ∧ y = h.point_p.2) ∨
    (x = h.point_q.1 ∧ y = h.point_q.2) :=
by sorry

end hyperbola_equation_l1355_135517


namespace prob_exact_tails_l1355_135571

def coin_flips : ℕ := 8
def p_tails : ℚ := 4/5
def p_heads : ℚ := 1/5
def exact_tails : ℕ := 3

theorem prob_exact_tails :
  (Nat.choose coin_flips exact_tails : ℚ) * p_tails ^ exact_tails * p_heads ^ (coin_flips - exact_tails) = 3584/390625 := by
  sorry

end prob_exact_tails_l1355_135571


namespace find_x1_l1355_135541

theorem find_x1 (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + 2*(x1 - x2)^2 + 2*(x2 - x3)^2 + x3^2 = 1/2) :
  x1 = 2/3 := by sorry

end find_x1_l1355_135541


namespace A_equiv_B_l1355_135515

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define set A
def A : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧
  (∃ k : ℤ, (sumOfDigits n + 1 = 5 * k ∨ sumOfDigits n - 1 = 5 * k))}

-- Define set B
def B : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧
  (∃ k : ℤ, sumOfDigits n = 5 * k ∨ sumOfDigits n - 2 = 5 * k)}

-- Theorem statement
theorem A_equiv_B : Equiv A B := by sorry

end A_equiv_B_l1355_135515


namespace triangle_similarity_after_bisections_l1355_135537

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle bisector construction process -/
def AngleBisectorProcess (T : Triangle) (n : ℕ) : Triangle :=
  sorry

/-- Similarity ratio between two triangles -/
def SimilarityRatio (T1 T2 : Triangle) : ℝ :=
  sorry

theorem triangle_similarity_after_bisections (T : Triangle) (h1 : T.a = 5) (h2 : T.b = 6) (h3 : T.c = 4) :
  let T_final := AngleBisectorProcess T 2021
  SimilarityRatio T T_final = (4/9)^2021 := by
  sorry

end triangle_similarity_after_bisections_l1355_135537


namespace symmetric_line_proof_l1355_135573

/-- The fixed point M through which all lines ax+y+3a-1=0 pass -/
def M : ℝ × ℝ := (-3, 1)

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2*x + 3*y - 6 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2*x + 3*y + 12 = 0

/-- The family of lines passing through M -/
def family_line (a x y : ℝ) : Prop := a*x + y + 3*a - 1 = 0

theorem symmetric_line_proof :
  ∀ (a : ℝ), family_line a M.1 M.2 →
  ∀ (x y : ℝ), symmetric_line x y ↔ 
    (x - M.1 = M.1 - x' ∧ y - M.2 = M.2 - y' ∧ original_line x' y') :=
by sorry

end symmetric_line_proof_l1355_135573


namespace negation_p_necessary_not_sufficient_l1355_135504

theorem negation_p_necessary_not_sufficient (p q : Prop) :
  (¬(¬p → ¬(p ∨ q))) ∧ (∃ (p q : Prop), ¬p ∧ (p ∨ q)) := by sorry

end negation_p_necessary_not_sufficient_l1355_135504


namespace triangle_angle_f_l1355_135592

theorem triangle_angle_f (D E F : Real) : 
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = Real.pi →
  5 * Real.sin D + 2 * Real.cos E = 8 →
  3 * Real.sin E + 5 * Real.cos D = 2 →
  Real.sin F = 43 / 50 := by
  sorry

end triangle_angle_f_l1355_135592


namespace concentric_circles_radius_l1355_135544

theorem concentric_circles_radius (r R : ℝ) (h1 : r = 4) 
  (h2 : (1.5 * R)^2 - (0.75 * r)^2 = 3.6 * (R^2 - r^2)) : R = 6 := by
  sorry

end concentric_circles_radius_l1355_135544


namespace no_solution_for_specific_p_range_l1355_135593

theorem no_solution_for_specific_p_range (p : ℝ) (h : 4/3 < p ∧ p < 2) :
  ¬∃ x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x :=
by sorry

end no_solution_for_specific_p_range_l1355_135593


namespace A_divisible_by_1980_l1355_135559

def A : ℕ := sorry  -- Definition of A as the concatenated number

-- Theorem statement
theorem A_divisible_by_1980 : 1980 ∣ A :=
by
  -- Proof goes here
  sorry

end A_divisible_by_1980_l1355_135559


namespace adam_magnets_l1355_135526

theorem adam_magnets (peter_magnets : ℕ) (adam_remaining : ℕ) (adam_initial : ℕ) : 
  peter_magnets = 24 →
  adam_remaining = peter_magnets / 2 →
  adam_remaining = adam_initial * 2 / 3 →
  adam_initial = 18 := by
sorry

end adam_magnets_l1355_135526


namespace bakery_children_count_l1355_135512

theorem bakery_children_count (initial_count : ℕ) (girls_entered : ℕ) (boys_left : ℕ) 
  (h1 : initial_count = 85) (h2 : girls_entered = 24) (h3 : boys_left = 31) :
  initial_count + girls_entered - boys_left = 78 :=
by
  sorry

end bakery_children_count_l1355_135512


namespace turkey_roasting_problem_l1355_135552

/-- Represents the turkey roasting problem --/
structure TurkeyRoasting where
  turkeyWeight : ℕ
  roastingTimePerPound : ℕ
  roastingStartTime : ℕ
  dinnerTime : ℕ

/-- Calculates the maximum number of turkeys that can be roasted --/
def maxTurkeys (tr : TurkeyRoasting) : ℕ :=
  let totalRoastingTime := tr.dinnerTime - tr.roastingStartTime
  let roastingTimePerTurkey := tr.turkeyWeight * tr.roastingTimePerPound
  totalRoastingTime / roastingTimePerTurkey

/-- Theorem statement for the turkey roasting problem --/
theorem turkey_roasting_problem :
  let tr : TurkeyRoasting := {
    turkeyWeight := 16,
    roastingTimePerPound := 15,
    roastingStartTime := 10 * 60,  -- 10:00 am in minutes
    dinnerTime := 18 * 60  -- 6:00 pm in minutes
  }
  maxTurkeys tr = 2 := by
  sorry


end turkey_roasting_problem_l1355_135552


namespace tan_equality_implies_x_120_l1355_135539

theorem tan_equality_implies_x_120 (x : Real) :
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 120 := by
  sorry

end tan_equality_implies_x_120_l1355_135539


namespace no_right_triangle_with_sqrt_2016_side_l1355_135501

theorem no_right_triangle_with_sqrt_2016_side : ¬ ∃ (a b : ℕ) (c : ℝ), 
  c = Real.sqrt 2016 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ c * c + b * b = a * a) :=
sorry

end no_right_triangle_with_sqrt_2016_side_l1355_135501


namespace solve_for_a_l1355_135569

theorem solve_for_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 8) 
  (eq3 : c = 4) : 
  a = 0 := by
sorry

end solve_for_a_l1355_135569


namespace least_five_digit_divisible_by_15_12_18_l1355_135532

theorem least_five_digit_divisible_by_15_12_18 :
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    n % 15 = 0 ∧ 
    n % 12 = 0 ∧ 
    n % 18 = 0 ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < n ∧ m % 15 = 0 ∧ m % 12 = 0 ∧ m % 18 = 0 → false) ∧
    n = 10080 :=
by sorry

end least_five_digit_divisible_by_15_12_18_l1355_135532


namespace series_sum_inequality_l1355_135572

theorem series_sum_inequality (S : ℝ) (h : S = 2^(1/4)) : 
  ∃ n : ℕ, 2^n < S^2007 ∧ S^2007 < 2^(n+1) ∧ n = 501 := by
  sorry

end series_sum_inequality_l1355_135572


namespace cylinder_surface_area_l1355_135528

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area :
  let h : ℝ := 8  -- height in inches
  let r : ℝ := 3  -- radius in inches
  let lateral_area : ℝ := 2 * π * r * h
  let base_area : ℝ := π * r^2
  let total_surface_area : ℝ := lateral_area + 2 * base_area
  total_surface_area = 66 * π :=
by sorry

end cylinder_surface_area_l1355_135528


namespace four_male_workers_selected_l1355_135563

/-- Represents the number of male workers selected in a stratified sampling -/
def male_workers_selected (total_workers female_workers selected_workers : ℕ) : ℕ :=
  (total_workers - female_workers) * selected_workers / total_workers

/-- Theorem stating that 4 male workers are selected in the given scenario -/
theorem four_male_workers_selected :
  male_workers_selected 30 10 6 = 4 := by
  sorry

#eval male_workers_selected 30 10 6

end four_male_workers_selected_l1355_135563


namespace extreme_value_point_l1355_135506

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem stating that -2 is an extreme value point of f
theorem extreme_value_point : 
  ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ x₁ ∈ Set.Ioo (-2 - δ) (-2), f' x₁ < 0) ∧
  (∀ x₂ ∈ Set.Ioo (-2) (-2 + δ), f' x₂ > 0) :=
sorry

end extreme_value_point_l1355_135506


namespace divisors_of_18n_cubed_l1355_135586

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_18n_cubed (n : ℕ) 
  (h_odd : Odd n) 
  (h_divisors : num_divisors n = 13) : 
  num_divisors (18 * n^3) = 222 := by sorry

end divisors_of_18n_cubed_l1355_135586


namespace total_rounded_to_nearest_dollar_l1355_135568

-- Define the rounding function
def roundToNearestDollar (x : ℚ) : ℤ :=
  if x - x.floor < 1/2 then x.floor else x.ceil

-- Define the purchases
def purchase1 : ℚ := 299/100
def purchase2 : ℚ := 651/100
def purchase3 : ℚ := 1049/100

-- Theorem statement
theorem total_rounded_to_nearest_dollar :
  (roundToNearestDollar purchase1 + 
   roundToNearestDollar purchase2 + 
   roundToNearestDollar purchase3) = 20 := by
  sorry

end total_rounded_to_nearest_dollar_l1355_135568


namespace rectangle_side_relationship_l1355_135519

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.x + r.y)

/-- Theorem: For a rectangle with perimeter 50 cm, y = 25 - x -/
theorem rectangle_side_relationship (r : Rectangle) 
  (h : perimeter r = 50) : r.y = 25 - r.x := by
  sorry

end rectangle_side_relationship_l1355_135519


namespace condition_one_condition_two_condition_three_l1355_135584

-- Define set A
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x = -1/(2*a)}

-- Theorem for condition ①
theorem condition_one : 
  ∀ a : ℝ, (A ∩ B a = B a) ↔ (a = 0 ∨ a = -1/2 ∨ a = 1/6) := by sorry

-- Theorem for condition ②
theorem condition_two :
  ∀ a : ℝ, ((Set.univ \ B a) ∩ A = {1}) ↔ (a = 1/6) := by sorry

-- Theorem for condition ③
theorem condition_three :
  ∀ a : ℝ, (A ∩ B a = ∅) ↔ (a ≠ 1/6 ∧ a ≠ -1/2) := by sorry

end condition_one_condition_two_condition_three_l1355_135584


namespace particular_number_solution_l1355_135589

theorem particular_number_solution (A B : ℤ) (h1 : A = 14) (h2 : B = 24) :
  ∃ x : ℚ, ((A + x) * A - B) / B = 13 ∧ x = 10 := by
  sorry

end particular_number_solution_l1355_135589


namespace unique_two_digit_ratio_l1355_135560

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem unique_two_digit_ratio :
  ∃! n : ℕ, is_two_digit n ∧ (n : ℚ) / (reverse_digits n : ℚ) = 7 / 4 :=
by
  use 21
  sorry

end unique_two_digit_ratio_l1355_135560


namespace mady_balls_theorem_l1355_135581

def to_nonary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_nonary_aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_nonary_aux (m / 9) ((m % 9) :: acc)
    to_nonary_aux n []

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.sum

theorem mady_balls_theorem (step : ℕ) (h : step = 2500) :
  sum_of_digits (to_nonary step) = 20 :=
sorry

end mady_balls_theorem_l1355_135581


namespace box_of_balls_theorem_l1355_135550

theorem box_of_balls_theorem :
  ∃ (B X Y : ℝ),
    40 < X ∧ X < 50 ∧
    60 < Y ∧ Y < 70 ∧
    B - X = Y - B ∧
    B = 55 := by sorry

end box_of_balls_theorem_l1355_135550


namespace chicken_nugget_ratio_l1355_135503

theorem chicken_nugget_ratio : 
  ∀ (keely kendall : ℕ),
  keely + kendall + 20 = 100 →
  (keely + kendall) / 20 = 4 := by
sorry

end chicken_nugget_ratio_l1355_135503


namespace max_sum_geometric_sequence_l1355_135508

/-- Given integers a, b, and c forming a strictly increasing geometric sequence with abc = 216,
    the maximum value of a + b + c is 43. -/
theorem max_sum_geometric_sequence (a b c : ℤ) : 
  a < b ∧ b < c ∧                 -- strictly increasing
  (∃ r : ℤ, r > 1 ∧ b = a * r ∧ c = b * r) ∧  -- geometric sequence
  a * b * c = 216 →               -- product condition
  (∀ x y z : ℤ, 
    x < y ∧ y < z ∧
    (∃ r : ℤ, r > 1 ∧ y = x * r ∧ z = y * r) ∧
    x * y * z = 216 →
    x + y + z ≤ a + b + c) ∧
  a + b + c = 43 := by
sorry

end max_sum_geometric_sequence_l1355_135508


namespace solution_water_amount_l1355_135564

/-- Given a solution with an original ratio of bleach : detergent : water as 2 : 40 : 100,
    when the ratio of bleach to detergent is tripled and the ratio of detergent to water is halved,
    and the new solution contains 60 liters of detergent, prove that the amount of water
    in the new solution is 75 liters. -/
theorem solution_water_amount
  (original_ratio : Fin 3 → ℚ)
  (h_original : original_ratio = ![2, 40, 100])
  (new_detergent : ℚ)
  (h_new_detergent : new_detergent = 60)
  : ∃ (new_ratio : Fin 3 → ℚ) (water : ℚ),
    (new_ratio 0 / new_ratio 1 = 3 * (original_ratio 0 / original_ratio 1)) ∧
    (new_ratio 1 / new_ratio 2 = (original_ratio 1 / original_ratio 2) / 2) ∧
    (new_ratio 1 = new_detergent) ∧
    (water = 75) := by
  sorry

end solution_water_amount_l1355_135564


namespace maddie_monday_viewing_l1355_135555

/-- The number of minutes Maddie watched TV on Monday -/
def monday_minutes (total_episodes : ℕ) (episode_length : ℕ) (thursday_minutes : ℕ) (friday_episodes : ℕ) (weekend_minutes : ℕ) : ℕ :=
  total_episodes * episode_length - (thursday_minutes + friday_episodes * episode_length + weekend_minutes)

theorem maddie_monday_viewing : 
  monday_minutes 8 44 21 2 105 = 138 := by
  sorry

end maddie_monday_viewing_l1355_135555


namespace compare_negative_fractions_l1355_135599

theorem compare_negative_fractions : -4/5 > -5/6 := by sorry

end compare_negative_fractions_l1355_135599


namespace simplified_inverse_sum_l1355_135565

theorem simplified_inverse_sum (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  (a * x⁻¹ + b * y⁻¹)⁻¹ = (x * y) / (a * y + b * x) := by
  sorry

end simplified_inverse_sum_l1355_135565


namespace interview_probability_correct_l1355_135567

structure TouristGroup where
  total : ℕ
  outside_fraction : ℚ
  inside_fraction : ℚ
  gold_fraction : ℚ
  silver_fraction : ℚ

def interview_probability (group : TouristGroup) : ℚ × ℚ :=
  let outside := (group.total : ℚ) * group.outside_fraction
  let inside := (group.total : ℚ) * group.inside_fraction
  let gold := outside * group.gold_fraction
  let silver := inside * group.silver_fraction
  let no_card := group.total - (gold + silver)
  let prob_one_silver := (silver * (group.total - silver)) / ((group.total * (group.total - 1)) / 2)
  let prob_equal := (((no_card * (no_card - 1)) / 2) + gold * silver) / ((group.total * (group.total - 1)) / 2)
  (prob_one_silver, prob_equal)

theorem interview_probability_correct (group : TouristGroup) 
  (h1 : group.total = 36)
  (h2 : group.outside_fraction = 3/4)
  (h3 : group.inside_fraction = 1/4)
  (h4 : group.gold_fraction = 1/3)
  (h5 : group.silver_fraction = 2/3) :
  interview_probability group = (2/7, 44/105) := by
  sorry

end interview_probability_correct_l1355_135567


namespace sum_of_digits_of_big_number_l1355_135583

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The number we're interested in -/
def big_number : ℕ := 10^95 - 97

/-- The theorem stating that the sum of digits of our big number is 840 -/
theorem sum_of_digits_of_big_number : sum_of_digits big_number = 840 := by sorry

end sum_of_digits_of_big_number_l1355_135583


namespace arithmetic_sequence_sum_and_remainder_l1355_135525

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_and_remainder :
  let sum := arithmetic_sequence_sum 3 5 103
  sum = 1113 ∧ sum % 10 = 3 := by sorry

end arithmetic_sequence_sum_and_remainder_l1355_135525


namespace parabola_equation_l1355_135558

/-- Given a parabola y^2 = 2px where p > 0, if a line with slope 1 passing through
    the focus intersects the parabola at points A and B such that |AB| = 8,
    then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2*p*x → (∃ t, y = t ∧ x = t + p/2)) →  -- Line passing through focus
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →                -- A and B on parabola
  ‖A - B‖ = 8 →                                        -- |AB| = 8
  ∀ x y, y^2 = 2*p*x ↔ y^2 = 4*x := by
  sorry

end parabola_equation_l1355_135558


namespace ellipse_property_l1355_135500

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Define the angle between PF1 and PF2
def angle_F1PF2 (P : ℝ × ℝ) : ℝ := 120

-- Theorem statement
theorem ellipse_property (P : ℝ × ℝ) 
  (h1 : is_on_ellipse P.1 P.2) 
  (h2 : angle_F1PF2 P = 120) : 
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * 
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 36 := by
  sorry

end ellipse_property_l1355_135500


namespace small_triangles_to_cover_large_l1355_135505

/-- The number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem small_triangles_to_cover_large (large_side small_side : ℝ) : 
  large_side = 12 → small_side = 2 → 
  (large_side^2 / small_side^2 : ℝ) = 36 := by
  sorry

end small_triangles_to_cover_large_l1355_135505


namespace intersection_of_three_lines_l1355_135579

/-- Given three lines that pass through the same point, prove the value of k -/
theorem intersection_of_three_lines (t : ℝ) (h_t : t = 6) :
  ∃ (x y : ℝ), (x + t * y + 8 = 0 ∧ 5 * x - t * y + 4 = 0 ∧ 3 * x - 5 * y + 1 = 0) →
  ∀ k : ℝ, (x + t * y + 8 = 0 ∧ 5 * x - t * y + 4 = 0 ∧ 3 * x - k * y + 1 = 0) →
  k = 5 :=
by
  sorry

end intersection_of_three_lines_l1355_135579


namespace sum_of_coefficients_is_27_l1355_135549

-- Define the polynomial
def p (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 4) + 6 * (x^6 + 9 * x^3 - 8)

-- Theorem statement
theorem sum_of_coefficients_is_27 : 
  p 1 = 27 := by sorry

end sum_of_coefficients_is_27_l1355_135549


namespace polynomial_expansion_l1355_135507

theorem polynomial_expansion :
  ∀ z : ℂ, (3 * z^3 + 2 * z^2 - 4 * z + 1) * (2 * z^4 - 3 * z^2 + z - 5) =
  6 * z^7 + 4 * z^6 - 4 * z^5 - 9 * z^3 + 7 * z^2 + z - 5 := by
  sorry

end polynomial_expansion_l1355_135507


namespace solve_for_k_l1355_135540

theorem solve_for_k (x k : ℝ) : x + k - 4 = 0 → x = 2 → k = 2 := by
  sorry

end solve_for_k_l1355_135540


namespace no_central_ring_numbers_l1355_135587

/-- Definition of a central ring number -/
def is_central_ring_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (n % 11 ≠ 0) ∧              -- not divisible by 11
  ((n / 1000) % 11 = 0) ∧     -- removing thousands digit
  ((n % 1000 + (n / 10000) * 100) % 11 = 0) ∧  -- removing hundreds digit
  ((n / 100 * 10 + n % 10) % 11 = 0) ∧         -- removing tens digit
  ((n / 10) % 11 = 0)         -- removing ones digit

/-- Theorem: There are no central ring numbers -/
theorem no_central_ring_numbers : ¬∃ n, is_central_ring_number n := by
  sorry

end no_central_ring_numbers_l1355_135587


namespace stating_no_equal_area_division_for_n_gt_2_l1355_135594

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an angle bisector in a triangle -/
structure AngleBisector where
  origin : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- 
  Given a triangle and a set of angle bisectors from one vertex, 
  checks if they divide the triangle into n equal-area parts
-/
def divideIntoEqualAreas (t : Triangle) (bisectors : List AngleBisector) (n : ℕ) : Prop :=
  sorry

/-- 
  Theorem stating that for all triangles and integers n > 2, 
  it is impossible for the angle bisectors of one of the triangle's vertices 
  to divide the triangle into n equal-area parts
-/
theorem no_equal_area_division_for_n_gt_2 :
  ∀ (t : Triangle) (n : ℕ), n > 2 → ¬∃ (bisectors : List AngleBisector), 
  divideIntoEqualAreas t bisectors n :=
sorry

end stating_no_equal_area_division_for_n_gt_2_l1355_135594


namespace optimal_circular_sector_radius_l1355_135548

/-- The radius that maximizes the area of a circular sector with given constraints -/
theorem optimal_circular_sector_radius : 
  ∀ (r : ℝ) (s : ℝ),
  -- Total perimeter is 32 meters
  2 * r + s = 32 →
  -- Ratio of radius to arc length is at least 2:3
  r / s ≥ 2 / 3 →
  -- Area of the sector is maximized
  ∀ (r' : ℝ) (s' : ℝ),
  2 * r' + s' = 32 →
  r' / s' ≥ 2 / 3 →
  r * s ≥ r' * s' →
  -- The optimal radius is 64/7
  r = 64 / 7 :=
by sorry

end optimal_circular_sector_radius_l1355_135548


namespace words_per_page_l1355_135524

/-- Calculates the number of words per page in books Sarah is reading --/
theorem words_per_page
  (reading_speed : ℕ)  -- Sarah's reading speed in words per minute
  (reading_time : ℕ)   -- Total reading time in hours
  (num_books : ℕ)      -- Number of books Sarah plans to read
  (pages_per_book : ℕ) -- Number of pages in each book
  (h1 : reading_speed = 40)
  (h2 : reading_time = 20)
  (h3 : num_books = 6)
  (h4 : pages_per_book = 80)
  : (reading_speed * 60 * reading_time) / (num_books * pages_per_book) = 100 := by
  sorry

end words_per_page_l1355_135524


namespace q_div_p_equals_162_l1355_135533

/-- The number of slips in the hat -/
def total_slips : ℕ := 40

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The number of slips with each number -/
def slips_per_number : ℕ := 4

/-- The probability that all four drawn slips bear the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability that two slips bear a number a and the other two bear a number b ≠ a -/
def q : ℚ := ((Nat.choose distinct_numbers 2 : ℚ) * 
              (Nat.choose slips_per_number 2 : ℚ) * 
              (Nat.choose slips_per_number 2 : ℚ)) / 
             (Nat.choose total_slips drawn_slips : ℚ)

/-- Theorem stating that q/p = 162 -/
theorem q_div_p_equals_162 : q / p = 162 := by sorry

end q_div_p_equals_162_l1355_135533


namespace derek_savings_and_expenses_l1355_135556

theorem derek_savings_and_expenses :
  let geometric_sum := (2 : ℝ) * (1 - 2^12) / (1 - 2)
  let arithmetic_sum := 12 / 2 * (2 * 3 + (12 - 1) * 2)
  geometric_sum - arithmetic_sum = 8022 := by
  sorry

end derek_savings_and_expenses_l1355_135556


namespace total_weight_loss_is_correct_l1355_135513

/-- The total weight loss of Seth, Jerome, Veronica, and Maya -/
def totalWeightLoss (sethLoss : ℝ) : ℝ :=
  let jeromeLoss := 3 * sethLoss
  let veronicaLoss := sethLoss + 1.56
  let sethVeronicaCombined := sethLoss + veronicaLoss
  let mayaLoss := sethVeronicaCombined * 0.75
  sethLoss + jeromeLoss + veronicaLoss + mayaLoss

/-- Theorem stating that the total weight loss is 116.675 pounds -/
theorem total_weight_loss_is_correct :
  totalWeightLoss 17.53 = 116.675 := by
  sorry

#eval totalWeightLoss 17.53

end total_weight_loss_is_correct_l1355_135513


namespace factor_implies_d_value_l1355_135578

theorem factor_implies_d_value (d : ℚ) :
  (∀ x : ℚ, (x - 4) ∣ (d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72)) →
  d = -83/42 := by
sorry

end factor_implies_d_value_l1355_135578


namespace basketball_not_football_l1355_135509

theorem basketball_not_football (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : basketball = 15)
  (h3 : football = 8)
  (h4 : neither = 8) :
  ∃ (x : ℕ), x = basketball - (basketball + football - total + neither) ∧ x = 14 :=
by sorry

end basketball_not_football_l1355_135509


namespace car_lot_problem_l1355_135570

theorem car_lot_problem (total : ℕ) (power_steering : ℕ) (power_windows : ℕ) (neither : ℕ) :
  total = 65 →
  power_steering = 45 →
  power_windows = 25 →
  neither = 12 →
  ∃ both : ℕ, both = 17 ∧
    total = power_steering + power_windows - both + neither :=
by sorry

end car_lot_problem_l1355_135570


namespace proportion_solution_l1355_135518

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 2) → x = 0.20 := by
  sorry

end proportion_solution_l1355_135518


namespace circle_radius_from_chord_and_central_angle_l1355_135529

theorem circle_radius_from_chord_and_central_angle (α : ℝ) (h : α > 0 ∧ α < 360) :
  let chord_length : ℝ := 10
  let radius : ℝ := 5 / Real.sin (α * π / 360)
  2 * radius * Real.sin (α * π / 360) = chord_length := by sorry

end circle_radius_from_chord_and_central_angle_l1355_135529


namespace zongzi_purchase_theorem_l1355_135543

/-- Represents the properties of zongzi purchases in a supermarket. -/
structure ZongziPurchase where
  price_a : ℝ  -- Unit price of type A zongzi
  price_b : ℝ  -- Unit price of type B zongzi
  quantity_a : ℝ  -- Quantity of type A zongzi
  quantity_b : ℝ  -- Quantity of type B zongzi

/-- Theorem stating the properties of the zongzi purchase and the maximum purchase of type A zongzi. -/
theorem zongzi_purchase_theorem (z : ZongziPurchase) : 
  z.price_a * z.quantity_a = 1200 ∧ 
  z.price_b * z.quantity_b = 800 ∧ 
  z.quantity_b = z.quantity_a + 50 ∧ 
  z.price_a = 2 * z.price_b → 
  z.price_a = 8 ∧ z.price_b = 4 ∧ 
  (∀ m : ℕ, m ≤ 87 ↔ (m : ℝ) * 8 + (200 - m) * 4 ≤ 1150) :=
by sorry

#check zongzi_purchase_theorem

end zongzi_purchase_theorem_l1355_135543


namespace min_value_f_l1355_135510

theorem min_value_f (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

end min_value_f_l1355_135510


namespace employee_pay_calculation_l1355_135534

theorem employee_pay_calculation (total_pay : ℝ) (percentage : ℝ) (y : ℝ) :
  total_pay = 880 →
  percentage = 120 →
  total_pay = y + (percentage / 100) * y →
  y = 400 := by
sorry

end employee_pay_calculation_l1355_135534


namespace solution_set_correct_l1355_135582

/-- The set of solutions to the system of equations:
    x^2 + y^2 + 8x - 6y = -20
    x^2 + z^2 + 8x + 4z = -10
    y^2 + z^2 - 6y + 4z = 0
-/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(-3, 1, 1), (-3, 1, -5), (-3, 5, 1), (-3, 5, -5),
   (-5, 1, 1), (-5, 1, -5), (-5, 5, 1), (-5, 5, -5)}

/-- The system of equations -/
def SystemEquations (x y z : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 6*y = -20 ∧
  x^2 + z^2 + 8*x + 4*z = -10 ∧
  y^2 + z^2 - 6*y + 4*z = 0

/-- Theorem stating that the SolutionSet contains exactly all solutions to the SystemEquations -/
theorem solution_set_correct :
  ∀ x y z, (x, y, z) ∈ SolutionSet ↔ SystemEquations x y z :=
sorry

end solution_set_correct_l1355_135582


namespace part_one_part_two_l1355_135538

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def q (x m : ℝ) : Prop := 1 - m^2 ≤ x ∧ x ≤ 1 + m^2

-- Part I: p is a necessary condition for q
theorem part_one (m : ℝ) : 
  (∀ x, q x m → p x) → -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

-- Part II: ¬p is a necessary but not sufficient condition for ¬q
theorem part_two (m : ℝ) :
  ((∀ x, ¬q x m → ¬p x) ∧ (∃ x, ¬p x ∧ q x m)) → m ≥ 3 ∨ m ≤ -3 :=
sorry

end part_one_part_two_l1355_135538


namespace power_of_81_l1355_135535

theorem power_of_81 : (81 : ℝ) ^ (5/4) = 243 := by
  sorry

end power_of_81_l1355_135535


namespace pyarelal_loss_calculation_l1355_135574

/-- Calculates Pyarelal's share of the loss given the total loss and the ratio of investments -/
def pyarelal_loss (total_loss : ℚ) (ashok_ratio : ℚ) (pyarelal_ratio : ℚ) : ℚ :=
  (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss

/-- Proves that Pyarelal's loss is 1080 given the conditions of the problem -/
theorem pyarelal_loss_calculation :
  let total_loss : ℚ := 1200
  let ashok_ratio : ℚ := 1
  let pyarelal_ratio : ℚ := 9
  pyarelal_loss total_loss ashok_ratio pyarelal_ratio = 1080 := by
  sorry

#eval pyarelal_loss 1200 1 9

end pyarelal_loss_calculation_l1355_135574


namespace homework_reading_assignment_l1355_135576

theorem homework_reading_assignment (sam_pages pam_pages harrison_pages assigned_pages : ℕ) : 
  sam_pages = 100 →
  sam_pages = 2 * pam_pages →
  pam_pages = harrison_pages + 15 →
  harrison_pages = assigned_pages + 10 →
  assigned_pages = 25 := by
sorry

end homework_reading_assignment_l1355_135576


namespace child_ticket_cost_l1355_135527

/-- Calculates the cost of a child movie ticket given the following information:
  * Adult ticket cost is $9.50
  * Total group size is 7
  * Number of adults is 3
  * Total amount paid is $54.50
-/
theorem child_ticket_cost : 
  let adult_cost : ℝ := 9.50
  let total_group : ℕ := 7
  let num_adults : ℕ := 3
  let total_paid : ℝ := 54.50
  let num_children : ℕ := total_group - num_adults
  let child_cost : ℝ := (total_paid - (adult_cost * num_adults)) / num_children
  child_cost = 6.50 := by sorry

end child_ticket_cost_l1355_135527


namespace complex_equation_sum_l1355_135516

theorem complex_equation_sum (x y : ℝ) : 
  Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 3) → x + y = 13 / 2 := by
  sorry

end complex_equation_sum_l1355_135516


namespace symmetry_origin_symmetry_point_l1355_135522

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the origin
def symmetricToOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

-- Define symmetry with respect to another point
def symmetricToPoint (p : Point2D) (k : Point2D) : Point2D :=
  { x := 2 * k.x - p.x, y := 2 * k.y - p.y }

-- Theorem for symmetry with respect to the origin
theorem symmetry_origin (m : Point2D) :
  symmetricToOrigin m = { x := -m.x, y := -m.y } := by
  sorry

-- Theorem for symmetry with respect to another point
theorem symmetry_point (m k : Point2D) :
  symmetricToPoint m k = { x := 2 * k.x - m.x, y := 2 * k.y - m.y } := by
  sorry

end symmetry_origin_symmetry_point_l1355_135522


namespace christmas_decorations_distribution_l1355_135523

/-- The number of decorations in each box -/
def decorations_per_box : ℕ := 10

/-- The total number of decorations handed out -/
def total_decorations : ℕ := 120

/-- The number of families who received a box of decorations -/
def num_families : ℕ := 11

theorem christmas_decorations_distribution :
  decorations_per_box * (num_families + 1) = total_decorations :=
sorry

end christmas_decorations_distribution_l1355_135523


namespace green_mm_probability_l1355_135551

-- Define the initial state and actions
def initial_green : ℕ := 20
def initial_red : ℕ := 20
def green_eaten : ℕ := 12
def red_eaten : ℕ := initial_red / 2
def yellow_added : ℕ := 14

-- Calculate the final numbers
def final_green : ℕ := initial_green - green_eaten
def final_red : ℕ := initial_red - red_eaten
def final_yellow : ℕ := yellow_added

-- Calculate the total number of M&Ms after all actions
def total_mms : ℕ := final_green + final_red + final_yellow

-- Define the probability of selecting a green M&M
def prob_green : ℚ := final_green / total_mms

-- Theorem statement
theorem green_mm_probability : prob_green = 1/4 := by sorry

end green_mm_probability_l1355_135551


namespace gcf_540_196_l1355_135591

theorem gcf_540_196 : Nat.gcd 540 196 = 4 := by
  sorry

end gcf_540_196_l1355_135591


namespace quadratic_one_solution_l1355_135596

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + k = 0) ↔ k = 49 / 12 := by
  sorry

end quadratic_one_solution_l1355_135596


namespace white_pieces_count_l1355_135554

/-- The number of possible arrangements of chess pieces -/
def total_arrangements : ℕ := 144

/-- The number of black chess pieces -/
def black_pieces : ℕ := 3

/-- Function to calculate the number of arrangements given white and black pieces -/
def arrangements (white : ℕ) (black : ℕ) : ℕ :=
  (Nat.factorial white) * (Nat.factorial black)

/-- Theorem stating that there are 4 white chess pieces -/
theorem white_pieces_count :
  ∃ (w : ℕ), w > 0 ∧ 
    arrangements w black_pieces = total_arrangements ∧ 
    (w = black_pieces ∨ w = black_pieces + 1) :=
by sorry

end white_pieces_count_l1355_135554


namespace gardener_hours_per_day_l1355_135502

/-- Calculates the number of hours a gardener works each day given the project details --/
theorem gardener_hours_per_day
  (total_cost : ℕ)
  (num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℕ)
  (gardener_hourly_rate : ℕ)
  (num_work_days : ℕ)
  (soil_volume : ℕ)
  (soil_cost_per_unit : ℕ)
  (h_total_cost : total_cost = 4100)
  (h_num_rose_bushes : num_rose_bushes = 20)
  (h_cost_per_rose_bush : cost_per_rose_bush = 150)
  (h_gardener_hourly_rate : gardener_hourly_rate = 30)
  (h_num_work_days : num_work_days = 4)
  (h_soil_volume : soil_volume = 100)
  (h_soil_cost_per_unit : soil_cost_per_unit = 5) :
  (total_cost - (num_rose_bushes * cost_per_rose_bush + soil_volume * soil_cost_per_unit)) / gardener_hourly_rate / num_work_days = 5 := by
  sorry

end gardener_hours_per_day_l1355_135502


namespace functional_equation_solution_l1355_135531

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) + f (x + y) = f x * f y + f x + f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l1355_135531


namespace periodic_function_value_l1355_135547

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value (f : ℝ → ℝ) :
  is_periodic f 4 →
  (∀ x ∈ Set.Icc (-2) 2, f x = x) →
  f 7.6 = -0.4 := by
sorry

end periodic_function_value_l1355_135547


namespace omega_range_l1355_135595

/-- Given a function f(x) = 2sin(ωx) with ω > 0, if f(x) has a minimum value of -2 
    on the interval [-π/3, π/4], then 0 < ω ≤ 3/2 -/
theorem omega_range (ω : ℝ) (h1 : ω > 0) : 
  (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) = -2) →
  0 < ω ∧ ω ≤ 3/2 := by
sorry

end omega_range_l1355_135595


namespace marty_voters_l1355_135511

theorem marty_voters (total : ℕ) (biff_percent : ℚ) (undecided_percent : ℚ) 
  (h1 : total = 200)
  (h2 : biff_percent = 45 / 100)
  (h3 : undecided_percent = 8 / 100) :
  ⌊(1 - biff_percent - undecided_percent) * total⌋ = 94 := by
  sorry

end marty_voters_l1355_135511


namespace lcm_18_45_l1355_135536

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l1355_135536


namespace unique_positive_number_l1355_135557

theorem unique_positive_number : ∃! (n : ℝ), n > 0 ∧ (1/5 * n) * (1/7 * n) = n := by
  sorry

end unique_positive_number_l1355_135557


namespace complex_equation_solution_l1355_135553

theorem complex_equation_solution : ∃ (a b : ℝ), (Complex.mk a b) * (Complex.mk a b + Complex.I) * (Complex.mk a b + 2 * Complex.I) = 1001 * Complex.I := by
  sorry

end complex_equation_solution_l1355_135553


namespace reservoir_capacity_difference_l1355_135520

/-- Represents the properties of a reservoir --/
structure Reservoir where
  current_level : ℝ
  normal_level : ℝ
  total_capacity : ℝ
  evaporation_rate : ℝ

/-- Theorem about the difference between total capacity and normal level after evaporation --/
theorem reservoir_capacity_difference (r : Reservoir) 
  (h1 : r.current_level = 14)
  (h2 : r.current_level = 2 * r.normal_level)
  (h3 : r.current_level = 0.7 * r.total_capacity)
  (h4 : r.evaporation_rate = 0.1) :
  r.total_capacity - (r.normal_level * (1 - r.evaporation_rate)) = 13.7 := by
  sorry

end reservoir_capacity_difference_l1355_135520


namespace one_fourths_in_seven_halves_l1355_135546

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end one_fourths_in_seven_halves_l1355_135546


namespace island_liars_count_l1355_135542

/-- Represents the types of inhabitants on the island -/
inductive Inhabitant
  | Knight
  | Liar

/-- The total number of inhabitants on the island -/
def total_inhabitants : Nat := 2001

/-- A function that returns true if the statement "more than half of the others are liars" is true -/
def more_than_half_others_are_liars (num_liars : Nat) : Prop :=
  num_liars > (total_inhabitants - 1) / 2

/-- A function that determines if an inhabitant's statement is consistent with their type -/
def consistent_statement (inhabitant : Inhabitant) (num_liars : Nat) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => more_than_half_others_are_liars num_liars
  | Inhabitant.Liar => ¬(more_than_half_others_are_liars num_liars)

theorem island_liars_count :
  ∃ (num_liars : Nat),
    num_liars ≤ total_inhabitants ∧
    (∀ (i : Inhabitant), consistent_statement i num_liars) ∧
    num_liars = 1001 := by
  sorry

end island_liars_count_l1355_135542


namespace asian_art_pieces_l1355_135566

theorem asian_art_pieces (total : ℕ) (egyptian : ℕ) (asian : ℕ) 
  (h1 : total = 992) 
  (h2 : egyptian = 527) 
  (h3 : total = egyptian + asian) : 
  asian = 465 := by
sorry

end asian_art_pieces_l1355_135566


namespace quadratic_equation_m_value_l1355_135575

theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 2) * x^(m^2 - 2) - m*x + 1 = a*x^2 + b*x + c) ↔ m = -2 :=
sorry

end quadratic_equation_m_value_l1355_135575


namespace parallel_lines_imply_a_eq_neg_three_l1355_135561

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) (x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

/-- Theorem: If l₁ and l₂ are parallel, then a = -3 -/
theorem parallel_lines_imply_a_eq_neg_three (a : ℝ) :
  (∀ x y : ℝ, line_l₁ a x y ↔ line_l₂ a x y) → a = -3 :=
sorry

end parallel_lines_imply_a_eq_neg_three_l1355_135561
