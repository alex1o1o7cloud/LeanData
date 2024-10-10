import Mathlib

namespace rectangle_width_calculation_l4085_408587

theorem rectangle_width_calculation (big_length : ℝ) (small_area : ℝ) :
  big_length = 40 →
  small_area = 200 →
  ∃ (big_width : ℝ),
    big_width = 20 ∧
    small_area = (big_length / 2) * (big_width / 2) :=
by sorry

end rectangle_width_calculation_l4085_408587


namespace find_p_value_l4085_408560

theorem find_p_value (p q r : ℂ) (h_p_real : p.im = 0) 
  (h_sum : p + q + r = 5)
  (h_sum_prod : p * q + q * r + r * p = 5)
  (h_prod : p * q * r = 5) : 
  p = 4 := by sorry

end find_p_value_l4085_408560


namespace squared_gt_iff_abs_gt_l4085_408590

theorem squared_gt_iff_abs_gt (a b : ℝ) : a^2 > b^2 ↔ |a| > |b| := by sorry

end squared_gt_iff_abs_gt_l4085_408590


namespace spinsters_to_cats_ratio_l4085_408510

theorem spinsters_to_cats_ratio : 
  ∀ (S C : ℕ),
  S = 22 →
  C = S + 55 →
  (S : ℚ) / C = 2 / 7 :=
by
  sorry

end spinsters_to_cats_ratio_l4085_408510


namespace derivative_tan_and_exp_minus_sqrt_l4085_408569

open Real

theorem derivative_tan_and_exp_minus_sqrt (x : ℝ) : 
  (deriv tan x = 1 / (cos x)^2) ∧ 
  (deriv (fun x => exp x - sqrt x) x = exp x - 1 / (2 * sqrt x)) :=
by sorry

end derivative_tan_and_exp_minus_sqrt_l4085_408569


namespace palindrome_difference_l4085_408541

def is_palindrome (n : ℕ) : Prop :=
  ∃ (d : List ℕ), n = d.foldl (λ acc x => acc * 10 + x) 0 ∧ d = d.reverse

def has_9_digits (n : ℕ) : Prop :=
  999999999 ≥ n ∧ n ≥ 100000000

def starts_with_nonzero (n : ℕ) : Prop :=
  n ≥ 100000000

def consecutive_palindromes (m n : ℕ) : Prop :=
  is_palindrome m ∧ is_palindrome n ∧ n > m ∧
  ∀ k, m < k ∧ k < n → ¬is_palindrome k

theorem palindrome_difference (m n : ℕ) :
  has_9_digits m ∧ has_9_digits n ∧
  starts_with_nonzero m ∧ starts_with_nonzero n ∧
  consecutive_palindromes m n →
  n - m = 100000011 := by sorry

end palindrome_difference_l4085_408541


namespace not_divisible_by_121_l4085_408509

theorem not_divisible_by_121 (n : ℤ) : ¬(121 ∣ (n^2 + 3*n + 5)) := by
  sorry

end not_divisible_by_121_l4085_408509


namespace mixture_volume_proof_l4085_408553

/-- Proves that given a mixture with an initial ratio of milk to water of 3:1,
    if adding 5 litres of milk changes the ratio to 4:1,
    then the initial volume of the mixture was 20 litres. -/
theorem mixture_volume_proof (V : ℝ) : 
  (3 / 4 * V) / (1 / 4 * V) = 3 / 1 →  -- Initial ratio of milk to water is 3:1
  ((3 / 4 * V + 5) / (1 / 4 * V) = 4 / 1) →  -- New ratio after adding 5 litres of milk is 4:1
  V = 20 := by  -- Initial volume is 20 litres
sorry

end mixture_volume_proof_l4085_408553


namespace book_difference_l4085_408549

def jungkook_initial : ℕ := 28
def seokjin_initial : ℕ := 28
def jungkook_bought : ℕ := 18
def seokjin_bought : ℕ := 11

theorem book_difference : 
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 := by
  sorry

end book_difference_l4085_408549


namespace david_average_marks_l4085_408536

def david_marks : List ℝ := [96, 95, 82, 87, 92]
def num_subjects : ℕ := 5

theorem david_average_marks :
  (david_marks.sum / num_subjects : ℝ) = 90.4 := by
  sorry

end david_average_marks_l4085_408536


namespace reflected_ray_equation_l4085_408597

/-- The equation of a reflected light ray given specific conditions -/
theorem reflected_ray_equation :
  let origin : ℝ × ℝ := (0, 0)
  let incident_line : ℝ → ℝ → Prop := λ x y => 2 * x - y + 5 = 0
  let reflection_point : ℝ × ℝ := (1, 3)
  let reflected_line : ℝ → ℝ → Prop := λ x y => x - 5 * y + 14 = 0
  ∀ (x y : ℝ), reflected_line x y ↔
    ∃ (p : ℝ × ℝ),
      incident_line p.1 p.2 ∧
      (p.1 - origin.1) * (y - p.2) = (x - p.1) * (p.2 - origin.2) ∧
      (p.1 - reflection_point.1) * (y - p.2) = (x - p.1) * (p.2 - reflection_point.2) :=
by sorry

end reflected_ray_equation_l4085_408597


namespace combination_equality_implies_x_values_l4085_408595

theorem combination_equality_implies_x_values (x : ℕ) : 
  (Nat.choose 25 (2 * x) = Nat.choose 25 (x + 4)) → (x = 4 ∨ x = 7) := by
  sorry

end combination_equality_implies_x_values_l4085_408595


namespace profit_percentage_is_25_percent_l4085_408506

def selling_price : ℝ := 670
def original_cost : ℝ := 536

theorem profit_percentage_is_25_percent : 
  (selling_price - original_cost) / original_cost * 100 = 25 := by
  sorry

end profit_percentage_is_25_percent_l4085_408506


namespace batsman_average_l4085_408573

/-- Calculates the new average of a batsman after the 17th inning -/
def newAverage (prevAverage : ℚ) (inningScore : ℕ) (numInnings : ℕ) : ℚ :=
  (prevAverage * (numInnings - 1) + inningScore) / numInnings

/-- Proves that the batsman's new average is 39 runs -/
theorem batsman_average : 
  ∀ (prevAverage : ℚ),
  newAverage prevAverage 87 17 = prevAverage + 3 →
  newAverage prevAverage 87 17 = 39 := by
    sorry

end batsman_average_l4085_408573


namespace textbook_delivery_problem_l4085_408558

theorem textbook_delivery_problem (x y : ℝ) : 
  (0.5 * x + 0.2 * y = 390) ∧ 
  (0.5 * x = 3 * 0.8 * y) →
  (x = 720 ∧ y = 150) := by
sorry

end textbook_delivery_problem_l4085_408558


namespace complex_product_real_l4085_408501

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 + a * I
  let z₂ : ℂ := a - 3 * I
  (z₁ * z₂).im = 0 → a = 3 ∨ a = -3 := by
  sorry

end complex_product_real_l4085_408501


namespace randy_picture_count_randy_drew_five_pictures_l4085_408548

theorem randy_picture_count : ℕ → ℕ → ℕ → Prop :=
  fun randy peter quincy =>
    (peter = randy + 3) →
    (quincy = peter + 20) →
    (randy + peter + quincy = 41) →
    (randy = 5)

-- The proof of the theorem
theorem randy_drew_five_pictures : ∃ (randy peter quincy : ℕ), randy_picture_count randy peter quincy :=
  sorry

end randy_picture_count_randy_drew_five_pictures_l4085_408548


namespace expand_expression_l4085_408502

theorem expand_expression (x : ℝ) : (17 * x - 12) * (3 * x) = 51 * x^2 - 36 * x := by
  sorry

end expand_expression_l4085_408502


namespace smallest_n_divisibility_l4085_408545

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → (¬(24 ∣ m^2) ∨ ¬(720 ∣ m^3))) ∧ 
  24 ∣ n^2 ∧ 720 ∣ n^3 ∧ n = 60 := by
  sorry

end smallest_n_divisibility_l4085_408545


namespace complex_number_property_l4085_408565

theorem complex_number_property (b : ℝ) : 
  let z : ℂ := (2 - b * I) / (1 + 2 * I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end complex_number_property_l4085_408565


namespace right_triangle_area_l4085_408505

/-- The area of a right-angled triangle with sides 30 cm and 40 cm adjacent to the right angle is 600 cm². -/
theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 40) : 
  (1/2) * a * b = 600 := by
  sorry

end right_triangle_area_l4085_408505


namespace crane_height_theorem_l4085_408551

/-- Represents the height of a crane and the building it's working on -/
structure CraneBuilding where
  crane_height : ℝ
  building_height : ℝ

/-- The problem setup -/
def construction_problem (crane2_height : ℝ) : Prop :=
  let crane1 : CraneBuilding := ⟨228, 200⟩
  let crane2 : CraneBuilding := ⟨crane2_height, 100⟩
  let crane3 : CraneBuilding := ⟨147, 140⟩
  let cranes : List CraneBuilding := [crane1, crane2, crane3]
  let avg_height_diff : ℝ := (cranes.map (λ c => c.crane_height - c.building_height)).sum / cranes.length
  let avg_building_height : ℝ := (cranes.map (λ c => c.building_height)).sum / cranes.length
  avg_height_diff = 0.13 * avg_building_height

/-- The theorem to be proved -/
theorem crane_height_theorem : 
  ∃ (h : ℝ), construction_problem h ∧ abs (h - 122) < 1 :=
sorry

end crane_height_theorem_l4085_408551


namespace max_silver_medals_for_27_points_l4085_408529

/-- Represents the types of medals in the competition -/
inductive Medal
| Gold
| Silver
| Bronze

/-- Returns the point value of a given medal -/
def medal_points (m : Medal) : Nat :=
  match m with
  | Medal.Gold => 5
  | Medal.Silver => 3
  | Medal.Bronze => 1

/-- Represents a competitor's medal collection -/
structure MedalCollection where
  gold : Nat
  silver : Nat
  bronze : Nat

/-- Calculates the total points for a given medal collection -/
def total_points (mc : MedalCollection) : Nat :=
  mc.gold * medal_points Medal.Gold +
  mc.silver * medal_points Medal.Silver +
  mc.bronze * medal_points Medal.Bronze

/-- The main theorem to prove -/
theorem max_silver_medals_for_27_points :
  ∃ (mc : MedalCollection),
    total_points mc = 27 ∧
    mc.gold + mc.silver + mc.bronze ≤ 8 ∧
    mc.silver = 4 ∧
    ∀ (mc' : MedalCollection),
      total_points mc' = 27 →
      mc'.gold + mc'.silver + mc'.bronze ≤ 8 →
      mc'.silver ≤ 4 := by
  sorry


end max_silver_medals_for_27_points_l4085_408529


namespace min_reciprocal_sum_l4085_408598

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 1) :
  1/x + 1/y + 1/z ≥ 9 := by
  sorry

end min_reciprocal_sum_l4085_408598


namespace negative_five_squared_opposite_l4085_408537

-- Define opposite numbers
def are_opposite (a b : ℤ) : Prop := a = -b

-- Theorem statement
theorem negative_five_squared_opposite : are_opposite (-5^2) ((-5)^2) := by
  sorry

end negative_five_squared_opposite_l4085_408537


namespace parallel_and_perpendicular_properties_l4085_408550

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_and_perpendicular_properties
  (a b c : Line) (γ : Plane) :
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perpendicular a γ ∧ perpendicular b γ → parallel a b) :=
sorry

end parallel_and_perpendicular_properties_l4085_408550


namespace quadratic_roots_product_l4085_408584

theorem quadratic_roots_product (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 1 →
  x₁ * x₂ = -2 := by
sorry

end quadratic_roots_product_l4085_408584


namespace no_real_roots_l4085_408577

theorem no_real_roots (m : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 4 ∧ ∀ x ∈ s, (x : ℝ) - m < 0 ∧ 7 - 2*(x : ℝ) ≤ 1) →
  ∀ x : ℝ, 8*x^2 - 8*x + m ≠ 0 :=
by sorry

end no_real_roots_l4085_408577


namespace unique_quadratic_solution_l4085_408508

/-- If the set A = {x ∈ ℝ | ax² + ax + 1 = 0} has only one element, then a = 4 -/
theorem unique_quadratic_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) → a = 4 := by
sorry

end unique_quadratic_solution_l4085_408508


namespace solve_salary_problem_l4085_408594

def salary_problem (S : ℝ) : Prop :=
  let rent := (2/5) * S
  let food := (3/10) * S
  let conveyance := (1/8) * S
  (food + conveyance = 3400) →
  (S - (rent + food + conveyance) = 1400)

theorem solve_salary_problem :
  ∃ S : ℝ, salary_problem S :=
sorry

end solve_salary_problem_l4085_408594


namespace plot_width_l4085_408592

/-- 
Given a rectangular plot with length 90 meters, if 60 poles placed 5 meters apart 
are needed to enclose the plot, then the width of the plot is 60 meters.
-/
theorem plot_width (poles : ℕ) (pole_distance : ℝ) (length width : ℝ) : 
  poles = 60 → 
  pole_distance = 5 → 
  length = 90 → 
  poles * pole_distance = 2 * (length + width) → 
  width = 60 := by sorry

end plot_width_l4085_408592


namespace polynomial_simplification_l4085_408512

theorem polynomial_simplification (x : ℝ) :
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = 32*x^5 := by
  sorry

end polynomial_simplification_l4085_408512


namespace parallel_vectors_m_value_l4085_408561

/-- Two 2D vectors are parallel if the cross product of their coordinates is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (m, m + 1)
  parallel a b → m = -3/2 := by
  sorry

end parallel_vectors_m_value_l4085_408561


namespace lunch_ratio_proof_l4085_408586

theorem lunch_ratio_proof (total_students : Nat) (cafeteria_students : Nat) (no_lunch_students : Nat) :
  total_students = 60 →
  cafeteria_students = 10 →
  no_lunch_students = 20 →
  ∃ k : Nat, total_students - cafeteria_students - no_lunch_students = k * cafeteria_students →
  (total_students - cafeteria_students - no_lunch_students) / cafeteria_students = 3 :=
by
  sorry

end lunch_ratio_proof_l4085_408586


namespace complex_power_modulus_l4085_408513

theorem complex_power_modulus : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 4) = 256 := by
  sorry

end complex_power_modulus_l4085_408513


namespace quadratic_minimum_unique_minimum_l4085_408581

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem quadratic_minimum (x : ℝ) : f x ≥ f 7 := by
  sorry

theorem unique_minimum : ∀ x : ℝ, x ≠ 7 → f x > f 7 := by
  sorry

end quadratic_minimum_unique_minimum_l4085_408581


namespace f_value_at_5pi_3_l4085_408522

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_value_at_5pi_3 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : is_periodic f π)
  (h_sin : ∀ x ∈ Set.Icc 0 (π/2), f x = Real.sin x) :
  f (5*π/3) = -Real.sqrt 3 / 2 := by
sorry

end f_value_at_5pi_3_l4085_408522


namespace power_of_power_l4085_408580

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l4085_408580


namespace functional_equation_solution_l4085_408572

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014) →
  ∃ c : ℤ, ∀ m : ℤ, f m = 2 * m + c :=
by sorry

end functional_equation_solution_l4085_408572


namespace thompson_class_median_l4085_408589

/-- Represents the number of families with a specific number of children -/
structure FamilyCount where
  childCount : ℕ
  familyCount : ℕ

/-- Calculates the median of a list of natural numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Expands a list of FamilyCount into a list of individual family sizes -/
def expandCounts (counts : List FamilyCount) : List ℕ :=
  sorry

theorem thompson_class_median :
  let familyCounts : List FamilyCount := [
    ⟨1, 4⟩, ⟨2, 3⟩, ⟨3, 5⟩, ⟨4, 2⟩, ⟨5, 1⟩
  ]
  let expandedList := expandCounts familyCounts
  median expandedList = 3 := by
  sorry

end thompson_class_median_l4085_408589


namespace sum_of_digits_9ab_l4085_408564

/-- Given an integer n and a digit d, returns the number composed of n repetitions of d -/
def repeat_digit (n : ℕ) (d : ℕ) : ℕ := 
  d * (10^n - 1) / 9

/-- Returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_9ab (a b : ℕ) : 
  a = repeat_digit 1985 8 → 
  b = repeat_digit 1985 5 → 
  sum_of_digits (9 * a * b) = 17865 := by
sorry

end sum_of_digits_9ab_l4085_408564


namespace impossible_wire_arrangement_l4085_408575

-- Define a regular heptagon with columns
structure RegularHeptagonWithColumns where
  vertices : Fin 7 → ℝ
  is_regular : True  -- Placeholder for regularity condition

-- Define the connection between vertices
def second_nearest_neighbors (i : Fin 7) : Fin 7 × Fin 7 :=
  ((i + 2) % 7, (i + 5) % 7)

-- Define the intersection of wires
def wire_intersections (h : RegularHeptagonWithColumns) (i j : Fin 7) : Prop :=
  let (a, b) := second_nearest_neighbors i
  let (c, d) := second_nearest_neighbors j
  (a = c ∧ b ≠ d) ∨ (a = d ∧ b ≠ c) ∨ (b = c ∧ a ≠ d) ∨ (b = d ∧ a ≠ c)

-- Define the condition for wire arrangement
def valid_wire_arrangement (h : RegularHeptagonWithColumns) : Prop :=
  ∀ i j k : Fin 7, wire_intersections h i j → wire_intersections h i k →
    (h.vertices i < h.vertices j ∧ h.vertices i > h.vertices k) ∨
    (h.vertices i > h.vertices j ∧ h.vertices i < h.vertices k)

-- Theorem statement
theorem impossible_wire_arrangement :
  ¬∃ (h : RegularHeptagonWithColumns), valid_wire_arrangement h :=
sorry

end impossible_wire_arrangement_l4085_408575


namespace fiftieth_term_divisible_by_five_l4085_408547

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem fiftieth_term_divisible_by_five : 
  5 ∣ modifiedLucas 49 := by sorry

end fiftieth_term_divisible_by_five_l4085_408547


namespace cannot_determine_best_method_l4085_408563

/-- Represents an investment method --/
inductive InvestmentMethod
  | OneYear
  | ThreeYearThenOneYear
  | FiveOneYearThenFiveYear

/-- Calculates the final amount for a given investment method --/
def calculateFinalAmount (method : InvestmentMethod) (initialAmount : ℝ) : ℝ :=
  match method with
  | .OneYear => initialAmount * (1 + 0.0156) ^ 10
  | .ThreeYearThenOneYear => initialAmount * (1 + 0.0206 * 3) ^ 3 * (1 + 0.0156)
  | .FiveOneYearThenFiveYear => initialAmount * (1 + 0.0156) ^ 5 * (1 + 0.0282 * 5)

/-- Theorem stating that the best investment method cannot be determined without calculation --/
theorem cannot_determine_best_method (initialAmount : ℝ) :
  ∀ (m1 m2 : InvestmentMethod), m1 ≠ m2 →
  ∃ (result1 result2 : ℝ),
    calculateFinalAmount m1 initialAmount = result1 ∧
    calculateFinalAmount m2 initialAmount = result2 ∧
    (result1 > result2 ∨ result1 < result2) :=
by
  sorry

#check cannot_determine_best_method

end cannot_determine_best_method_l4085_408563


namespace james_profit_l4085_408566

def total_toys : ℕ := 200
def buy_price : ℕ := 20
def sell_price : ℕ := 30
def sell_percentage : ℚ := 80 / 100

theorem james_profit :
  (↑total_toys * sell_percentage * sell_price : ℚ) -
  (↑total_toys * sell_percentage * buy_price : ℚ) = 1600 := by
  sorry

end james_profit_l4085_408566


namespace nth_power_divisibility_l4085_408593

theorem nth_power_divisibility (b n : ℕ) (h1 : b > 1) (h2 : n > 1)
  (h3 : ∀ k : ℕ, k > 1 → ∃ a_k : ℕ, k ∣ (b - a_k^n)) :
  ∃ A : ℕ, b = A^n := by sorry

end nth_power_divisibility_l4085_408593


namespace inequality_proof_l4085_408567

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 := by
  sorry

end inequality_proof_l4085_408567


namespace fifth_day_distance_l4085_408562

def running_sequence (n : ℕ) : ℕ := 2 + n - 1

theorem fifth_day_distance : running_sequence 5 = 6 := by
  sorry

end fifth_day_distance_l4085_408562


namespace compare_expressions_l4085_408582

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 / b + b^2 / a) > (a + b) := by
  sorry

end compare_expressions_l4085_408582


namespace problem_solution_l4085_408514

theorem problem_solution (x y z : ℝ) 
  (sum_condition : x + y + z = 150)
  (equal_condition : x - 5 = y + 3 ∧ y + 3 = z^2) :
  y = 71 := by
  sorry

end problem_solution_l4085_408514


namespace negative_three_and_half_equality_l4085_408540

theorem negative_three_and_half_equality : -4 + (1/2 : ℚ) = -(7/2 : ℚ) := by
  sorry

end negative_three_and_half_equality_l4085_408540


namespace fifteen_consecutive_naturals_l4085_408588

theorem fifteen_consecutive_naturals (N : ℕ) : 
  (N < 81 ∧ 
   ∀ k : ℕ, (N < k ∧ k < 81) → (k - N ≤ 15)) ∧ 
  (∃ m : ℕ, N < m ∧ m < 81 ∧ m - N = 15) →
  N = 66 := by
sorry

end fifteen_consecutive_naturals_l4085_408588


namespace marble_sum_theorem_l4085_408530

theorem marble_sum_theorem (atticus jensen cruz : ℕ) : 
  atticus = 4 → 
  cruz = 8 → 
  atticus = jensen / 2 → 
  3 * (atticus + jensen + cruz) = 60 := by
  sorry

end marble_sum_theorem_l4085_408530


namespace inequality_proof_l4085_408515

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (2 * a * b) / (a + b) := by
  sorry

end inequality_proof_l4085_408515


namespace lily_shopping_exceeds_budget_l4085_408527

/-- Proves that the total cost of items exceeds Lily's initial amount --/
theorem lily_shopping_exceeds_budget :
  let initial_amount : ℝ := 70
  let celery_price : ℝ := 8 * (1 - 0.2)
  let cereal_price : ℝ := 14
  let bread_price : ℝ := 10 * (1 - 0.05)
  let milk_price : ℝ := 12 * (1 - 0.15)
  let potato_price : ℝ := 2 * 8
  let cookie_price : ℝ := 15
  let tax_rate : ℝ := 0.07
  let total_cost : ℝ := (celery_price + cereal_price + bread_price + milk_price + potato_price + cookie_price) * (1 + tax_rate)
  total_cost > initial_amount := by sorry

end lily_shopping_exceeds_budget_l4085_408527


namespace max_d_is_15_l4085_408585

/-- Represents a 6-digit number of the form x5d,33e -/
structure SixDigitNumber where
  x : Nat
  d : Nat
  e : Nat
  h_x : x < 10
  h_d : d < 10
  h_e : e < 10

/-- Checks if a SixDigitNumber is divisible by 33 -/
def isDivisibleBy33 (n : SixDigitNumber) : Prop :=
  (n.x + n.d + n.e + 11) % 3 = 0 ∧ (n.x + n.d - n.e - 5) % 11 = 0

/-- The maximum value of d in a SixDigitNumber divisible by 33 is 15 -/
theorem max_d_is_15 : 
  ∀ n : SixDigitNumber, isDivisibleBy33 n → n.d ≤ 15 ∧ 
  ∃ m : SixDigitNumber, isDivisibleBy33 m ∧ m.d = 15 := by sorry

end max_d_is_15_l4085_408585


namespace student_tickets_sold_l4085_408535

theorem student_tickets_sold (total_tickets : ℕ) (total_money : ℕ) 
  (student_price : ℕ) (nonstudent_price : ℕ) 
  (h1 : total_tickets = 821)
  (h2 : total_money = 1933)
  (h3 : student_price = 2)
  (h4 : nonstudent_price = 3) :
  ∃ (student_tickets : ℕ), 
    student_tickets + (total_tickets - student_tickets) = total_tickets ∧
    student_price * student_tickets + nonstudent_price * (total_tickets - student_tickets) = total_money ∧
    student_tickets = 530 :=
by
  sorry

#check student_tickets_sold

end student_tickets_sold_l4085_408535


namespace max_obtuse_triangles_four_points_l4085_408557

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle formed by three points -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Predicate to check if a triangle is obtuse -/
def isObtuse (t : Triangle) : Prop :=
  sorry

/-- The set of all possible triangles formed by 4 points -/
def allTriangles (p1 p2 p3 p4 : Point) : Set Triangle :=
  sorry

/-- The number of obtuse triangles in a set of triangles -/
def numObtuseTriangles (ts : Set Triangle) : ℕ :=
  sorry

/-- Theorem: The maximum number of obtuse triangles formed by 4 points is 4 -/
theorem max_obtuse_triangles_four_points (p1 p2 p3 p4 : Point) :
  ∃ (arrangement : Point → Point),
    numObtuseTriangles (allTriangles (arrangement p1) (arrangement p2) (arrangement p3) (arrangement p4)) ≤ 4 ∧
    ∃ (q1 q2 q3 q4 : Point),
      numObtuseTriangles (allTriangles q1 q2 q3 q4) = 4 :=
sorry

end max_obtuse_triangles_four_points_l4085_408557


namespace solve_linear_equation_l4085_408500

theorem solve_linear_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end solve_linear_equation_l4085_408500


namespace arithmetic_expressions_evaluation_l4085_408570

theorem arithmetic_expressions_evaluation :
  (2 * (-1)^3 - (-2)^2 / 4 + 10 = 7) ∧
  (abs (-3) - (-6 + 4) / (-1/2)^3 + (-1)^2013 = -14) := by
  sorry

end arithmetic_expressions_evaluation_l4085_408570


namespace bill_sunday_saturday_difference_l4085_408517

/-- Represents the miles run by Bill and Julia on Saturday and Sunday -/
structure WeekendRun where
  billSat : ℕ
  billSun : ℕ
  juliaSat : ℕ
  juliaSun : ℕ

/-- The conditions of the problem -/
def weekend_run_conditions (run : WeekendRun) : Prop :=
  run.billSun > run.billSat ∧
  run.juliaSat = 0 ∧
  run.juliaSun = 2 * run.billSun ∧
  run.billSat + run.billSun + run.juliaSat + run.juliaSun = 28 ∧
  run.billSun = 8

/-- The theorem to prove -/
theorem bill_sunday_saturday_difference (run : WeekendRun) 
  (h : weekend_run_conditions run) : 
  run.billSun - run.billSat = 4 := by
sorry

end bill_sunday_saturday_difference_l4085_408517


namespace gravel_cost_is_correct_l4085_408552

/-- Represents the dimensions and cost parameters of a rectangular plot with a gravel path --/
structure PlotWithPath where
  length : Real
  width : Real
  pathWidth : Real
  gravelCost : Real

/-- Calculates the cost of gravelling the path for a given plot --/
def calculateGravellingCost (plot : PlotWithPath) : Real :=
  let outerArea := plot.length * plot.width
  let innerLength := plot.length - 2 * plot.pathWidth
  let innerWidth := plot.width - 2 * plot.pathWidth
  let innerArea := innerLength * innerWidth
  let pathArea := outerArea - innerArea
  pathArea * plot.gravelCost

/-- Theorem stating that the cost of gravelling the path for the given plot is 8.844 rupees --/
theorem gravel_cost_is_correct (plot : PlotWithPath) 
  (h1 : plot.length = 110)
  (h2 : plot.width = 0.65)
  (h3 : plot.pathWidth = 0.05)
  (h4 : plot.gravelCost = 0.8) :
  calculateGravellingCost plot = 8.844 := by
  sorry

#eval calculateGravellingCost { length := 110, width := 0.65, pathWidth := 0.05, gravelCost := 0.8 }

end gravel_cost_is_correct_l4085_408552


namespace range_of_m_l4085_408511

theorem range_of_m : ∀ m : ℝ, 
  (¬∃ x : ℝ, 1 < x ∧ x < 3 ∧ x^2 - m*x - 1 = 0) ↔ 
  (m ≤ 0 ∨ m ≥ 8/3) := by sorry

end range_of_m_l4085_408511


namespace goods_train_length_l4085_408518

/-- The length of a goods train passing a man in an opposite moving train -/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) (passing_time : ℝ) : 
  man_train_speed = 20 →
  goods_train_speed = 92 →
  passing_time = 9 →
  ∃ (length : ℝ), abs (length - 279.99) < 0.01 ∧ 
    length = (man_train_speed + goods_train_speed) * (5/18) * passing_time :=
by
  sorry

#check goods_train_length

end goods_train_length_l4085_408518


namespace largest_sum_of_digits_l4085_408531

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of all digits displayed on the watch -/
def totalSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits on a 24-hour format digital watch -/
def maxSumOfDigits : Nat := 23

theorem largest_sum_of_digits :
  ∀ t : Time24, totalSumOfDigits t ≤ maxSumOfDigits :=
by sorry

end largest_sum_of_digits_l4085_408531


namespace absolute_value_equation_solution_l4085_408591

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x - 3| = 5 - 2*x ↔ x = 8/3 ∨ x = 2 := by sorry

end absolute_value_equation_solution_l4085_408591


namespace separate_amount_possible_l4085_408539

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | EqualGroup (value : ℚ)
  | UnequalGroups (value1 value2 : ℚ)

/-- Represents a weighing operation -/
def Weighing := List ℚ → WeighingResult

/-- The total amount of money in rubles -/
def total_amount : ℚ := 80

/-- The value of a single coin in rubles -/
def coin_value : ℚ := 1/20

/-- The target amount to be separated -/
def target_amount : ℚ := 25

/-- The maximum number of weighings allowed -/
def max_weighings : ℕ := 4

/-- 
  Proves that it's possible to separate the target amount from the total amount 
  using coins of the given value with only a balance scale in the specified number of weighings
-/
theorem separate_amount_possible : 
  ∃ (weighings : List Weighing), 
    weighings.length ≤ max_weighings ∧ 
    ∃ (result : List ℚ), 
      result.sum = target_amount ∧ 
      result.all (λ x => x ≤ total_amount) :=
sorry

end separate_amount_possible_l4085_408539


namespace train_overtake_l4085_408568

-- Define the speeds of the trains
def speed_A : ℝ := 30
def speed_B : ℝ := 45

-- Define the overtake distance
def overtake_distance : ℝ := 180

-- Define the time difference between train departures
def time_difference : ℝ := 2

-- Theorem statement
theorem train_overtake :
  speed_A * (time_difference + (overtake_distance / speed_B)) = overtake_distance ∧
  speed_B * (overtake_distance / speed_B) = overtake_distance := by
  sorry

end train_overtake_l4085_408568


namespace square_of_1023_l4085_408533

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end square_of_1023_l4085_408533


namespace school_trip_seats_l4085_408520

/-- Given a total number of students and buses, calculate the number of seats per bus -/
def seatsPerBus (students : ℕ) (buses : ℕ) : ℚ :=
  (students : ℚ) / (buses : ℚ)

/-- Theorem: Given 14 students and 7 buses, the number of seats on each bus is 2 -/
theorem school_trip_seats :
  seatsPerBus 14 7 = 2 := by
  sorry

end school_trip_seats_l4085_408520


namespace problems_per_page_l4085_408532

theorem problems_per_page 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (total_problems : ℕ) 
  (h1 : math_pages = 6) 
  (h2 : reading_pages = 4) 
  (h3 : total_problems = 30) : 
  total_problems / (math_pages + reading_pages) = 3 := by
sorry

end problems_per_page_l4085_408532


namespace other_solution_quadratic_equation_l4085_408559

theorem other_solution_quadratic_equation :
  let f : ℚ → ℚ := λ x ↦ 45 * x^2 - 56 * x + 31
  ∃ x : ℚ, x ≠ 2/5 ∧ f x = 0 ∧ x = 7/9 := by
  sorry

end other_solution_quadratic_equation_l4085_408559


namespace consecutive_triangle_sides_l4085_408519

theorem consecutive_triangle_sides (n : ℕ) (h : n ≥ 1) :
  ∀ (a : ℕ → ℕ), (∀ i, i < 6*n → a (i+1) = a i + 1) →
  ∃ i, i + 2 < 6*n ∧ 
    a i + a (i+1) > a (i+2) ∧
    a i + a (i+2) > a (i+1) ∧
    a (i+1) + a (i+2) > a i :=
sorry

end consecutive_triangle_sides_l4085_408519


namespace gcd_eight_factorial_seven_factorial_l4085_408583

theorem gcd_eight_factorial_seven_factorial :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 7) = Nat.factorial 7 := by
  sorry

end gcd_eight_factorial_seven_factorial_l4085_408583


namespace train_speed_l4085_408556

/-- Prove that given the conditions, the train's speed is 20 miles per hour. -/
theorem train_speed (distance_to_work : ℝ) (walking_speed : ℝ) (additional_train_time : ℝ) 
  (walking_vs_train_time_diff : ℝ) :
  distance_to_work = 1.5 →
  walking_speed = 3 →
  additional_train_time = 10.5 / 60 →
  walking_vs_train_time_diff = 15 / 60 →
  ∃ (train_speed : ℝ), 
    train_speed = 20 ∧
    distance_to_work / walking_speed = 
      distance_to_work / train_speed + additional_train_time + walking_vs_train_time_diff :=
by sorry

end train_speed_l4085_408556


namespace a_16_value_l4085_408524

def sequence_a : ℕ → ℚ
  | 0 => 2
  | (n + 1) => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_16_value : sequence_a 16 = -1/3 := by
  sorry

end a_16_value_l4085_408524


namespace discriminant_neither_sufficient_nor_necessary_l4085_408523

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f(x) = ax^2 + bx + c is always above the x-axis -/
def always_above_x_axis (a b c : ℝ) : Prop :=
  ∀ x, quadratic_function a b c x > 0

/-- The discriminant condition b^2 - 4ac < 0 -/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

/-- Theorem stating that the discriminant condition is neither sufficient nor necessary 
    for the quadratic function to always be above the x-axis -/
theorem discriminant_neither_sufficient_nor_necessary :
  ¬(∀ a b c : ℝ, discriminant_condition a b c → always_above_x_axis a b c) ∧
  ¬(∀ a b c : ℝ, always_above_x_axis a b c → discriminant_condition a b c) :=
sorry

end discriminant_neither_sufficient_nor_necessary_l4085_408523


namespace circle_properties_l4085_408521

/-- Given a circle with equation 3x^2 - 4y - 12 = -3y^2 + 8x, 
    prove its center coordinates, radius, and a + 2b + r -/
theorem circle_properties : 
  ∃ (a b r : ℝ), 
    (∀ x y : ℝ, 3 * x^2 - 4 * y - 12 = -3 * y^2 + 8 * x → 
      (x - a)^2 + (y - b)^2 = r^2) ∧ 
    a = 4/3 ∧ 
    b = 2/3 ∧ 
    r = 2 * Real.sqrt 13 / 3 ∧
    a + 2 * b + r = (8 + 2 * Real.sqrt 13) / 3 :=
by sorry

end circle_properties_l4085_408521


namespace all_three_sports_count_l4085_408574

/-- Represents a sports club with members playing various sports -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  tennis_players : ℕ
  basketball_players : ℕ
  no_sport_players : ℕ
  badminton_tennis_players : ℕ
  badminton_basketball_players : ℕ
  tennis_basketball_players : ℕ

/-- Calculates the number of members playing all three sports -/
def all_three_sports (club : SportsClub) : ℕ :=
  club.total_members - club.no_sport_players -
    (club.badminton_players + club.tennis_players + club.basketball_players -
     club.badminton_tennis_players - club.badminton_basketball_players - club.tennis_basketball_players)

/-- Theorem stating the number of members playing all three sports -/
theorem all_three_sports_count (club : SportsClub)
    (h1 : club.total_members = 60)
    (h2 : club.badminton_players = 25)
    (h3 : club.tennis_players = 30)
    (h4 : club.basketball_players = 15)
    (h5 : club.no_sport_players = 10)
    (h6 : club.badminton_tennis_players = 15)
    (h7 : club.badminton_basketball_players = 10)
    (h8 : club.tennis_basketball_players = 5) :
    all_three_sports club = 10 := by
  sorry


end all_three_sports_count_l4085_408574


namespace team_size_l4085_408542

theorem team_size (first_day_per_person : ℕ) (second_day_multiplier : ℕ) (third_day_total : ℕ) (total_blankets : ℕ) :
  first_day_per_person = 2 →
  second_day_multiplier = 3 →
  third_day_total = 22 →
  total_blankets = 142 →
  ∃ team_size : ℕ, 
    team_size * first_day_per_person + 
    team_size * first_day_per_person * second_day_multiplier + 
    third_day_total = total_blankets ∧
    team_size = 15 :=
by sorry

end team_size_l4085_408542


namespace probability_same_color_is_half_l4085_408503

def num_red_balls : ℕ := 2
def num_white_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_white_balls

def num_possible_outcomes : ℕ := total_balls * total_balls
def num_same_color_outcomes : ℕ := num_red_balls * num_red_balls + num_white_balls * num_white_balls

def probability_same_color : ℚ := num_same_color_outcomes / num_possible_outcomes

theorem probability_same_color_is_half : probability_same_color = 1/2 := by
  sorry

end probability_same_color_is_half_l4085_408503


namespace people_who_got_off_l4085_408507

theorem people_who_got_off (initial_people : ℕ) (remaining_people : ℕ) (h1 : initial_people = 48) (h2 : remaining_people = 31) :
  initial_people - remaining_people = 17 := by
  sorry

end people_who_got_off_l4085_408507


namespace product_zero_implies_factor_zero_l4085_408571

theorem product_zero_implies_factor_zero (a b c : ℝ) : a * b * c = 0 → (a = 0 ∨ b = 0 ∨ c = 0) := by
  sorry

end product_zero_implies_factor_zero_l4085_408571


namespace junior_count_l4085_408516

theorem junior_count (total : ℕ) (junior_percent : ℚ) (senior_percent : ℚ)
  (h_total : total = 40)
  (h_junior_percent : junior_percent = 1/5)
  (h_senior_percent : senior_percent = 1/10)
  (h_equal_team : ∃ (x : ℕ), x * 5 = junior_percent * total ∧ x * 10 = senior_percent * total) :
  ∃ (j : ℕ), j = 12 ∧ j + (total - j) = total ∧
  (junior_percent * j).num = (senior_percent * (total - j)).num :=
sorry

end junior_count_l4085_408516


namespace second_player_winning_strategy_l4085_408525

/-- Represents a position on the 10x10 board -/
def Position := Fin 10 × Fin 10

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a mark on the board -/
inductive Mark
| X
| O

/-- Represents the game state -/
structure GameState where
  board : Position → Option Mark
  currentPlayer : Player

/-- Checks if a position is winning -/
def isWinningPosition (board : Position → Option Mark) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (pos : Position) : GameState :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Position

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Second strategy :=
sorry

end second_player_winning_strategy_l4085_408525


namespace min_moves_to_align_cups_l4085_408555

/-- Represents the state of cups on a table -/
structure CupState where
  totalCups : Nat
  upsideCups : Nat
  downsideCups : Nat

/-- Represents a move that flips exactly 3 cups -/
def flipThreeCups (state : CupState) : CupState :=
  { totalCups := state.totalCups,
    upsideCups := state.upsideCups + 3 - 2 * min 3 state.upsideCups,
    downsideCups := state.downsideCups + 3 - 2 * min 3 state.downsideCups }

/-- Predicate to check if all cups are facing the same direction -/
def allSameDirection (state : CupState) : Prop :=
  state.upsideCups = 0 ∨ state.upsideCups = state.totalCups

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_to_align_cups : 
  ∃ (n : Nat), 
    (∀ (state : CupState), 
      state.totalCups = 10 → 
      state.upsideCups = 5 → 
      state.downsideCups = 5 → 
      ∃ (moves : List (CupState → CupState)), 
        moves.length ≤ n ∧ 
        allSameDirection (moves.foldl (fun s m => m s) state) ∧
        ∀ m, m ∈ moves → m = flipThreeCups) ∧
    (∀ (k : Nat), 
      k < n → 
      ∃ (state : CupState), 
        state.totalCups = 10 ∧ 
        state.upsideCups = 5 ∧ 
        state.downsideCups = 5 ∧ 
        ∀ (moves : List (CupState → CupState)), 
          moves.length ≤ k → 
          (∀ m, m ∈ moves → m = flipThreeCups) → 
          ¬allSameDirection (moves.foldl (fun s m => m s) state)) ∧
    n = 3 :=
sorry

end min_moves_to_align_cups_l4085_408555


namespace unique_solution_3x_4y_5z_l4085_408546

theorem unique_solution_3x_4y_5z : 
  ∀ x y z : ℕ+, 3^(x:ℕ) + 4^(y:ℕ) = 5^(z:ℕ) → x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end unique_solution_3x_4y_5z_l4085_408546


namespace question_types_sum_steve_answerable_relation_l4085_408576

/-- Represents a math test with different types of questions -/
structure MathTest where
  total : ℕ
  word : ℕ
  addition_subtraction : ℕ
  geometry : ℕ

/-- Defines the properties of a valid math test -/
def is_valid_test (test : MathTest) : Prop :=
  test.word = test.total / 2 ∧
  test.addition_subtraction = test.total / 3 ∧
  test.geometry = test.total - test.word - test.addition_subtraction

/-- Theorem stating the relationship between question types and total questions -/
theorem question_types_sum (test : MathTest) (h : is_valid_test test) :
  test.word + test.addition_subtraction + test.geometry = test.total := by
  sorry

/-- Function representing the number of questions Steve can answer -/
def steve_answerable (total : ℕ) : ℕ :=
  total / 2 - 4

/-- Theorem stating the relationship between Steve's answerable questions and total questions -/
theorem steve_answerable_relation (test : MathTest) (h : is_valid_test test) :
  steve_answerable test.total = test.total / 2 - 4 := by
  sorry

end question_types_sum_steve_answerable_relation_l4085_408576


namespace sum_of_products_equals_25079720_l4085_408578

def T : Finset ℕ := Finset.image (fun i => 3^i) (Finset.range 8)

def M : ℕ := (Finset.sum T fun x => 
  (Finset.sum (T.erase x) fun y => x * y))

theorem sum_of_products_equals_25079720 : M = 25079720 := by
  sorry

end sum_of_products_equals_25079720_l4085_408578


namespace smallest_angle_tangent_equality_l4085_408534

theorem smallest_angle_tangent_equality (x : ℝ) : 
  (x > 0) → 
  (x * (180 / π) = 5.625) → 
  (Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x))) → 
  ∀ y : ℝ, (y > 0) → 
    (Real.tan (6 * y) = (Real.cos (2 * y) - Real.sin (2 * y)) / (Real.cos (2 * y) + Real.sin (2 * y))) → 
    (y ≥ x) :=
by sorry

end smallest_angle_tangent_equality_l4085_408534


namespace ochos_friends_ratio_l4085_408538

/-- Given that Ocho has 8 friends, all boys play theater with him, and 4 boys play theater with him,
    prove that the ratio of girls to boys among Ocho's friends is 1:1 -/
theorem ochos_friends_ratio (total_friends : ℕ) (boys_theater : ℕ) 
    (h1 : total_friends = 8)
    (h2 : boys_theater = 4) :
    (total_friends - boys_theater) / boys_theater = 1 := by
  sorry

end ochos_friends_ratio_l4085_408538


namespace f_is_even_and_has_zero_point_l4085_408579

-- Define the function f(x) = x^2 - 1
def f (x : ℝ) : ℝ := x^2 - 1

-- Theorem stating that f is an even function and has a zero point
theorem f_is_even_and_has_zero_point :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ x : ℝ, f x = 0) := by
  sorry

end f_is_even_and_has_zero_point_l4085_408579


namespace count_zeros_up_to_2376_l4085_408544

/-- Returns true if the given positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- Counts the number of positive integers less than or equal to n that contain the digit 0 -/
def countZeros (n : ℕ+) : ℕ :=
  sorry

/-- The number of positive integers less than or equal to 2376 that contain the digit 0 is 578 -/
theorem count_zeros_up_to_2376 : countZeros 2376 = 578 :=
  sorry

end count_zeros_up_to_2376_l4085_408544


namespace five_times_seven_and_two_fifths_l4085_408596

theorem five_times_seven_and_two_fifths (x : ℚ) : x = 5 * (7 + 2/5) → x = 37 := by
  sorry

end five_times_seven_and_two_fifths_l4085_408596


namespace total_cost_is_660_l4085_408528

/-- Represents the cost of t-shirts for employees -/
structure TShirtCost where
  white_men : ℕ
  black_men : ℕ
  women_discount : ℕ
  total_employees : ℕ

/-- Calculates the total cost of t-shirts given the conditions -/
def total_cost (c : TShirtCost) : ℕ :=
  let employees_per_type := c.total_employees / 4
  let white_men_cost := c.white_men * employees_per_type
  let white_women_cost := (c.white_men - c.women_discount) * employees_per_type
  let black_men_cost := c.black_men * employees_per_type
  let black_women_cost := (c.black_men - c.women_discount) * employees_per_type
  white_men_cost + white_women_cost + black_men_cost + black_women_cost

/-- Theorem stating that the total cost of t-shirts is $660 -/
theorem total_cost_is_660 (c : TShirtCost)
  (h1 : c.white_men = 20)
  (h2 : c.black_men = 18)
  (h3 : c.women_discount = 5)
  (h4 : c.total_employees = 40) :
  total_cost c = 660 := by
  sorry

#eval total_cost { white_men := 20, black_men := 18, women_discount := 5, total_employees := 40 }

end total_cost_is_660_l4085_408528


namespace percentage_difference_l4085_408554

theorem percentage_difference (x y : ℝ) (h : y = 1.25 * x) : 
  x = 0.8 * y := by
sorry

end percentage_difference_l4085_408554


namespace root_product_sum_l4085_408526

-- Define the polynomial
def f (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 17 * x - 7

-- Define the roots
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- State the theorem
theorem root_product_sum :
  f p = 0 ∧ f q = 0 ∧ f r = 0 →
  p * q + p * r + q * r = 17 / 5 := by
  sorry

end root_product_sum_l4085_408526


namespace arithmetic_sequence_sum_l4085_408504

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

/-- Theorem: If a_3 + a_4 + a_5 + a_6 + a_7 = 20 in an arithmetic sequence, then S_9 = 36 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h : seq.a 3 + seq.a 4 + seq.a 5 + seq.a 6 + seq.a 7 = 20) :
  seq.S 9 = 36 := by
  sorry

end arithmetic_sequence_sum_l4085_408504


namespace divisor_problem_l4085_408599

theorem divisor_problem (n : ℕ+) : 
  (∃ k : ℕ, n = 2019 * k) →
  (∃ d : Fin 38 → ℕ+, 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ n) ∧
    (d 0 = 1) ∧
    (d 37 = n) ∧
    (n = d 18 * d 19)) →
  (n = 3^18 * 673 ∨ n = 673^18 * 3) := by
sorry

end divisor_problem_l4085_408599


namespace bird_fence_difference_l4085_408543

/-- Given the initial and additional numbers of sparrows and pigeons on a fence,
    and the fact that all starlings flew away, prove that there are 2 more sparrows
    than pigeons on the fence after these events. -/
theorem bird_fence_difference
  (initial_sparrows : ℕ)
  (initial_pigeons : ℕ)
  (additional_sparrows : ℕ)
  (additional_pigeons : ℕ)
  (h1 : initial_sparrows = 3)
  (h2 : initial_pigeons = 2)
  (h3 : additional_sparrows = 4)
  (h4 : additional_pigeons = 3) :
  (initial_sparrows + additional_sparrows) - (initial_pigeons + additional_pigeons) = 2 := by
  sorry

end bird_fence_difference_l4085_408543
