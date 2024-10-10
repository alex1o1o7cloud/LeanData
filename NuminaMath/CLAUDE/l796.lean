import Mathlib

namespace statement_B_is_algorithm_l796_79662

-- Define what constitutes an algorithm
def is_algorithm (statement : String) : Prop :=
  ∃ (steps : List String), steps.length > 0 ∧ steps.all (λ step => step ≠ "")

-- Define the given statements
def statement_A : String := "At home, it is generally the mother who cooks."
def statement_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def statement_C : String := "Cooking outdoors is called a picnic."
def statement_D : String := "Rice is necessary for cooking."

-- Theorem to prove
theorem statement_B_is_algorithm :
  is_algorithm statement_B ∧
  ¬is_algorithm statement_A ∧
  ¬is_algorithm statement_C ∧
  ¬is_algorithm statement_D :=
sorry

end statement_B_is_algorithm_l796_79662


namespace worm_pages_in_four_volumes_l796_79675

/-- Represents a collection of book volumes -/
structure BookCollection where
  num_volumes : ℕ
  pages_per_volume : ℕ

/-- Calculates the number of pages a worm burrows through in a book collection -/
def worm_burrowed_pages (books : BookCollection) : ℕ :=
  (books.num_volumes - 2) * books.pages_per_volume

/-- Theorem stating the number of pages a worm burrows through in a specific book collection -/
theorem worm_pages_in_four_volumes :
  let books : BookCollection := ⟨4, 200⟩
  worm_burrowed_pages books = 400 := by sorry

end worm_pages_in_four_volumes_l796_79675


namespace existence_of_mn_l796_79649

theorem existence_of_mn : ∃ (m n : ℕ), ∀ (a b : ℝ), 
  ((-2 * a^n * b^n)^m + (3 * a^m * b^m)^n) = a^6 * b^6 := by
  sorry

end existence_of_mn_l796_79649


namespace forester_count_impossible_l796_79658

/-- Represents a circle in the forest --/
structure Circle where
  id : Nat
  trees : Finset Nat

/-- Represents the forest with circles and pine trees --/
structure Forest where
  circles : Finset Circle
  total_trees : Finset Nat

/-- The property that each circle contains exactly 3 distinct trees --/
def validCount (f : Forest) : Prop :=
  ∀ c ∈ f.circles, c.trees.card = 3

/-- The property that all trees in circles are from the total set of trees --/
def validTrees (f : Forest) : Prop :=
  ∀ c ∈ f.circles, c.trees ⊆ f.total_trees

/-- The main theorem stating the impossibility of the forester's count --/
theorem forester_count_impossible (f : Forest) :
  f.circles.card = 5 → validCount f → validTrees f → False := by
  sorry

end forester_count_impossible_l796_79658


namespace b_4_lt_b_7_l796_79657

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (α 1 + 1 / b n (fun k => α (k + 1)))

theorem b_4_lt_b_7 (α : ℕ → ℕ) (h : ∀ k, α k ≥ 1) : b 4 α < b 7 α := by
  sorry

end b_4_lt_b_7_l796_79657


namespace smallest_number_proof_l796_79670

def smallest_number : ℕ := 3153

theorem smallest_number_proof :
  (∀ n : ℕ, n < smallest_number →
    ¬(((n + 3) % 18 = 0) ∧ ((n + 3) % 25 = 0) ∧ ((n + 3) % 21 = 0))) ∧
  ((smallest_number + 3) % 18 = 0) ∧
  ((smallest_number + 3) % 25 = 0) ∧
  ((smallest_number + 3) % 21 = 0) :=
by sorry

end smallest_number_proof_l796_79670


namespace football_game_ratio_l796_79636

theorem football_game_ratio : 
  -- Given conditions
  let total_start : ℕ := 600
  let girls_start : ℕ := 240
  let remaining : ℕ := 480
  let girls_left : ℕ := girls_start / 8

  -- Derived values
  let boys_start : ℕ := total_start - girls_start
  let total_left : ℕ := total_start - remaining
  let boys_left : ℕ := total_left - girls_left

  -- Theorem statement
  boys_left * 4 = boys_start :=
by sorry

end football_game_ratio_l796_79636


namespace table_capacity_l796_79641

theorem table_capacity (invited : ℕ) (no_show : ℕ) (tables : ℕ) : 
  invited = 45 → no_show = 35 → tables = 5 → (invited - no_show) / tables = 2 := by
  sorry

end table_capacity_l796_79641


namespace andrews_age_l796_79695

theorem andrews_age (grandfather_age andrew_age : ℝ) 
  (h1 : grandfather_age = 9 * andrew_age)
  (h2 : grandfather_age - andrew_age = 63) : 
  andrew_age = 7.875 := by
sorry

end andrews_age_l796_79695


namespace parabola_equation_from_hyperbola_focus_l796_79673

/-- Given a hyperbola with equation x²/4 - y²/5 = 1, prove that the standard equation
    of a parabola with its focus at the left focus of the hyperbola is y² = -12x. -/
theorem parabola_equation_from_hyperbola_focus (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) →
  ∃ (x₀ y₀ : ℝ), (x₀ = -3 ∧ y₀ = 0) ∧
    (∀ (x' y' : ℝ), y'^2 = -12 * x' ↔ 
      ((x' - x₀)^2 + (y' - y₀)^2 = (x' - (x₀ + 3/4))^2 + y'^2)) :=
by sorry

end parabola_equation_from_hyperbola_focus_l796_79673


namespace correct_arrangements_l796_79655

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- Calculates the number of arrangements given specific conditions -/
noncomputable def arrangements (condition : ℕ) : ℕ :=
  match condition with
  | 1 => 3720  -- A cannot stand at the head, and B cannot stand at the tail
  | 2 => 720   -- A, B, and C must stand next to each other
  | 3 => 1440  -- A, B, and C must not stand next to each other
  | 4 => 1200  -- There is exactly one person between A and B
  | 5 => 840   -- A, B, and C must stand in order from left to right
  | _ => 0     -- Invalid condition

/-- Theorem stating the correct number of arrangements for each condition -/
theorem correct_arrangements :
  (arrangements 1 = 3720) ∧
  (arrangements 2 = 720) ∧
  (arrangements 3 = 1440) ∧
  (arrangements 4 = 1200) ∧
  (arrangements 5 = 840) :=
by sorry

end correct_arrangements_l796_79655


namespace negation_of_forall_is_exists_not_l796_79612

variable (S : Set ℝ)

-- Define the original property
def P (x : ℝ) : Prop := x^2 = x

-- State the theorem
theorem negation_of_forall_is_exists_not (h : ∀ x ∈ S, P x) : 
  ¬(∀ x ∈ S, P x) ↔ ∃ x ∈ S, ¬(P x) := by sorry

end negation_of_forall_is_exists_not_l796_79612


namespace no_candies_to_remove_for_30_and_5_l796_79681

/-- Given a number of candies and sisters, calculate the minimum number of candies to remove for even distribution -/
def min_candies_to_remove (candies : ℕ) (sisters : ℕ) : ℕ :=
  candies % sisters

/-- Prove that for 30 candies and 5 sisters, no candies need to be removed for even distribution -/
theorem no_candies_to_remove_for_30_and_5 :
  min_candies_to_remove 30 5 = 0 := by
  sorry

#eval min_candies_to_remove 30 5

end no_candies_to_remove_for_30_and_5_l796_79681


namespace tylers_age_l796_79676

theorem tylers_age (T B S : ℕ) : 
  T = B - 3 → 
  T + B + S = 25 → 
  S = B + 1 → 
  T = 6 := by
sorry

end tylers_age_l796_79676


namespace gabriel_pages_read_l796_79611

theorem gabriel_pages_read (beatrix_pages cristobal_pages gabriel_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 3 * beatrix_pages + 15 →
  gabriel_pages = 3 * (cristobal_pages + beatrix_pages) →
  gabriel_pages = 8493 := by
  sorry

end gabriel_pages_read_l796_79611


namespace inscribed_cube_surface_area_l796_79666

theorem inscribed_cube_surface_area (outer_cube_surface_area : ℝ) :
  outer_cube_surface_area = 54 →
  ∃ (inner_cube_surface_area : ℝ),
    inner_cube_surface_area = 18 ∧
    (∃ (outer_cube_side : ℝ) (sphere_diameter : ℝ) (inner_cube_side : ℝ),
      outer_cube_side^3 = outer_cube_surface_area / 6 ∧
      sphere_diameter = outer_cube_side ∧
      inner_cube_side^2 * 3 = sphere_diameter^2 ∧
      inner_cube_surface_area = 6 * inner_cube_side^2) :=
by
  sorry

end inscribed_cube_surface_area_l796_79666


namespace unique_solution_condition_l796_79665

theorem unique_solution_condition (k : ℤ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 4) = -40 + k * x) ↔ (k = 8 ∨ k = -22) := by
  sorry

end unique_solution_condition_l796_79665


namespace mixture_volume_proportion_l796_79654

/-- Given two solutions P and Q, where P is 80% carbonated water and Q is 55% carbonated water,
    if a mixture of P and Q contains 67.5% carbonated water, then the volume of P in the mixture
    is 50% of the total volume. -/
theorem mixture_volume_proportion (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  0.80 * x + 0.55 * y = 0.675 * (x + y) →
  x / (x + y) = 1 / 2 := by
  sorry

end mixture_volume_proportion_l796_79654


namespace complex_fraction_product_l796_79632

theorem complex_fraction_product (a b : ℝ) :
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b →
  a * b = -3 := by
  sorry

end complex_fraction_product_l796_79632


namespace denominator_numerator_difference_l796_79616

/-- Represents a base-12 number as a pair of integers (numerator, denominator) -/
def Base12Fraction := ℤ × ℤ

/-- Converts a repeating decimal in base 12 to a fraction -/
def repeating_decimal_to_fraction (digits : List ℕ) : Base12Fraction := sorry

/-- Simplifies a fraction to its lowest terms -/
def simplify_fraction (f : Base12Fraction) : Base12Fraction := sorry

/-- The infinite repeating decimal 0.127127127... in base 12 -/
def G : Base12Fraction := repeating_decimal_to_fraction [1, 2, 7]

theorem denominator_numerator_difference :
  let simplified_G := simplify_fraction G
  (simplified_G.2 - simplified_G.1) = 342 := by sorry

end denominator_numerator_difference_l796_79616


namespace trailing_zeros_count_product_trailing_zeros_l796_79621

def product : ℕ := 25^7 * 8^3

theorem trailing_zeros_count (n : ℕ) : ℕ :=
  sorry

theorem product_trailing_zeros : trailing_zeros_count product = 9 := by
  sorry

end trailing_zeros_count_product_trailing_zeros_l796_79621


namespace sum_subfixed_points_ln_exp_is_zero_l796_79607

/-- A sub-fixed point of a function f is a real number t such that f(t) = -t -/
def SubFixedPoint (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

/-- The natural logarithm function -/
noncomputable def ln : ℝ → ℝ := Real.log

/-- The exponential function -/
noncomputable def exp : ℝ → ℝ := Real.exp

/-- The sub-fixed point of the natural logarithm function -/
noncomputable def t : ℝ := sorry

/-- Statement: The sum of sub-fixed points of ln and exp is zero -/
theorem sum_subfixed_points_ln_exp_is_zero :
  SubFixedPoint ln t ∧ SubFixedPoint exp (-t) → t + (-t) = 0 := by sorry

end sum_subfixed_points_ln_exp_is_zero_l796_79607


namespace smallest_divisor_square_plus_divisor_square_l796_79692

theorem smallest_divisor_square_plus_divisor_square (n : ℕ) :
  n ≥ 2 →
  (∃ k d : ℕ,
    k > 1 ∧
    k ∣ n ∧
    (∀ m : ℕ, m > 1 → m ∣ n → m ≥ k) ∧
    d ∣ n ∧
    n = k^2 + d^2) ↔
  n = 8 ∨ n = 20 :=
sorry

end smallest_divisor_square_plus_divisor_square_l796_79692


namespace propositions_B_and_C_l796_79638

theorem propositions_B_and_C :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3) * x^2 + (1/2) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) := by
  sorry

end propositions_B_and_C_l796_79638


namespace building_meets_safety_regulations_l796_79608

/-- Represents the school building configuration and safety requirements -/
structure SchoolBuilding where
  floors : Nat
  classrooms_per_floor : Nat
  main_doors : Nat
  side_doors : Nat
  students_all_doors_2min : Nat
  students_half_doors_4min : Nat
  emergency_efficiency_decrease : Rat
  evacuation_time_limit : Nat
  students_per_classroom : Nat

/-- Calculates the flow rate of students through doors -/
def calculate_flow_rates (building : SchoolBuilding) : Nat × Nat :=
  sorry

/-- Checks if the building meets safety regulations -/
def meets_safety_regulations (building : SchoolBuilding) : Bool :=
  sorry

/-- Theorem stating that the given building configuration meets safety regulations -/
theorem building_meets_safety_regulations :
  let building : SchoolBuilding := {
    floors := 4,
    classrooms_per_floor := 8,
    main_doors := 2,
    side_doors := 2,
    students_all_doors_2min := 560,
    students_half_doors_4min := 800,
    emergency_efficiency_decrease := 1/5,
    evacuation_time_limit := 5,
    students_per_classroom := 45
  }
  meets_safety_regulations building = true :=
sorry

end building_meets_safety_regulations_l796_79608


namespace log_negative_undefined_l796_79690

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_negative_undefined (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 3) :
  ¬∃ y, f a (-2) = y := by
  sorry


end log_negative_undefined_l796_79690


namespace arithmetic_sequence_problem_l796_79650

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_a5 : a 5 = 6)
  (h_a8 : a 8 = 15) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d = 3) ∧ a 11 = 24 :=
by sorry

end arithmetic_sequence_problem_l796_79650


namespace correct_yellow_balls_drawn_l796_79653

/-- Calculates the number of yellow balls to be drawn in a stratified sampling -/
def yellowBallsToDraw (totalBalls : ℕ) (yellowBalls : ℕ) (sampleSize : ℕ) : ℕ :=
  (yellowBalls * sampleSize) / totalBalls

theorem correct_yellow_balls_drawn (totalBalls : ℕ) (yellowBalls : ℕ) (sampleSize : ℕ) 
    (h1 : totalBalls = 800) 
    (h2 : yellowBalls = 40) 
    (h3 : sampleSize = 60) : 
  yellowBallsToDraw totalBalls yellowBalls sampleSize = 3 := by
  sorry

end correct_yellow_balls_drawn_l796_79653


namespace smallest_n_for_2007n_mod_1000_l796_79689

theorem smallest_n_for_2007n_mod_1000 : 
  ∀ n : ℕ+, n < 691 → (2007 * n.val) % 1000 ≠ 837 ∧ (2007 * 691) % 1000 = 837 := by
  sorry

end smallest_n_for_2007n_mod_1000_l796_79689


namespace chip_cost_theorem_l796_79634

theorem chip_cost_theorem (calories_per_chip : ℕ) (chips_per_bag : ℕ) (cost_per_bag : ℕ) (target_calories : ℕ) : 
  calories_per_chip = 10 →
  chips_per_bag = 24 →
  cost_per_bag = 2 →
  target_calories = 480 →
  (target_calories / (calories_per_chip * chips_per_bag)) * cost_per_bag = 4 := by
  sorry

end chip_cost_theorem_l796_79634


namespace part_one_part_two_l796_79642

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part_one (m : ℝ) (h : m > 0) :
  (∀ x, p x → q m x) → m ∈ Set.Ici 4 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  (p x ∨ q 5 x) ∧ ¬(p x ∧ q 5 x) →
  x ∈ Set.Ioc (-3) (-2) ∪ Set.Ioc 6 7 := by sorry

end part_one_part_two_l796_79642


namespace square_area_ratio_l796_79652

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 48) (h2 : side_D = 60) :
  (side_C ^ 2) / (side_D ^ 2) = 16 / 25 := by
  sorry

end square_area_ratio_l796_79652


namespace inequality_proof_l796_79677

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (3 : ℝ) / 2 < 1 / (a^3 + 1) + 1 / (b^3 + 1) ∧ 1 / (a^3 + 1) + 1 / (b^3 + 1) ≤ 16 / 9 := by
  sorry

end inequality_proof_l796_79677


namespace unique_solution_l796_79682

def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4) * (y.val ^ 4) - 16 * (x.val ^ 2) * (y.val ^ 2) + 15 = 0

theorem unique_solution : 
  ∃! p : ℕ+ × ℕ+, satisfies_equation p.1 p.2 :=
sorry

end unique_solution_l796_79682


namespace f_is_integer_valued_l796_79698

/-- The polynomial f(x) = (1/5)x^5 + (1/2)x^4 + (1/3)x^3 - (1/30)x -/
def f (x : ℚ) : ℚ := (1/5) * x^5 + (1/2) * x^4 + (1/3) * x^3 - (1/30) * x

/-- Theorem stating that f(x) is an integer-valued polynomial -/
theorem f_is_integer_valued : ∀ (x : ℤ), ∃ (y : ℤ), f x = y := by
  sorry

end f_is_integer_valued_l796_79698


namespace coefficient_of_x_term_l796_79696

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (x - x + 1)^3
  ∃ a b c d : ℝ, expansion = a*x^3 + b*x^2 + c*x + d ∧ c = -3 :=
by sorry

end coefficient_of_x_term_l796_79696


namespace menelaus_theorem_l796_79693

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the intersection points
def F (t : Triangle) (l : Line) : ℝ × ℝ := sorry
def D (t : Triangle) (l : Line) : ℝ × ℝ := sorry
def E (t : Triangle) (l : Line) : ℝ × ℝ := sorry

-- Define the ratio function
def ratio (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem menelaus_theorem (t : Triangle) (l : Line) :
  ratio (F t l) t.A t.B * ratio (D t l) t.B t.C * ratio (E t l) t.C t.A = -1 := by sorry

end menelaus_theorem_l796_79693


namespace leftover_money_l796_79615

/-- Calculates the leftover money after reading books and buying candy -/
theorem leftover_money
  (payment_rate : ℚ)
  (pages_per_book : ℕ)
  (books_read : ℕ)
  (candy_cost : ℚ)
  (h1 : payment_rate = 1 / 100)  -- $0.01 per page
  (h2 : pages_per_book = 150)
  (h3 : books_read = 12)
  (h4 : candy_cost = 15) :
  payment_rate * (pages_per_book * books_read : ℚ) - candy_cost = 3 := by
sorry

end leftover_money_l796_79615


namespace quadratic_real_roots_l796_79623

theorem quadratic_real_roots (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ x : ℝ, x^2 + 2*x*(a : ℝ) + 3*((b : ℝ) + (c : ℝ)) = 0 :=
sorry

end quadratic_real_roots_l796_79623


namespace symmetry_about_xoz_plane_l796_79600

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry operation about the xOz plane
def symmetryAboutXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Theorem statement
theorem symmetry_about_xoz_plane :
  let A : Point3D := { x := 3, y := -2, z := 5 }
  let A_sym : Point3D := symmetryAboutXOZ A
  A_sym = { x := 3, y := 2, z := 5 } := by sorry

end symmetry_about_xoz_plane_l796_79600


namespace sum_nonzero_digits_base8_999_l796_79669

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the non-zero elements of a list -/
def sumNonZero (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of non-zero digits in the base 8 representation of 999 is 19 -/
theorem sum_nonzero_digits_base8_999 : sumNonZero (toBase8 999) = 19 := by
  sorry

end sum_nonzero_digits_base8_999_l796_79669


namespace spinner_probability_l796_79686

theorem spinner_probability (pA pB pC pD pE : ℚ) : 
  pA = 1/3 →
  pB = 1/6 →
  pC = 2*pE →
  pD = 2*pE →
  pA + pB + pC + pD + pE = 1 →
  pE = 1/10 := by
  sorry

end spinner_probability_l796_79686


namespace kate_money_left_l796_79699

def march_savings : ℕ := 27
def april_savings : ℕ := 13
def may_savings : ℕ := 28
def keyboard_cost : ℕ := 49
def mouse_cost : ℕ := 5

def total_savings : ℕ := march_savings + april_savings + may_savings
def total_spent : ℕ := keyboard_cost + mouse_cost
def money_left : ℕ := total_savings - total_spent

theorem kate_money_left : money_left = 14 := by
  sorry

end kate_money_left_l796_79699


namespace unique_solution_quadratic_system_l796_79672

theorem unique_solution_quadratic_system :
  ∃! y : ℚ, (9 * y^2 + 8 * y - 3 = 0) ∧ (27 * y^2 + 35 * y - 12 = 0) ∧ (y = 1/3) := by
  sorry

end unique_solution_quadratic_system_l796_79672


namespace integer_equation_solution_l796_79640

theorem integer_equation_solution : 
  ∀ m n : ℕ+, m^2 + 2*n^2 = 3*(m + 2*n) ↔ (m = 3 ∧ n = 3) ∨ (m = 4 ∧ n = 2) := by
  sorry

end integer_equation_solution_l796_79640


namespace conference_duration_l796_79644

theorem conference_duration (h₁ : 9 > 0) (h₂ : 11 > 0) (h₃ : 12 > 0) :
  Nat.lcm (Nat.lcm 9 11) 12 = 396 := by
  sorry

end conference_duration_l796_79644


namespace triangle_properties_l796_79663

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = Real.pi ∧
  a / Real.sin A = 2 * c / Real.sqrt 3 ∧
  c = Real.sqrt 7 ∧
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = Real.pi / 3 ∧ a^2 + b^2 = 13 := by
sorry

end triangle_properties_l796_79663


namespace original_number_l796_79639

theorem original_number (x : ℝ) : (1.15 * (1.10 * x) = 632.5) → x = 500 := by
  sorry

end original_number_l796_79639


namespace simplify_expression_l796_79602

theorem simplify_expression : (576 : ℝ) ^ (1/4) * (216 : ℝ) ^ (1/2) = 72 := by
  sorry

end simplify_expression_l796_79602


namespace dinner_bill_split_l796_79629

theorem dinner_bill_split (total_bill : ℝ) (num_friends : ℕ) 
  (h_total_bill : total_bill = 150)
  (h_num_friends : num_friends = 6) :
  let silas_payment := total_bill / 2
  let remaining_amount := total_bill - silas_payment
  let tip := total_bill * 0.1
  let total_to_split := remaining_amount + tip
  let num_remaining_friends := num_friends - 1
  total_to_split / num_remaining_friends = 18 := by
sorry

end dinner_bill_split_l796_79629


namespace min_value_on_ellipse_l796_79606

/-- The minimum value of d for points on the given ellipse --/
theorem min_value_on_ellipse :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / 4) + (P.2^2 / 3) = 1}
  let d (P : ℝ × ℝ) := Real.sqrt (P.1^2 + P.2^2 + 4*P.2 + 4) - P.1/2
  ∀ P ∈ ellipse, d P ≥ 2 * Real.sqrt 2 - 1 ∧ ∃ Q ∈ ellipse, d Q = 2 * Real.sqrt 2 - 1 :=
by sorry


end min_value_on_ellipse_l796_79606


namespace rectangle_perimeter_l796_79659

theorem rectangle_perimeter (area : ℝ) (length width : ℝ) : 
  area = 450 ∧ length = 2 * width ∧ area = length * width → 
  2 * (length + width) = 90 := by
  sorry

end rectangle_perimeter_l796_79659


namespace ellipse_condition_l796_79661

/-- The equation of the graph is x^2 + 9y^2 - 6x + 27y = k -/
def graph_equation (x y k : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 27*y = k

/-- A non-degenerate ellipse has a positive right-hand side when in standard form -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -29.25

theorem ellipse_condition (k : ℝ) :
  (∀ x y, graph_equation x y k ↔ is_non_degenerate_ellipse k) :=
sorry

end ellipse_condition_l796_79661


namespace corrected_mean_l796_79643

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ wrong_value = 23 ∧ correct_value = 46 →
  ((n : ℚ) * original_mean - wrong_value + correct_value) / n = 36.46 := by
  sorry

end corrected_mean_l796_79643


namespace company_production_l796_79617

/-- The number of bottles produced daily by a company -/
def bottles_produced (cases_required : ℕ) (bottles_per_case : ℕ) : ℕ :=
  cases_required * bottles_per_case

/-- Theorem stating the company's daily water bottle production -/
theorem company_production :
  bottles_produced 10000 12 = 120000 := by
  sorry

end company_production_l796_79617


namespace N_mod_500_l796_79685

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The sequence of positive integers whose binary representation has exactly 7 ones -/
def S : List ℕ := sorry

/-- The 500th number in the sequence S -/
def N : ℕ := sorry

theorem N_mod_500 : N % 500 = 375 := by sorry

end N_mod_500_l796_79685


namespace problem_solution_l796_79618

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by sorry

end problem_solution_l796_79618


namespace triangle_area_l796_79674

/-- The area of a triangle with base 4 and height 5 is 10 -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 4 → 
  height = 5 → 
  area = (base * height) / 2 → 
  area = 10 := by
  sorry

end triangle_area_l796_79674


namespace gathering_drinks_l796_79651

/-- Represents the number of people who took both wine and soda at a gathering -/
def people_took_both (total : ℕ) (wine : ℕ) (soda : ℕ) : ℕ :=
  wine + soda - total

theorem gathering_drinks (total : ℕ) (wine : ℕ) (soda : ℕ) 
  (h_total : total = 31) 
  (h_wine : wine = 26) 
  (h_soda : soda = 22) :
  people_took_both total wine soda = 17 := by
  sorry

#eval people_took_both 31 26 22

end gathering_drinks_l796_79651


namespace triangle_inradius_l796_79635

/-- Given a triangle with perimeter 20 cm and area 25 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 20) 
  (h_area : A = 25) 
  (h_inradius : A = r * p / 2) : 
  r = 2.5 := by
  sorry

end triangle_inradius_l796_79635


namespace square_root_of_36_l796_79613

theorem square_root_of_36 : ∃ (x : ℝ), x^2 = 36 ↔ x = 6 ∨ x = -6 := by
  sorry

end square_root_of_36_l796_79613


namespace raft_sticks_difference_l796_79678

theorem raft_sticks_difference (simon_sticks : ℕ) (total_sticks : ℕ) : 
  simon_sticks = 36 →
  total_sticks = 129 →
  let gerry_sticks := (2 * simon_sticks) / 3
  let simon_and_gerry_sticks := simon_sticks + gerry_sticks
  let micky_sticks := total_sticks - simon_and_gerry_sticks
  micky_sticks - simon_and_gerry_sticks = 9 := by
  sorry

end raft_sticks_difference_l796_79678


namespace min_sum_pqr_l796_79664

/-- Given five positive integers with pairwise GCDs as specified, 
    the minimum sum of p, q, and r is 9 -/
theorem min_sum_pqr (a b c d e : ℕ+) 
  (h : ∃ (p q r : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
    Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
    Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
    Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
    Set.toFinset {2, 3, 4, 5, 6, 7, 8, p.val, q.val, r.val}) : 
  (∃ (p q r : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
    Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
    Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
    Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
    Set.toFinset {2, 3, 4, 5, 6, 7, 8, p.val, q.val, r.val} ∧ 
    p.val + q.val + r.val = 9 ∧ 
    ∀ (p' q' r' : ℕ+), Set.toFinset {Nat.gcd a.val b.val, Nat.gcd a.val c.val, 
      Nat.gcd a.val d.val, Nat.gcd a.val e.val, Nat.gcd b.val c.val, 
      Nat.gcd b.val d.val, Nat.gcd b.val e.val, Nat.gcd c.val d.val, 
      Nat.gcd c.val e.val, Nat.gcd d.val e.val} = 
      Set.toFinset {2, 3, 4, 5, 6, 7, 8, p'.val, q'.val, r'.val} → 
      p'.val + q'.val + r'.val ≥ 9) := by
  sorry

end min_sum_pqr_l796_79664


namespace point_distance_on_line_l796_79624

/-- Given a line with equation x - 5/2y + 1 = 0 and two points on this line,
    if the x-coordinate of the second point is 1/2 unit more than the x-coordinate of the first point,
    then the difference between their x-coordinates is 1/2. -/
theorem point_distance_on_line (m n a : ℝ) : 
  (m - (5/2) * n + 1 = 0) →  -- First point (m, n) satisfies the line equation
  (m + a - (5/2) * (n + 1) + 1 = 0) →  -- Second point (m + a, n + 1) satisfies the line equation
  (m + a = m + 1/2) →  -- x-coordinate of second point is 1/2 more than first point
  a = 1/2 := by
sorry

end point_distance_on_line_l796_79624


namespace slower_train_speed_l796_79687

theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 80) 
  (h2 : faster_speed = 52) 
  (h3 : passing_time = 36) : 
  ∃ slower_speed : ℝ, 
    slower_speed = 36 ∧ 
    (faster_speed - slower_speed) * passing_time / 3600 * 1000 = 2 * train_length :=
by sorry

end slower_train_speed_l796_79687


namespace max_x_minus_y_l796_79656

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ z :=
sorry

end max_x_minus_y_l796_79656


namespace oprah_car_collection_reduction_l796_79667

/-- The number of years required to reduce a car collection -/
def years_to_reduce (initial_cars : ℕ) (target_cars : ℕ) (cars_per_year : ℕ) : ℕ :=
  (initial_cars - target_cars) / cars_per_year

/-- Theorem: It takes 60 years to reduce Oprah's car collection from 3500 to 500 cars -/
theorem oprah_car_collection_reduction :
  years_to_reduce 3500 500 50 = 60 := by
  sorry

end oprah_car_collection_reduction_l796_79667


namespace farmer_apples_l796_79631

/-- The number of apples remaining after giving some away -/
def applesRemaining (initial : ℕ) (givenAway : ℕ) : ℕ := initial - givenAway

/-- Theorem: A farmer with 127 apples who gives away 88 apples has 39 apples remaining -/
theorem farmer_apples : applesRemaining 127 88 = 39 := by
  sorry

end farmer_apples_l796_79631


namespace subtracted_number_l796_79679

theorem subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 6 * x - y = 102) : y = 138 := by
  sorry

end subtracted_number_l796_79679


namespace geometric_sequence_fourth_term_l796_79688

theorem geometric_sequence_fourth_term
  (a : ℝ)  -- first term
  (a₆ : ℝ) -- sixth term
  (h₁ : a = 81)
  (h₂ : a₆ = 32)
  (h₃ : ∃ r : ℝ, r > 0 ∧ a₆ = a * r^5) :
  ∃ a₄ : ℝ, a₄ = 24 ∧ ∃ r : ℝ, r > 0 ∧ a₄ = a * r^3 := by
  sorry

end geometric_sequence_fourth_term_l796_79688


namespace probability_multiple_of_100_is_zero_l796_79628

def is_single_digit_multiple_of_5 (n : ℕ) : Prop :=
  n > 0 ∧ n < 10 ∧ n % 5 = 0

def is_prime_less_than_50 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 50

def is_multiple_of_100 (n : ℕ) : Prop :=
  n % 100 = 0

theorem probability_multiple_of_100_is_zero :
  ∀ (n p : ℕ), is_single_digit_multiple_of_5 n → is_prime_less_than_50 p →
  ¬(is_multiple_of_100 (n * p)) :=
sorry

end probability_multiple_of_100_is_zero_l796_79628


namespace quadratic_inequality_solution_sets_l796_79694

/-- Given that the solution set of ax² + bx + c > 0 is (-1/3, 2),
    prove that the solution set of cx² + bx + a < 0 is (-3, 1/2) -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ -3 < x ∧ x < 1/2 := by
  sorry

end quadratic_inequality_solution_sets_l796_79694


namespace number_of_b_objects_l796_79622

theorem number_of_b_objects (total : ℕ) (a : ℕ) (b : ℕ) : 
  total = 35 →
  total = a + b →
  a = 17 →
  b = 18 := by
sorry

end number_of_b_objects_l796_79622


namespace long_division_unique_solution_l796_79680

theorem long_division_unique_solution :
  ∃! (dividend divisor quotient : ℕ),
    dividend ≥ 100000 ∧ dividend < 1000000 ∧
    divisor ≥ 100 ∧ divisor < 1000 ∧
    quotient ≥ 100 ∧ quotient < 1000 ∧
    quotient % 10 = 8 ∧
    (divisor * (quotient / 100)) % 10 = 5 ∧
    dividend = divisor * quotient :=
by sorry

end long_division_unique_solution_l796_79680


namespace problem_solution_l796_79603

theorem problem_solution (x y : ℝ) (h : 2 * x = Real.log (x + y - 1) + Real.log (x - y - 1) + 4) :
  2015 * x^2 + 2016 * y^3 = 8060 := by
  sorry

end problem_solution_l796_79603


namespace mixing_ways_count_l796_79645

/-- Represents a container used in the mixing process -/
inductive Container
| Barrel : Container  -- 12-liter barrel
| Small : Container   -- 2-liter container
| Medium : Container  -- 8-liter container

/-- Represents a liquid type -/
inductive Liquid
| Wine : Liquid
| Water : Liquid

/-- Represents a mixing operation -/
structure MixingOperation :=
(source : Container)
(destination : Container)
(liquid : Liquid)
(amount : ℕ)

/-- The set of all valid mixing operations -/
def valid_operations : Set MixingOperation := sorry

/-- A mixing sequence is a list of mixing operations -/
def MixingSequence := List MixingOperation

/-- Checks if a mixing sequence results in the correct final mixture -/
def is_valid_mixture (seq : MixingSequence) : Prop := sorry

/-- The number of distinct valid mixing sequences -/
def num_valid_sequences : ℕ := sorry

/-- Main theorem: There are exactly 32 ways to mix the liquids -/
theorem mixing_ways_count :
  num_valid_sequences = 32 := by sorry

end mixing_ways_count_l796_79645


namespace bridge_length_l796_79609

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 90)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time - train_length = 285 :=
by sorry

end bridge_length_l796_79609


namespace grunters_win_probability_l796_79610

theorem grunters_win_probability (p : ℝ) (n : ℕ) (h1 : p = 2/3) (h2 : n = 6) :
  p^n = 64/729 := by
  sorry

end grunters_win_probability_l796_79610


namespace essay_competition_probability_l796_79683

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end essay_competition_probability_l796_79683


namespace snow_volume_calculation_l796_79671

/-- Calculates the total volume of snow on a sidewalk with two layers -/
theorem snow_volume_calculation 
  (length : ℝ) 
  (width : ℝ) 
  (depth1 : ℝ) 
  (depth2 : ℝ) 
  (h1 : length = 25) 
  (h2 : width = 3) 
  (h3 : depth1 = 1/3) 
  (h4 : depth2 = 1/4) : 
  length * width * depth1 + length * width * depth2 = 43.75 := by
  sorry

end snow_volume_calculation_l796_79671


namespace no_natural_solution_l796_79630

theorem no_natural_solution :
  ¬∃ (x y : ℕ), x^2 + y^2 + 1 = 6*x*y := by
  sorry

end no_natural_solution_l796_79630


namespace opposite_is_negation_l796_79604

-- Define the concept of opposite number
def opposite (a : ℝ) : ℝ := -a

-- Theorem stating that the opposite of a is -a
theorem opposite_is_negation (a : ℝ) : opposite a = -a := by
  sorry

end opposite_is_negation_l796_79604


namespace mens_wages_l796_79605

theorem mens_wages (men women boys : ℕ) (total_earnings : ℚ) : 
  men = 5 →
  5 * women = men * 8 →
  total_earnings = 60 →
  total_earnings / (3 * men) = 4 :=
by sorry

end mens_wages_l796_79605


namespace min_rows_correct_l796_79660

/-- The minimum number of rows required to seat students under given conditions -/
def min_rows (total_students : ℕ) (max_per_school : ℕ) (seats_per_row : ℕ) : ℕ :=
  -- Definition to be proved
  15

theorem min_rows_correct (total_students max_per_school seats_per_row : ℕ) 
  (h1 : total_students = 2016)
  (h2 : max_per_school = 40)
  (h3 : seats_per_row = 168)
  (h4 : ∀ (school_size : ℕ), school_size ≤ max_per_school → school_size ≤ seats_per_row) :
  min_rows total_students max_per_school seats_per_row = 15 := by
  sorry

#eval min_rows 2016 40 168

end min_rows_correct_l796_79660


namespace circle_mass_is_one_kg_l796_79626

/-- Given three balanced scales and the mass of λ, prove that the circle has a mass of 1 kg. -/
theorem circle_mass_is_one_kg (x y z : ℝ) : 
  3 * y = 2 * x →  -- First scale
  x + z = 3 * y →  -- Second scale
  2 * y = x + 1 →  -- Third scale (λ has mass 1)
  z = 1 :=         -- Mass of circle is 1 kg
by sorry

end circle_mass_is_one_kg_l796_79626


namespace kelly_string_cheese_problem_l796_79633

/-- The number of string cheeses Kelly's youngest child eats per day -/
def youngest_daily_cheese : ℕ := by sorry

theorem kelly_string_cheese_problem :
  let days_per_week : ℕ := 5
  let oldest_daily_cheese : ℕ := 2
  let cheeses_per_pack : ℕ := 30
  let weeks : ℕ := 4
  let packs_needed : ℕ := 2

  youngest_daily_cheese = 1 := by sorry

end kelly_string_cheese_problem_l796_79633


namespace one_fourth_in_one_eighth_l796_79697

theorem one_fourth_in_one_eighth : (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l796_79697


namespace board_number_equation_l796_79620

theorem board_number_equation (n : ℤ) : 7 * n + 3 = (3 * n + 7) + 84 ↔ n = 22 := by
  sorry

end board_number_equation_l796_79620


namespace work_of_two_springs_in_series_l796_79614

/-- The work required to stretch a system of two springs in series -/
theorem work_of_two_springs_in_series 
  (k₁ k₂ : Real) 
  (x : Real) 
  (h₁ : k₁ = 3000) -- 3 kN/m = 3000 N/m
  (h₂ : k₂ = 6000) -- 6 kN/m = 6000 N/m
  (h₃ : x = 0.05)  -- 5 cm = 0.05 m
  : (1/2) * (1 / (1/k₁ + 1/k₂)) * x^2 = 2.5 := by
  sorry

#check work_of_two_springs_in_series

end work_of_two_springs_in_series_l796_79614


namespace puppy_weight_l796_79668

theorem puppy_weight (puppy smaller_kitten larger_kitten : ℝ)
  (total_weight : puppy + smaller_kitten + larger_kitten = 30)
  (weight_comparison1 : puppy + larger_kitten = 3 * smaller_kitten)
  (weight_comparison2 : puppy + smaller_kitten = larger_kitten) :
  puppy = 7.5 := by
  sorry

end puppy_weight_l796_79668


namespace negation_of_existence_negation_of_proposition_l796_79619

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x + 1 > 0) := by sorry

end negation_of_existence_negation_of_proposition_l796_79619


namespace bowling_tournament_sequences_l796_79691

/-- A tournament with 6 players and 5 matches -/
structure Tournament :=
  (num_players : Nat)
  (num_matches : Nat)
  (outcomes_per_match : Nat)

/-- The number of possible prize distribution sequences in the tournament -/
def prize_sequences (t : Tournament) : Nat :=
  t.outcomes_per_match ^ t.num_matches

/-- Theorem stating that for a tournament with 6 players, 5 matches, and 2 possible outcomes per match,
    the number of possible prize distribution sequences is 32 -/
theorem bowling_tournament_sequences :
  ∀ t : Tournament, t.num_players = 6 → t.num_matches = 5 → t.outcomes_per_match = 2 →
  prize_sequences t = 32 := by
  sorry

end bowling_tournament_sequences_l796_79691


namespace remainder_of_470521_div_5_l796_79625

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := by
  sorry

end remainder_of_470521_div_5_l796_79625


namespace cafeteria_discussion_participation_l796_79648

theorem cafeteria_discussion_participation 
  (students_like : ℕ) 
  (students_dislike : ℕ) 
  (h1 : students_like = 383) 
  (h2 : students_dislike = 431) : 
  students_like + students_dislike = 814 := by
sorry

end cafeteria_discussion_participation_l796_79648


namespace tangent_line_segment_region_area_l796_79647

theorem tangent_line_segment_region_area (r : ℝ) (h : r = 3) : 
  let outer_radius := r * Real.sqrt 2
  let inner_area := π * r^2
  let outer_area := π * outer_radius^2
  outer_area - inner_area = 9 * π :=
by sorry

end tangent_line_segment_region_area_l796_79647


namespace rectangle_tiling_l796_79627

theorem rectangle_tiling (m n a b : ℕ) (hm : m > 0) (hn : n > 0) 
  (h_tiling : ∃ (h v : ℕ), a * b = h * m + v * n) :
  n ∣ a ∨ m ∣ b :=
sorry

end rectangle_tiling_l796_79627


namespace smallest_integer_satisfying_inequality_l796_79637

theorem smallest_integer_satisfying_inequality : 
  (∀ x : ℤ, x < 11 → 2*x ≥ 3*x - 10) ∧ (2*11 < 3*11 - 10) := by
  sorry

end smallest_integer_satisfying_inequality_l796_79637


namespace pump_emptying_time_l796_79684

/-- Given a pool and two pumps A and B:
    * Pump A can empty the pool in 4 hours alone
    * Pumps A and B together can empty the pool in 80 minutes
    Prove that pump B can empty the pool in 2 hours alone -/
theorem pump_emptying_time (pool : ℝ) (pump_a pump_b : ℝ → ℝ) :
  (pump_a pool = pool / 4) →  -- Pump A empties the pool in 4 hours
  (pump_a pool + pump_b pool = pool / (80 / 60)) →  -- A and B together empty the pool in 80 minutes
  (pump_b pool = pool / 2) :=  -- Pump B empties the pool in 2 hours
by sorry

end pump_emptying_time_l796_79684


namespace contractor_problem_l796_79601

theorem contractor_problem (total_days : ℕ) (absent_workers : ℕ) (actual_days : ℕ) :
  total_days = 6 →
  absent_workers = 7 →
  actual_days = 10 →
  ∃ (original_workers : ℕ), 
    original_workers * total_days = (original_workers - absent_workers) * actual_days ∧ 
    original_workers = 18 :=
by sorry

end contractor_problem_l796_79601


namespace quadratic_inequality_solution_set_l796_79646

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 2*x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
by sorry

end quadratic_inequality_solution_set_l796_79646
