import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_proof_l3367_336786

/-- The circle C with equation x^2 + y^2 + 10x + 10y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 10*x + 10*y = 0

/-- The point A with coordinates (0, 6) -/
def point_A : ℝ × ℝ := (0, 6)

/-- The desired circle passing through A and tangent to C at the origin -/
def desired_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 18

theorem circle_equation_proof :
  ∀ x y : ℝ,
  circle_C 0 0 ∧  -- C passes through the origin
  desired_circle (point_A.1) (point_A.2) ∧  -- Desired circle passes through A
  (∃ t : ℝ, t ≠ 0 ∧ 
    (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ 
      ∀ x' y' : ℝ, 
      (x' - 0)^2 + (y' - 0)^2 < δ^2 → 
      (circle_C x' y' ∧ desired_circle x' y') ∨ 
      (¬circle_C x' y' ∧ ¬desired_circle x' y'))) →  -- Tangency condition
  desired_circle x y  -- The equation of the desired circle
:= by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3367_336786


namespace NUMINAMATH_CALUDE_marys_characters_l3367_336746

theorem marys_characters (total : ℕ) (a b c d e f : ℕ) : 
  total = 120 →
  a = total / 3 →
  b = (total - a) / 4 →
  c = (total - a - b) / 5 →
  d + e + f = total - a - b - c →
  d = 3 * e →
  e = f / 2 →
  d = 24 := by sorry

end NUMINAMATH_CALUDE_marys_characters_l3367_336746


namespace NUMINAMATH_CALUDE_work_fraction_is_half_l3367_336791

/-- Represents the highway construction project -/
structure HighwayProject where
  initialWorkers : ℕ
  totalLength : ℝ
  initialDuration : ℕ
  initialDailyHours : ℕ
  completedDays : ℕ
  additionalWorkers : ℕ
  newDailyHours : ℕ

/-- Calculates the total man-hours for a given number of workers, days, and daily hours -/
def manHours (workers : ℕ) (days : ℕ) (hours : ℕ) : ℕ :=
  workers * days * hours

/-- Theorem stating that the fraction of work completed is 1/2 -/
theorem work_fraction_is_half (project : HighwayProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.totalLength = 2)
  (h3 : project.initialDuration = 50)
  (h4 : project.initialDailyHours = 8)
  (h5 : project.completedDays = 25)
  (h6 : project.additionalWorkers = 60)
  (h7 : project.newDailyHours = 10)
  (h8 : manHours (project.initialWorkers + project.additionalWorkers) 
              (project.initialDuration - project.completedDays) 
              project.newDailyHours = 
        manHours project.initialWorkers project.initialDuration project.initialDailyHours) :
  (manHours project.initialWorkers project.completedDays project.initialDailyHours : ℝ) / 
  (manHours project.initialWorkers project.initialDuration project.initialDailyHours) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_is_half_l3367_336791


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3367_336782

/-- For any real number a, the inequality x^2 - 2(a-2)x + a > 0 holds for all x ∈ (-∞, 1) ∪ (5, +∞) if and only if a ∈ (1, 5]. -/
theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ (1 < a ∧ a ≤ 5) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3367_336782


namespace NUMINAMATH_CALUDE_short_story_booklets_l3367_336720

/-- The number of booklets in Jack's short story section -/
def num_booklets : ℕ := 441 / 9

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The total number of pages Jack needs to read -/
def total_pages : ℕ := 441

theorem short_story_booklets :
  num_booklets = 49 ∧
  pages_per_booklet * num_booklets = total_pages :=
sorry

end NUMINAMATH_CALUDE_short_story_booklets_l3367_336720


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_difference_l3367_336731

theorem consecutive_odd_integers_difference (x y z : ℤ) : 
  (y = x + 2 ∧ z = y + 2) →  -- consecutive odd integers
  z = 15 →                   -- third integer is 15
  3 * x > 2 * z →            -- 3 times first is more than twice third
  3 * x - 2 * z = 3 :=       -- difference is 3
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_difference_l3367_336731


namespace NUMINAMATH_CALUDE_increase_in_average_age_increase_in_average_age_is_two_l3367_336718

/-- The increase in average age when replacing two men with two women in a group -/
theorem increase_in_average_age : ℝ :=
  let initial_group_size : ℕ := 8
  let replaced_men_ages : List ℝ := [20, 24]
  let women_average_age : ℝ := 30
  let total_age_increase : ℝ := 2 * women_average_age - replaced_men_ages.sum
  total_age_increase / initial_group_size

/-- Proof that the increase in average age is 2 years -/
theorem increase_in_average_age_is_two :
  increase_in_average_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_age_increase_in_average_age_is_two_l3367_336718


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3367_336713

/-- A regular hexagon with a line segment connecting opposite vertices. -/
structure RegularHexagon :=
  (side_length : ℝ)
  (center_to_midpoint : ℝ)

/-- Theorem: If the distance from the center to the midpoint of a line segment
    connecting opposite vertices in a regular hexagon is 9, then the side length
    is 6√3. -/
theorem hexagon_side_length (h : RegularHexagon) 
    (h_center_to_midpoint : h.center_to_midpoint = 9) : 
    h.side_length = 6 * Real.sqrt 3 := by
  sorry

#check hexagon_side_length

end NUMINAMATH_CALUDE_hexagon_side_length_l3367_336713


namespace NUMINAMATH_CALUDE_g_of_5_eq_neg_7_l3367_336774

/-- The polynomial function g(x) -/
def g (x : ℝ) : ℝ := 2 * x^4 - 15 * x^3 + 24 * x^2 - 18 * x - 72

/-- Theorem: g(5) equals -7 -/
theorem g_of_5_eq_neg_7 : g 5 = -7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_eq_neg_7_l3367_336774


namespace NUMINAMATH_CALUDE_g_2010_value_l3367_336715

-- Define the property of the function g
def g_property (g : ℕ → ℝ) : Prop :=
  ∀ x y m : ℕ, x > 0 → y > 0 → m > 0 → x + y = 2^m → g x + g y = ((m + 1) : ℝ)^2

-- Theorem statement
theorem g_2010_value (g : ℕ → ℝ) (h : g_property g) : g 2010 = 126 := by
  sorry

end NUMINAMATH_CALUDE_g_2010_value_l3367_336715


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l3367_336726

/-- Surface area of a cuboid -/
def surface_area (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 10 cm, and height 12 cm is 592 square centimeters -/
theorem cuboid_surface_area :
  surface_area 8 10 12 = 592 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l3367_336726


namespace NUMINAMATH_CALUDE_choir_composition_l3367_336735

theorem choir_composition (initial_total : ℕ) : 
  let initial_girls : ℕ := (6 * initial_total) / 10
  let final_total : ℕ := initial_total + 6 - 4 - 2
  let final_girls : ℕ := initial_girls - 4
  (2 * final_girls = final_total) → initial_girls = 24 := by
sorry

end NUMINAMATH_CALUDE_choir_composition_l3367_336735


namespace NUMINAMATH_CALUDE_sam_distance_l3367_336777

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam drove for 4 hours
    at the same average rate as Marguerite, prove that Sam drove 200 miles. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
    (h1 : marguerite_distance = 150)
    (h2 : marguerite_time = 3)
    (h3 : sam_time = 4) :
    let marguerite_rate := marguerite_distance / marguerite_time
    marguerite_rate * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l3367_336777


namespace NUMINAMATH_CALUDE_ad_ratio_l3367_336729

/-- Represents the number of ads on each web page -/
structure WebPages :=
  (page1 : ℕ)
  (page2 : ℕ)
  (page3 : ℕ)
  (page4 : ℕ)

/-- Conditions of the problem -/
def adConditions (w : WebPages) : Prop :=
  w.page1 = 12 ∧
  w.page2 = 2 * w.page1 ∧
  w.page3 = w.page2 + 24 ∧
  2 * 68 = 3 * (w.page1 + w.page2 + w.page3 + w.page4)

/-- The theorem to be proved -/
theorem ad_ratio (w : WebPages) :
  adConditions w →
  (w.page4 : ℚ) / w.page2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ad_ratio_l3367_336729


namespace NUMINAMATH_CALUDE_square_binomial_identity_l3367_336756

theorem square_binomial_identity : (1/2)^2 + 2*(1/2)*5 + 5^2 = 121/4 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_identity_l3367_336756


namespace NUMINAMATH_CALUDE_total_cost_proof_l3367_336742

def bow_cost : ℕ := 5
def vinegar_cost : ℕ := 2
def baking_soda_cost : ℕ := 1
def num_students : ℕ := 23

def total_cost_per_student : ℕ := bow_cost + vinegar_cost + baking_soda_cost

theorem total_cost_proof : 
  total_cost_per_student * num_students = 184 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_proof_l3367_336742


namespace NUMINAMATH_CALUDE_cosine_squared_inequality_l3367_336728

theorem cosine_squared_inequality (x y : ℝ) : 
  (Real.cos (x - y))^2 ≤ 4 * (1 - Real.sin x * Real.cos y) * (1 - Real.cos x * Real.sin y) := by
  sorry

end NUMINAMATH_CALUDE_cosine_squared_inequality_l3367_336728


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l3367_336739

theorem matrix_sum_theorem (x y z k : ℝ) 
  (h1 : x * (x^2 - y*z) - y * (z^2 - y*x) + z * (z*x - y^2) = 0)
  (h2 : x + y + z = k)
  (h3 : y + z ≠ k)
  (h4 : z + x ≠ k)
  (h5 : x + y ≠ k) :
  x / (y + z - k) + y / (z + x - k) + z / (x + y - k) = -3 := by
sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l3367_336739


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l3367_336796

theorem rationalize_and_simplify :
  (Real.sqrt 18) / (Real.sqrt 9 - Real.sqrt 3) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l3367_336796


namespace NUMINAMATH_CALUDE_test_questions_l3367_336710

theorem test_questions (Q : ℝ) : 
  (0.9 * (Q / 2) + 0.95 * (Q / 2) = 74) → Q = 80 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l3367_336710


namespace NUMINAMATH_CALUDE_simplify_absolute_value_sum_l3367_336783

theorem simplify_absolute_value_sum (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2*b + 5| + |-3*a + 2*b - 2| = 4*a - 4*b + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_sum_l3367_336783


namespace NUMINAMATH_CALUDE_tara_book_sales_tara_clarinet_purchase_l3367_336770

theorem tara_book_sales (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) : ℕ :=
  let halfway_goal := clarinet_cost / 2
  let books_to_halfway := (halfway_goal - initial_savings) / book_price
  let books_to_full_goal := clarinet_cost / book_price
  books_to_halfway + books_to_full_goal

theorem tara_clarinet_purchase : tara_book_sales 10 90 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_tara_book_sales_tara_clarinet_purchase_l3367_336770


namespace NUMINAMATH_CALUDE_gcf_of_48_and_14_l3367_336784

theorem gcf_of_48_and_14 :
  let n : ℕ := 48
  let m : ℕ := 14
  let lcm_nm : ℕ := 56
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 12 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_48_and_14_l3367_336784


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3367_336737

theorem complex_equation_solution (a b : ℝ) : 
  (b : ℂ) + 5*I = 9 - a + a*I → b = 6 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3367_336737


namespace NUMINAMATH_CALUDE_muffin_count_l3367_336798

/-- Given a ratio of doughnuts to cookies to muffins and the number of doughnuts,
    calculate the number of muffins. -/
def calculate_muffins (doughnut_ratio : ℕ) (cookie_ratio : ℕ) (muffin_ratio : ℕ) (num_doughnuts : ℕ) : ℕ :=
  (num_doughnuts / doughnut_ratio) * muffin_ratio

/-- Theorem stating that given the ratio 5:3:1 for doughnuts:cookies:muffins
    and 50 doughnuts, there are 10 muffins. -/
theorem muffin_count : calculate_muffins 5 3 1 50 = 10 := by
  sorry

#eval calculate_muffins 5 3 1 50

end NUMINAMATH_CALUDE_muffin_count_l3367_336798


namespace NUMINAMATH_CALUDE_consecutive_sum_26_l3367_336733

theorem consecutive_sum_26 (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 26 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_26_l3367_336733


namespace NUMINAMATH_CALUDE_no_integer_roots_l3367_336701

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3367_336701


namespace NUMINAMATH_CALUDE_number_problem_l3367_336769

theorem number_problem (N : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 17) 
  (h2 : Real.sqrt (0.6 * N) = (N^(1/3)) / 2) : 
  0.4 * N = 204 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3367_336769


namespace NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l3367_336748

theorem divisible_by_seven_last_digit :
  ∀ d : ℕ, d < 10 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l3367_336748


namespace NUMINAMATH_CALUDE_constant_term_value_l3367_336779

theorem constant_term_value (x y z : ℤ) (k : ℤ) 
  (eq1 : 4 * x + y + z = 80)
  (eq2 : 2 * x - y - z = 40)
  (eq3 : 3 * x + y - z = k)
  (h_x : x = 20) : k = 60 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l3367_336779


namespace NUMINAMATH_CALUDE_power_of_two_equation_l3367_336780

theorem power_of_two_equation (k : ℤ) : 
  2^1998 - 2^1997 - 2^1996 + 2^1995 = k * 2^1995 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l3367_336780


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3367_336743

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ 72 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    2 * x₀^2 + 8 * x₀ * y₀ + 32 * y₀^2 + 16 * y₀ * z₀ + 8 * z₀^2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3367_336743


namespace NUMINAMATH_CALUDE_range_of_a_for_C_subset_B_l3367_336793

def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem range_of_a_for_C_subset_B :
  {a : ℝ | C a ⊆ B} = {a : ℝ | 2 ≤ a ∧ a ≤ 8} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_C_subset_B_l3367_336793


namespace NUMINAMATH_CALUDE_power_product_equality_l3367_336767

theorem power_product_equality (a b : ℝ) : a^3 * b^3 = (a*b)^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3367_336767


namespace NUMINAMATH_CALUDE_protest_jail_time_l3367_336732

/-- Calculates the total combined weeks of jail time given protest conditions --/
theorem protest_jail_time 
  (days_of_protest : ℕ) 
  (num_cities : ℕ) 
  (arrests_per_day_per_city : ℕ) 
  (days_in_jail_before_trial : ℕ) 
  (sentence_weeks : ℕ) 
  (h1 : days_of_protest = 30)
  (h2 : num_cities = 21)
  (h3 : arrests_per_day_per_city = 10)
  (h4 : days_in_jail_before_trial = 4)
  (h5 : sentence_weeks = 2) :
  (days_of_protest * num_cities * arrests_per_day_per_city * days_in_jail_before_trial) / 7 +
  (days_of_protest * num_cities * arrests_per_day_per_city * sentence_weeks) / 2 = 9900 := by
  sorry


end NUMINAMATH_CALUDE_protest_jail_time_l3367_336732


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3367_336722

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^3 + 2*z^2 + z - 2) / (z^2 - 3*z + 2)
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, f z = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3367_336722


namespace NUMINAMATH_CALUDE_communication_arrangement_l3367_336704

def letter_arrangement (n : ℕ) (triple : ℕ) (double : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial 3 * (Nat.factorial 2)^double * Nat.factorial (n - triple - 2*double))

theorem communication_arrangement :
  letter_arrangement 14 1 2 = 908107825 := by
  sorry

end NUMINAMATH_CALUDE_communication_arrangement_l3367_336704


namespace NUMINAMATH_CALUDE_factor_expression_l3367_336741

theorem factor_expression (x : ℝ) : 72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3367_336741


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l3367_336747

/-- Permutation function -/
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The problem statement -/
theorem permutation_equation_solution :
  ∃! (x : ℕ), x > 0 ∧ x ≤ 5 ∧ A 5 x = 2 * A 6 (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l3367_336747


namespace NUMINAMATH_CALUDE_unique_valid_pair_l3367_336745

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (Nat.factorial a - a * b) % 10 = 2

theorem unique_valid_pair : ∃! p : ℕ × ℕ, is_valid_pair p.1 p.2 :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_pair_l3367_336745


namespace NUMINAMATH_CALUDE_gift_wrap_sales_l3367_336749

/-- Proves that the total number of gift wrap rolls sold is 480 given the specified conditions -/
theorem gift_wrap_sales (solid_price print_price total_amount print_rolls : ℚ)
  (h1 : solid_price = 4)
  (h2 : print_price = 6)
  (h3 : total_amount = 2340)
  (h4 : print_rolls = 210)
  (h5 : ∃ solid_rolls : ℚ, solid_price * solid_rolls + print_price * print_rolls = total_amount) :
  ∃ total_rolls : ℚ, total_rolls = 480 ∧ 
    ∃ solid_rolls : ℚ, total_rolls = solid_rolls + print_rolls ∧
    solid_price * solid_rolls + print_price * print_rolls = total_amount := by
  sorry


end NUMINAMATH_CALUDE_gift_wrap_sales_l3367_336749


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l3367_336754

theorem partial_fraction_sum (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 1 / 30 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l3367_336754


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l3367_336709

theorem simplify_radical_expression :
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l3367_336709


namespace NUMINAMATH_CALUDE_cistern_length_l3367_336789

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: A cistern with given dimensions has a length of 4 meters --/
theorem cistern_length : 
  ∃ (c : Cistern), c.width = 8 ∧ c.depth = 1.25 ∧ wetSurfaceArea c = 62 → c.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_l3367_336789


namespace NUMINAMATH_CALUDE_cube_square_fraction_inequality_l3367_336768

theorem cube_square_fraction_inequality (s r : ℝ) (hs : s > 0) (hr : r > 0) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_square_fraction_inequality_l3367_336768


namespace NUMINAMATH_CALUDE_fencing_cost_is_5300_l3367_336700

/-- The cost of fencing per meter -/
def fencing_cost_per_meter : ℝ := 26.50

/-- The length of the rectangular plot in meters -/
def plot_length : ℝ := 57

/-- Calculate the breadth of the plot given the length -/
def plot_breadth : ℝ := plot_length - 14

/-- Calculate the perimeter of the rectangular plot -/
def plot_perimeter : ℝ := 2 * (plot_length + plot_breadth)

/-- Calculate the total cost of fencing the plot -/
def total_fencing_cost : ℝ := plot_perimeter * fencing_cost_per_meter

/-- Theorem stating that the total cost of fencing is 5300 currency units -/
theorem fencing_cost_is_5300 : total_fencing_cost = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_5300_l3367_336700


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3367_336727

/-- The quadratic function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The vertex of f(x) -/
def vertex : ℝ × ℝ := (1, -1)

theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3367_336727


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l3367_336724

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x * x^(1/3))^(1/4) = x^(1/3) := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l3367_336724


namespace NUMINAMATH_CALUDE_area_ABCD_less_than_one_l3367_336790

-- Define the quadrilateral ABCD
variable (A B C D M P Q : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def diagonals_intersect_at (A B C D M : ℝ × ℝ) : Prop := sorry

def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

def is_midpoint (P X Y : ℝ × ℝ) : Prop := sorry

def distance (X Y : ℝ × ℝ) : ℝ := sorry

def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ABCD_less_than_one
  (h_convex : is_convex_quadrilateral A B C D)
  (h_diagonals : diagonals_intersect_at A B C D M)
  (h_area : area_triangle A D M > area_triangle B C M)
  (h_midpoint_P : is_midpoint P B C)
  (h_midpoint_Q : is_midpoint Q A D)
  (h_distance : distance A P + distance A Q = Real.sqrt 2) :
  area_quadrilateral A B C D < 1 := by sorry

end NUMINAMATH_CALUDE_area_ABCD_less_than_one_l3367_336790


namespace NUMINAMATH_CALUDE_max_even_integer_quadratic_inequality_l3367_336787

theorem max_even_integer_quadratic_inequality :
  (∃ a : ℤ, a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0) →
  (∀ a : ℤ, a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0 → a ≤ 8) ∧
  (∃ a : ℤ, a = 8 ∧ a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_even_integer_quadratic_inequality_l3367_336787


namespace NUMINAMATH_CALUDE_parallel_condition_l3367_336734

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The vector a as a function of k -/
def a (k : ℝ) : ℝ × ℝ := (k^2, k + 1)

/-- The vector b as a function of k -/
def b (k : ℝ) : ℝ × ℝ := (k, 4)

/-- Theorem stating the conditions for parallelism of vectors a and b -/
theorem parallel_condition (k : ℝ) : 
  are_parallel (a k) (b k) ↔ k = 0 ∨ k = 1/3 := by sorry

end NUMINAMATH_CALUDE_parallel_condition_l3367_336734


namespace NUMINAMATH_CALUDE_existence_of_sequence_l3367_336712

theorem existence_of_sequence : ∃ (s : List ℕ), 
  (s.length > 10) ∧ 
  (s.sum = 20) ∧ 
  (∀ (i j : ℕ), i ≤ j → j < s.length → (s.take (j + 1)).drop i ≠ [3]) :=
sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l3367_336712


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3367_336721

theorem complex_equation_solution :
  ∃ z : ℂ, (z - Complex.I) * Complex.I = 2 + Complex.I ∧ z = 1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3367_336721


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_80_l3367_336765

theorem largest_multiple_of_8_with_negation_greater_than_neg_80 : 
  ∀ n : ℤ, (∃ k : ℤ, n = 8 * k) → -n > -80 → n ≤ 72 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_80_l3367_336765


namespace NUMINAMATH_CALUDE_platform_length_l3367_336773

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 26 seconds to cross a signal pole, prove that the length of the platform is 150 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 26) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 150 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3367_336773


namespace NUMINAMATH_CALUDE_regular_tetrahedron_side_edge_length_l3367_336744

/-- A regular triangular pyramid (tetrahedron) with base edge length 1 and side faces forming 120° angles -/
structure RegularTetrahedron where
  base_edge : ℝ
  face_angle : ℝ
  side_edge : ℝ
  base_edge_is_unit : base_edge = 1
  face_angle_is_120 : face_angle = 2 * Real.pi / 3

/-- The side edge length of a regular tetrahedron with unit base edge and 120° face angles is √6/4 -/
theorem regular_tetrahedron_side_edge_length (t : RegularTetrahedron) : t.side_edge = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_side_edge_length_l3367_336744


namespace NUMINAMATH_CALUDE_max_cross_section_area_l3367_336760

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  base : List Point3D
  heights : List ℝ

def crossSectionArea (prism : TriangularPrism) (plane : Plane) : ℝ := sorry

/-- The main theorem statement -/
theorem max_cross_section_area :
  let prism : TriangularPrism := {
    base := [
      { x := 4, y := 0, z := 0 },
      { x := -2, y := 2 * Real.sqrt 3, z := 0 },
      { x := -2, y := -2 * Real.sqrt 3, z := 0 }
    ],
    heights := [2, 4, 3]
  }
  let plane : Plane := { a := 5, b := -3, c := 2, d := 30 }
  let area := crossSectionArea prism plane
  ∃ ε > 0, abs (area - 104.25) < ε := by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l3367_336760


namespace NUMINAMATH_CALUDE_constant_function_sqrt_l3367_336794

/-- Given a function f that is constant 3 for all real inputs, 
    prove that f(√x) + 1 = 4 for all non-negative real x -/
theorem constant_function_sqrt (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3) :
  ∀ x : ℝ, x ≥ 0 → f (Real.sqrt x) + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_sqrt_l3367_336794


namespace NUMINAMATH_CALUDE_bus_fraction_is_two_thirds_l3367_336730

def total_distance : ℝ := 30.000000000000007

theorem bus_fraction_is_two_thirds :
  let foot_distance := (1 / 5 : ℝ) * total_distance
  let car_distance := 4
  let bus_distance := total_distance - (foot_distance + car_distance)
  bus_distance / total_distance = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_bus_fraction_is_two_thirds_l3367_336730


namespace NUMINAMATH_CALUDE_isabella_exchange_l3367_336764

/-- Exchange rate from U.S. dollars to Euros -/
def exchange_rate : ℚ := 5 / 8

/-- The amount of Euros Isabella spent -/
def euros_spent : ℕ := 80

theorem isabella_exchange (d : ℕ) : 
  (exchange_rate * d : ℚ) - euros_spent = 2 * d → d = 58 := by
  sorry

end NUMINAMATH_CALUDE_isabella_exchange_l3367_336764


namespace NUMINAMATH_CALUDE_mary_money_left_l3367_336766

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 2 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that the amount of money Mary has left is 50 - 10p -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 10 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l3367_336766


namespace NUMINAMATH_CALUDE_composite_rectangle_theorem_l3367_336705

/-- The side length of square S2 in the composite rectangle. -/
def side_length_S2 : ℕ := 775

/-- The width of the composite rectangle. -/
def total_width : ℕ := 4000

/-- The height of the composite rectangle. -/
def total_height : ℕ := 2450

/-- The shorter side length of rectangles R1 and R2. -/
def shorter_side_R : ℕ := (total_height - side_length_S2) / 2

theorem composite_rectangle_theorem :
  (2 * shorter_side_R + side_length_S2 = total_height) ∧
  (2 * shorter_side_R + 3 * side_length_S2 = total_width) := by
  sorry

#check composite_rectangle_theorem

end NUMINAMATH_CALUDE_composite_rectangle_theorem_l3367_336705


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3367_336758

theorem complex_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3367_336758


namespace NUMINAMATH_CALUDE_time_to_empty_tank_l3367_336725

/-- Represents the volume of the tank in cubic feet -/
def tank_volume : ℝ := 30

/-- Represents the rate of the inlet pipe in cubic inches per minute -/
def inlet_rate : ℝ := 3

/-- Represents the rate of the first outlet pipe in cubic inches per minute -/
def outlet_rate_1 : ℝ := 12

/-- Represents the rate of the second outlet pipe in cubic inches per minute -/
def outlet_rate_2 : ℝ := 6

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Theorem stating the time to empty the tank -/
theorem time_to_empty_tank :
  let tank_volume_inches := tank_volume * feet_to_inches * feet_to_inches * feet_to_inches
  let net_emptying_rate := outlet_rate_1 + outlet_rate_2 - inlet_rate
  tank_volume_inches / net_emptying_rate = 3456 := by
  sorry


end NUMINAMATH_CALUDE_time_to_empty_tank_l3367_336725


namespace NUMINAMATH_CALUDE_all_points_on_line_l3367_336716

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def isOnLine (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem all_points_on_line :
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨-2, -4⟩
  let points : List Point := [⟨1, 2⟩, ⟨0, 0⟩, ⟨2, 4⟩, ⟨5, 10⟩, ⟨-1, -2⟩]
  ∀ p ∈ points, isOnLine p p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_all_points_on_line_l3367_336716


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3367_336761

theorem other_root_of_quadratic (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3367_336761


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l3367_336702

theorem multiplication_division_equality : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l3367_336702


namespace NUMINAMATH_CALUDE_point_in_region_implies_a_negative_l3367_336717

theorem point_in_region_implies_a_negative (a : ℝ) :
  (2 * a + 3 < 3) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_implies_a_negative_l3367_336717


namespace NUMINAMATH_CALUDE_cake_surface_area_change_l3367_336799

/-- The change in surface area of a cylindrical cake after removing a cube from its top center -/
theorem cake_surface_area_change 
  (h : ℝ) -- height of the cylinder
  (r : ℝ) -- radius of the cylinder
  (s : ℝ) -- side length of the cube
  (h_pos : h > 0)
  (r_pos : r > 0)
  (s_pos : s > 0)
  (h_val : h = 5)
  (r_val : r = 2)
  (s_val : s = 1)
  (h_ge_s : h ≥ s) -- ensure the cube fits in the cylinder
  : (2 * π * r * s + s^2) - s^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_cake_surface_area_change_l3367_336799


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3367_336763

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3367_336763


namespace NUMINAMATH_CALUDE_probability_theorem_l3367_336785

/-- The number of boys in the group -/
def num_boys : ℕ := 5

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one girl -/
def prob_one_girl : ℚ := 15 / 28

/-- The probability of selecting exactly one girl given that at least one girl is selected -/
def prob_one_girl_given_at_least_one : ℚ := 5 / 6

/-- Theorem stating the probabilities for the given scenario -/
theorem probability_theorem :
  (prob_one_girl = 15 / 28) ∧
  (prob_one_girl_given_at_least_one = 5 / 6) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l3367_336785


namespace NUMINAMATH_CALUDE_exists_b_for_234_quadrants_l3367_336706

-- Define the linear function
def f (b : ℝ) (x : ℝ) : ℝ := -2 * x + b

-- Define the property of passing through the second, third, and fourth quadrants
def passes_through_234_quadrants (b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ,
    (x₁ < 0 ∧ f b x₁ > 0) ∧  -- Second quadrant
    (x₂ < 0 ∧ f b x₂ < 0) ∧  -- Third quadrant
    (x₃ > 0 ∧ f b x₃ < 0)    -- Fourth quadrant

-- Theorem statement
theorem exists_b_for_234_quadrants :
  ∃ b : ℝ, b < 0 ∧ passes_through_234_quadrants b :=
sorry

end NUMINAMATH_CALUDE_exists_b_for_234_quadrants_l3367_336706


namespace NUMINAMATH_CALUDE_article_pricing_gain_l3367_336753

/-- Proves that if selling an article at 2/3 of its original price results in a 10% loss,
    then selling it at the original price results in a 35% gain. -/
theorem article_pricing_gain (P : ℝ) (P_pos : P > 0) :
  (2 / 3 : ℝ) * P = (9 / 10 : ℝ) * ((20 / 27 : ℝ) * P) →
  ((P - (20 / 27 : ℝ) * P) / ((20 / 27 : ℝ) * P)) * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_article_pricing_gain_l3367_336753


namespace NUMINAMATH_CALUDE_smallest_positive_b_squared_l3367_336762

/-- Definition of circle u₁ -/
def u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0

/-- Definition of circle u₂ -/
def u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 175 = 0

/-- A circle is externally tangent to u₂ and internally tangent to u₁ -/
def is_tangent_circle (x y t : ℝ) : Prop :=
  t + 7 = Real.sqrt ((x - 4)^2 + (y - 10)^2) ∧
  11 - t = Real.sqrt ((x + 4)^2 + (y - 10)^2)

/-- The center of the tangent circle lies on the line y = bx -/
def center_on_line (x y b : ℝ) : Prop := y = b * x

/-- Main theorem: The smallest positive b satisfying the conditions has b² = 5/16 -/
theorem smallest_positive_b_squared (b : ℝ) :
  (∃ x y t, u₁ x y ∧ u₂ x y ∧ is_tangent_circle x y t ∧ center_on_line x y b) →
  (∀ b' : ℝ, 0 < b' → b' < b →
    ¬∃ x y t, u₁ x y ∧ u₂ x y ∧ is_tangent_circle x y t ∧ center_on_line x y b') →
  b^2 = 5/16 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_squared_l3367_336762


namespace NUMINAMATH_CALUDE_license_plate_count_l3367_336703

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of possible odd digits for the first digit position -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits for the second digit position -/
def num_even_digits : ℕ := 5

/-- The number of possible digits for the third digit position -/
def num_all_digits : ℕ := 10

/-- The total number of possible license plates under the given conditions -/
def total_license_plates : ℕ := num_letters^3 * num_odd_digits * num_even_digits * num_all_digits

theorem license_plate_count : total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3367_336703


namespace NUMINAMATH_CALUDE_infinite_non_prime_generating_numbers_l3367_336719

theorem infinite_non_prime_generating_numbers :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, m > 1 → ∀ n : ℕ, ¬ Nat.Prime (n^4 + f m) := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_prime_generating_numbers_l3367_336719


namespace NUMINAMATH_CALUDE_tournament_max_wins_l3367_336708

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Minimum number of participants required for n wins -/
def f (n : ℕ) : ℕ := fib (n + 2)

/-- Tournament properties -/
structure Tournament :=
  (participants : ℕ)
  (one_match_at_a_time : Bool)
  (loser_drops_out : Bool)
  (max_win_diff : ℕ)

/-- Main theorem -/
theorem tournament_max_wins (t : Tournament) (h1 : t.participants = 55) 
  (h2 : t.one_match_at_a_time = true) (h3 : t.loser_drops_out = true) 
  (h4 : t.max_win_diff = 1) : 
  (∃ (n : ℕ), f n ≤ t.participants ∧ f (n + 1) > t.participants ∧ n = 8) :=
sorry

end NUMINAMATH_CALUDE_tournament_max_wins_l3367_336708


namespace NUMINAMATH_CALUDE_last_four_average_l3367_336755

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 70 →
  ((list.take 3).sum / 3 : ℝ) = 65 →
  ((list.drop 3).sum / 4 : ℝ) = 73.75 := by
  sorry

end NUMINAMATH_CALUDE_last_four_average_l3367_336755


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3367_336736

theorem arithmetic_calculations :
  (4 + (-7) - (-5) = 2) ∧
  (-1^2023 + 27 * (-1/3)^2 - |(-5)| = -3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3367_336736


namespace NUMINAMATH_CALUDE_ratio_and_average_theorem_l3367_336776

theorem ratio_and_average_theorem (a b c d : ℕ+) : 
  (a : ℚ) / b = 2 / 3 ∧ 
  (b : ℚ) / c = 3 / 4 ∧ 
  (c : ℚ) / d = 4 / 5 ∧ 
  (a + b + c + d : ℚ) / 4 = 42 →
  a = 24 := by sorry

end NUMINAMATH_CALUDE_ratio_and_average_theorem_l3367_336776


namespace NUMINAMATH_CALUDE_wall_building_time_l3367_336757

theorem wall_building_time (avery_time tom_time : ℝ) : 
  avery_time = 4 →
  1 / avery_time + 1 / tom_time + 0.5 / tom_time = 1 →
  tom_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l3367_336757


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_and_f_properties_l3367_336752

/-- Triangle ABC with base AB and height 1 -/
structure Triangle :=
  (base : ℝ)
  (height : ℝ)
  (height_eq_one : height = 1)

/-- Rectangle PQRS with width PQ and height 1 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (height_eq_one : height = 1)

/-- The function f(x) representing the height of rectangle PQNM -/
def f (x : ℝ) : ℝ := 2 * x - x^2

theorem triangle_rectangle_ratio_and_f_properties 
  (triangle : Triangle) (rectangle : Rectangle) :
  (triangle.base / rectangle.width = 2) ∧
  (triangle.base * triangle.height / 2 = rectangle.width * rectangle.height) ∧
  (f (1/2) = 3/4) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
    f x = 2 * x - x^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_and_f_properties_l3367_336752


namespace NUMINAMATH_CALUDE_inequality_proof_l3367_336788

theorem inequality_proof (α β γ : ℝ) : 1 - Real.sin (α / 2) ≥ 2 * Real.sin (β / 2) * Real.sin (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3367_336788


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3367_336772

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 30 → 2 * y - 3 * x = 5 → |y - x| = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3367_336772


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_five_satisfies_inequality_least_n_is_five_l3367_336778

theorem least_n_satisfying_inequality :
  ∀ n : ℕ+, n < 5 → (1 : ℚ) / n.val - (1 : ℚ) / (n.val + 2) ≥ (1 : ℚ) / 15 :=
by sorry

theorem five_satisfies_inequality :
  (1 : ℚ) / 5 - (1 : ℚ) / 7 < (1 : ℚ) / 15 :=
by sorry

theorem least_n_is_five :
  ∃! (n : ℕ+), 
    ((1 : ℚ) / n.val - (1 : ℚ) / (n.val + 2) < (1 : ℚ) / 15) ∧
    (∀ m : ℕ+, m < n → (1 : ℚ) / m.val - (1 : ℚ) / (m.val + 2) ≥ (1 : ℚ) / 15) :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_five_satisfies_inequality_least_n_is_five_l3367_336778


namespace NUMINAMATH_CALUDE_parabola_max_value_implies_a_greater_than_two_l3367_336775

/-- Given a parabola y = (2-a)x^2 + 3x - 2, if it has a maximum value, then a > 2 -/
theorem parabola_max_value_implies_a_greater_than_two (a : ℝ) :
  (∃ (y_max : ℝ), ∀ (x : ℝ), (2 - a) * x^2 + 3 * x - 2 ≤ y_max) →
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_max_value_implies_a_greater_than_two_l3367_336775


namespace NUMINAMATH_CALUDE_distribute_4_3_l3367_336750

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 36 ways to distribute 4 distinct objects
    into 3 distinct containers, with each container having at least one object. -/
theorem distribute_4_3 : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_4_3_l3367_336750


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3367_336707

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3367_336707


namespace NUMINAMATH_CALUDE_min_sum_squares_l3367_336738

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 3 ∧ (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3367_336738


namespace NUMINAMATH_CALUDE_valid_partition_exists_l3367_336759

/-- Represents a person in the group -/
structure Person where
  id : Nat

/-- Represents the friendship and enmity relations in the group -/
structure Relations (P : Type) where
  friend : P → P
  enemy : P → P

/-- Represents a partition of the group into two subsets -/
structure Partition (P : Type) where
  set1 : Set P
  set2 : Set P
  partition_complete : set1 ∪ set2 = Set.univ
  partition_disjoint : set1 ∩ set2 = ∅

/-- The main theorem stating that a valid partition exists -/
theorem valid_partition_exists (P : Type) [Finite P] (r : Relations P) 
  (friend_injective : Function.Injective r.friend)
  (enemy_injective : Function.Injective r.enemy)
  (friend_enemy_distinct : ∀ p : P, r.friend p ≠ r.enemy p) :
  ∃ (part : Partition P), 
    (∀ p ∈ part.set1, r.friend p ∉ part.set1 ∧ r.enemy p ∉ part.set1) ∧
    (∀ p ∈ part.set2, r.friend p ∉ part.set2 ∧ r.enemy p ∉ part.set2) :=
  sorry

end NUMINAMATH_CALUDE_valid_partition_exists_l3367_336759


namespace NUMINAMATH_CALUDE_evaluate_expression_l3367_336771

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  y * (y - 3 * x + 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3367_336771


namespace NUMINAMATH_CALUDE_min_m_for_perfect_fourth_power_min_m_value_exact_min_m_l3367_336751

theorem min_m_for_perfect_fourth_power (m n : ℕ+) (h : 24 * m = n ^ 4) : 
  ∀ k : ℕ+, 24 * k = (some_nat : ℕ) ^ 4 → m ≤ k := by
  sorry

theorem min_m_value (m n : ℕ+) (h : 24 * m = n ^ 4) : m ≥ 54 := by
  sorry

theorem exact_min_m (m n : ℕ+) (h : 24 * m = n ^ 4) : 
  (∃ k : ℕ+, 24 * 54 = k ^ 4) ∧ m ≥ 54 := by
  sorry

end NUMINAMATH_CALUDE_min_m_for_perfect_fourth_power_min_m_value_exact_min_m_l3367_336751


namespace NUMINAMATH_CALUDE_cone_radius_from_melted_cylinder_l3367_336723

/-- The radius of a cone formed by melting a cylinder -/
theorem cone_radius_from_melted_cylinder (r_cylinder h_cylinder h_cone : ℝ) 
  (h_r : r_cylinder = 8)
  (h_h_cylinder : h_cylinder = 2)
  (h_h_cone : h_cone = 6) : 
  ∃ (r_cone : ℝ), r_cone = 8 ∧ 
  (π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone) :=
by
  sorry

#check cone_radius_from_melted_cylinder

end NUMINAMATH_CALUDE_cone_radius_from_melted_cylinder_l3367_336723


namespace NUMINAMATH_CALUDE_roots_are_irrational_l3367_336792

theorem roots_are_irrational (k : ℝ) : 
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0) →
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 4*k*x + 3*k^2 + 1 = 0 ∧ y^2 - 4*k*y + 3*k^2 + 1 = 0 ∧ 
   ¬(∃ m n : ℤ, x = ↑m / ↑n) ∧ ¬(∃ m n : ℤ, y = ↑m / ↑n)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l3367_336792


namespace NUMINAMATH_CALUDE_harry_initial_bid_was_500_l3367_336714

/-- Represents the auction scenario with given conditions --/
structure Auction where
  startingBid : ℕ
  harryFirstBid : ℕ
  harryFinalBid : ℕ
  finalBidDifference : ℕ

/-- Calculates the second bidder's bid --/
def secondBid (a : Auction) : ℕ := a.startingBid + 2 * a.harryFirstBid

/-- Calculates the third bidder's bid --/
def thirdBid (a : Auction) : ℕ := a.startingBid + 5 * a.harryFirstBid

/-- Theorem stating that Harry's initial bid increment was $500 --/
theorem harry_initial_bid_was_500 (a : Auction) 
  (h1 : a.startingBid = 300)
  (h2 : a.harryFinalBid = 4000)
  (h3 : a.finalBidDifference = 1500)
  (h4 : a.harryFinalBid = thirdBid a + a.finalBidDifference) :
  a.harryFirstBid = 500 := by
  sorry


end NUMINAMATH_CALUDE_harry_initial_bid_was_500_l3367_336714


namespace NUMINAMATH_CALUDE_percentage_of_returned_books_l3367_336711

def initial_books : ℕ := 75
def final_books : ℕ := 57
def loaned_books : ℕ := 60

theorem percentage_of_returned_books :
  (initial_books - final_books) * 100 / loaned_books = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_returned_books_l3367_336711


namespace NUMINAMATH_CALUDE_banana_boxes_theorem_l3367_336797

/-- The number of bananas Marilyn has -/
def total_bananas : ℕ := 40

/-- The number of bananas each box must contain -/
def bananas_per_box : ℕ := 5

/-- The number of boxes needed to store all bananas -/
def num_boxes : ℕ := total_bananas / bananas_per_box

theorem banana_boxes_theorem : num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_banana_boxes_theorem_l3367_336797


namespace NUMINAMATH_CALUDE_initial_pigs_l3367_336795

theorem initial_pigs (initial : ℕ) : initial + 22 = 86 → initial = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_pigs_l3367_336795


namespace NUMINAMATH_CALUDE_vector_collinearity_problem_l3367_336781

/-- Given two 2D vectors are collinear if the cross product of their coordinates is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- The problem statement -/
theorem vector_collinearity_problem (m : ℝ) :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-1, 2)
  collinear (m * a.1 + 4 * b.1, m * a.2 + 4 * b.2) (a.1 - 2 * b.1, a.2 - 2 * b.2) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_problem_l3367_336781


namespace NUMINAMATH_CALUDE_largest_and_smallest_decimal_l3367_336740

def Digits : Set ℕ := {0, 1, 2, 3}

def IsValidDecimal (d : ℚ) : Prop :=
  ∃ (a b c : ℕ) (n : ℕ), 
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧
    d = (100 * a + 10 * b + c : ℚ) / (10^n : ℚ) ∧
    (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3)

theorem largest_and_smallest_decimal :
  (∀ d : ℚ, IsValidDecimal d → d ≤ 321) ∧
  (∀ d : ℚ, IsValidDecimal d → 0.123 ≤ d) :=
sorry

end NUMINAMATH_CALUDE_largest_and_smallest_decimal_l3367_336740
