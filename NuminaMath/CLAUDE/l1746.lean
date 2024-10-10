import Mathlib

namespace roots_of_unity_real_fifth_power_l1746_174694

theorem roots_of_unity_real_fifth_power :
  ∃ (S : Finset ℂ), 
    (S.card = 30) ∧ 
    (∀ z ∈ S, z^30 = 1) ∧
    (∃ (T : Finset ℂ), 
      (T ⊆ S) ∧ 
      (T.card = 10) ∧ 
      (∀ z ∈ T, ∃ (r : ℝ), z^5 = r) ∧
      (∀ z ∈ S \ T, ¬∃ (r : ℝ), z^5 = r)) := by
  sorry

end roots_of_unity_real_fifth_power_l1746_174694


namespace sarah_candy_duration_l1746_174693

/-- The number of days Sarah's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (friends_candy : ℕ) 
  (traded_candy : ℕ) (given_away_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  let total_received := neighbors_candy + sister_candy + friends_candy
  let total_removed := traded_candy + given_away_candy
  let remaining_candy := total_received - total_removed
  remaining_candy / daily_consumption

/-- Theorem stating that Sarah's candy will last 9 days -/
theorem sarah_candy_duration : 
  candy_duration 66 15 20 10 5 9 = 9 := by
  sorry

end sarah_candy_duration_l1746_174693


namespace common_tangent_sum_l1746_174625

/-- Parabola Q₁ -/
def Q₁ (x y : ℝ) : Prop := y = x^2 + 2

/-- Parabola Q₂ -/
def Q₂ (x y : ℝ) : Prop := x = y^2 + 8

/-- Common tangent line M -/
def M (d e f : ℤ) (x y : ℝ) : Prop := d * x + e * y = f

/-- M has nonzero integer slope -/
def nonzero_integer_slope (d e : ℤ) : Prop := d ≠ 0 ∧ e ≠ 0

/-- d, e, f are coprime -/
def coprime (d e f : ℤ) : Prop := Nat.gcd (Nat.gcd d.natAbs e.natAbs) f.natAbs = 1

/-- Main theorem -/
theorem common_tangent_sum (d e f : ℤ) :
  (∃ x y : ℝ, Q₁ x y ∧ Q₂ x y ∧ M d e f x y) →
  nonzero_integer_slope d e →
  coprime d e f →
  d + e + f = 8 := by sorry

end common_tangent_sum_l1746_174625


namespace distinct_sums_lower_bound_l1746_174692

theorem distinct_sums_lower_bound (n : ℕ) (a : ℕ → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ i, 0 < a i) :
  (Finset.powerset (Finset.range n)).card ≥ n * (n + 1) / 2 := by
  sorry

end distinct_sums_lower_bound_l1746_174692


namespace max_value_of_z_l1746_174624

theorem max_value_of_z (x y : ℝ) (h1 : y ≤ 1) (h2 : x + y ≥ 0) (h3 : x - y - 2 ≤ 0) :
  ∃ (z : ℝ), z = x - 2*y ∧ z ≤ 3 ∧ ∀ (w : ℝ), w = x - 2*y → w ≤ z :=
by sorry

end max_value_of_z_l1746_174624


namespace max_integer_solution_inequality_system_negative_six_satisfies_system_max_integer_solution_is_negative_six_l1746_174697

theorem max_integer_solution_inequality_system :
  ∀ x : ℤ, (x + 5 < 0 ∧ (3 * x - 1) / 2 ≥ 2 * x + 1) → x ≤ -6 :=
by
  sorry

theorem negative_six_satisfies_system :
  -6 + 5 < 0 ∧ (3 * (-6) - 1) / 2 ≥ 2 * (-6) + 1 :=
by
  sorry

theorem max_integer_solution_is_negative_six :
  ∃ x : ℤ, x + 5 < 0 ∧ (3 * x - 1) / 2 ≥ 2 * x + 1 ∧
  ∀ y : ℤ, (y + 5 < 0 ∧ (3 * y - 1) / 2 ≥ 2 * y + 1) → y ≤ x :=
by
  sorry

end max_integer_solution_inequality_system_negative_six_satisfies_system_max_integer_solution_is_negative_six_l1746_174697


namespace arithmetic_calculations_l1746_174651

theorem arithmetic_calculations :
  (5 + (-6) + 3 - 8 - (-4) = -2) ∧
  (-2^2 - 3 * (-1)^3 - (-1) / (-1/2)^2 = 3) := by
  sorry

end arithmetic_calculations_l1746_174651


namespace garden_area_l1746_174663

theorem garden_area (width length : ℝ) : 
  length = 3 * width + 30 →
  2 * (width + length) = 780 →
  width * length = 27000 := by
sorry

end garden_area_l1746_174663


namespace imaginary_part_of_complex_fraction_l1746_174636

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (3 - i)
  Complex.im z = -1/5 := by sorry

end imaginary_part_of_complex_fraction_l1746_174636


namespace bryans_skittles_count_l1746_174660

theorem bryans_skittles_count (ben_mm : ℕ) (bryan_extra : ℕ) 
  (h1 : ben_mm = 20) 
  (h2 : bryan_extra = 30) : 
  ben_mm + bryan_extra = 50 := by
  sorry

end bryans_skittles_count_l1746_174660


namespace arithmetic_sequence_problem_l1746_174632

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 2 = 4 →                     -- given condition
  ∃ r : ℝ,                      -- existence of common ratio for geometric sequence
    (1 + a 3) * r = a 6 ∧       -- geometric sequence conditions
    a 6 * r = 4 + a 10 →
  d = 3 := by
sorry

end arithmetic_sequence_problem_l1746_174632


namespace arithmetic_progression_prime_divisibility_l1746_174638

theorem arithmetic_progression_prime_divisibility
  (p : ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (h_prime : Prime p)
  (h_seq : ∀ i ∈ Finset.range p, Prime (a i))
  (h_arith : ∀ i ∈ Finset.range (p - 1), a (i + 1) = a i + d)
  (h_incr : ∀ i ∈ Finset.range (p - 1), a i < a (i + 1))
  (h_greater : p < a 0) :
  p ∣ d := by
sorry

end arithmetic_progression_prime_divisibility_l1746_174638


namespace normal_distribution_two_std_dev_below_mean_l1746_174648

theorem normal_distribution_two_std_dev_below_mean 
  (μ : ℝ) (σ : ℝ) (h_μ : μ = 17.5) (h_σ : σ = 2.5) :
  μ - 2 * σ = 12.5 :=
by sorry

end normal_distribution_two_std_dev_below_mean_l1746_174648


namespace triangle_area_l1746_174657

theorem triangle_area (a b c A B C : Real) (h1 : A = π/4) (h2 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) : 
  (1/2) * b * c * Real.sin A = 2 := by
  sorry

end triangle_area_l1746_174657


namespace max_area_parabola_triangle_l1746_174626

/-- Given two points on a parabola, prove the maximum area of a triangle formed with a specific third point -/
theorem max_area_parabola_triangle (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ ≠ x₂ →
  x₁ + x₂ = 4 →
  y₁^2 = 6*x₁ →
  y₂^2 = 6*x₂ →
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let M := ((x₁ + x₂)/2, (y₁ + y₂)/2)
  let k_AB := (y₂ - y₁)/(x₂ - x₁)
  let C := (5, 0)
  let triangle_area := abs ((x₁ - 5)*(y₂ - 0) + (x₂ - x₁)*(0 - y₁) + (5 - x₂)*(y₁ - y₂)) / 2
  ∃ (max_area : ℝ), max_area = 14 * Real.sqrt 7 / 3 ∧ 
    ∀ (x₁' x₂' y₁' y₂' : ℝ), 
      x₁' ≠ x₂' → 
      x₁' + x₂' = 4 → 
      y₁'^2 = 6*x₁' → 
      y₂'^2 = 6*x₂' → 
      let A' := (x₁', y₁')
      let B' := (x₂', y₂')
      let triangle_area' := abs ((x₁' - 5)*(y₂' - 0) + (x₂' - x₁')*(0 - y₁') + (5 - x₂')*(y₁' - y₂')) / 2
      triangle_area' ≤ max_area := by
  sorry

end max_area_parabola_triangle_l1746_174626


namespace average_difference_l1746_174675

theorem average_difference (x y z w : ℝ) : 
  (x + y + z) / 3 = (y + z + w) / 3 + 10 → w = x - 30 := by
sorry

end average_difference_l1746_174675


namespace corn_acreage_l1746_174690

theorem corn_acreage (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ) 
  (h1 : total_land = 1034)
  (h2 : ratio_beans = 5)
  (h3 : ratio_wheat = 2)
  (h4 : ratio_corn = 4) : 
  (total_land * ratio_corn) / (ratio_beans + ratio_wheat + ratio_corn) = 376 := by
  sorry

#eval (1034 * 4) / (5 + 2 + 4)

end corn_acreage_l1746_174690


namespace louis_ate_nine_boxes_l1746_174699

/-- The number of Lemon Heads in each package -/
def lemon_heads_per_package : ℕ := 6

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := total_lemon_heads / lemon_heads_per_package

theorem louis_ate_nine_boxes : whole_boxes = 9 := by
  sorry

end louis_ate_nine_boxes_l1746_174699


namespace complex_modulus_l1746_174662

theorem complex_modulus (z : ℂ) (h : z * Complex.I = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end complex_modulus_l1746_174662


namespace quadratic_roots_ratio_l1746_174633

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end quadratic_roots_ratio_l1746_174633


namespace percent_problem_l1746_174681

theorem percent_problem (x : ℝ) : (0.25 * x = 0.12 * 1500 - 15) → x = 660 := by
  sorry

end percent_problem_l1746_174681


namespace inverse_proportion_l1746_174680

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 54) (h3 : x = 3 * y) :
  ∃ (y' : ℝ), 5 * y' = k ∧ y' = 109.35 := by
  sorry

end inverse_proportion_l1746_174680


namespace initial_files_correct_l1746_174621

/-- The number of files Megan initially had on her computer -/
def initial_files : ℕ := 93

/-- The number of files Megan deleted -/
def deleted_files : ℕ := 21

/-- The number of files in each folder -/
def files_per_folder : ℕ := 8

/-- The number of folders Megan ended up with -/
def num_folders : ℕ := 9

/-- Theorem stating that the initial number of files is correct -/
theorem initial_files_correct : 
  initial_files = deleted_files + num_folders * files_per_folder :=
by sorry

end initial_files_correct_l1746_174621


namespace sum_of_last_two_digits_of_7_25_plus_13_25_l1746_174628

theorem sum_of_last_two_digits_of_7_25_plus_13_25 : 
  (7^25 + 13^25) % 100 = 50 := by
sorry

end sum_of_last_two_digits_of_7_25_plus_13_25_l1746_174628


namespace train_speed_l1746_174641

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 170)
  (h3 : time_to_cross = 13.998880089592832) :
  (train_length + bridge_length) / time_to_cross = 20.0014286607 := by
  sorry

end train_speed_l1746_174641


namespace x_intercept_of_specific_line_l1746_174674

/-- A line passing through three points in a rectangular coordinate system -/
structure Line where
  p1 : Prod ℝ ℝ
  p2 : Prod ℝ ℝ
  p3 : Prod ℝ ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- Theorem: The x-intercept of the line passing through (10, 3), (-10, -7), and (5, 1) is 4 -/
theorem x_intercept_of_specific_line :
  let l : Line := { p1 := (10, 3), p2 := (-10, -7), p3 := (5, 1) }
  x_intercept l = 4 := by
  sorry

end x_intercept_of_specific_line_l1746_174674


namespace geometric_sequence_sixth_term_l1746_174671

/-- Given a geometric sequence {a_n} where a₂ = 2 and a₁₀ = 8, prove that a₆ = 4 -/
theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  (h_2 : a 2 = 2)  -- Second term is 2
  (h_10 : a 10 = 8)  -- Tenth term is 8
  : a 6 = 4 :=
by sorry

end geometric_sequence_sixth_term_l1746_174671


namespace student_arrangement_count_l1746_174629

theorem student_arrangement_count : ℕ := by
  -- Define the total number of students
  let total_students : ℕ := 7
  
  -- Define the condition that A and B are adjacent
  let adjacent_pair : ℕ := 1
  
  -- Define the condition that C and D are not adjacent
  let non_adjacent_pair : ℕ := 2
  
  -- Define the number of entities to arrange after bundling A and B
  let entities : ℕ := total_students - adjacent_pair
  
  -- Define the number of gaps after arranging the entities
  let gaps : ℕ := entities + 1
  
  -- Calculate the total number of arrangements
  let arrangements : ℕ := 
    (Nat.factorial entities) *    -- Arrange entities
    (gaps * (gaps - 1)) *         -- Place C and D in gaps
    2                             -- Arrange A and B within their bundle
  
  -- Prove that the number of arrangements is 960
  sorry

end student_arrangement_count_l1746_174629


namespace jakes_total_earnings_l1746_174696

/-- Calculates Jake's total earnings from selling baby snakes --/
def jakes_earnings (viper_count cobra_count python_count anaconda_count : ℕ)
  (viper_eggs cobra_eggs python_eggs anaconda_eggs : ℕ)
  (viper_price cobra_price python_price anaconda_price : ℚ)
  (viper_discount cobra_discount python_discount anaconda_discount : ℚ) : ℚ :=
  let viper_total := viper_count * viper_eggs * (viper_price * (1 - viper_discount))
  let cobra_total := cobra_count * cobra_eggs * (cobra_price * (1 - cobra_discount))
  let python_total := python_count * python_eggs * (python_price * (1 - python_discount))
  let anaconda_total := anaconda_count * anaconda_eggs * (anaconda_price * (1 - anaconda_discount))
  viper_total + cobra_total + python_total + anaconda_total

/-- Theorem stating Jake's total earnings --/
theorem jakes_total_earnings :
  jakes_earnings 3 2 1 1 3 2 4 5 300 250 450 500 (10/100) (5/100) (75/1000) (12/100) = 7245 := by
  sorry

end jakes_total_earnings_l1746_174696


namespace river_length_problem_l1746_174695

theorem river_length_problem (straight_length crooked_length total_length : ℝ) :
  straight_length * 3 = crooked_length →
  straight_length + crooked_length = total_length →
  total_length = 80 →
  straight_length = 20 := by
  sorry

end river_length_problem_l1746_174695


namespace trajectory_and_line_equation_l1746_174614

/-- The trajectory of point P and the equation of line l -/
theorem trajectory_and_line_equation :
  ∀ (P : ℝ × ℝ) (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (M : ℝ × ℝ),
  F = (3 * Real.sqrt 3, 0) →
  l = {(x, y) | x = 4 * Real.sqrt 3} →
  M = (4, 2) →
  (∀ (x y : ℝ), P = (x, y) →
    Real.sqrt ((x - 3 * Real.sqrt 3)^2 + y^2) / |x - 4 * Real.sqrt 3| = Real.sqrt 3 / 2) →
  (∃ (B C : ℝ × ℝ), B ∈ l ∧ C ∈ l ∧ B ≠ C ∧ M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) →
  (∀ (x y : ℝ), P = (x, y) → x^2 / 36 + y^2 / 9 = 1) ∧
  (∃ (k : ℝ), k = -1/2 ∧ ∀ (x y : ℝ), y - 2 = k * (x - 4) ↔ x + 2*y - 8 = 0) :=
by sorry

end trajectory_and_line_equation_l1746_174614


namespace sum_in_base5_l1746_174673

/-- Converts a number from base 10 to base 5 -/
def toBase5 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 5 to base 10 -/
def fromBase5 (n : ℕ) : ℕ := sorry

theorem sum_in_base5 : toBase5 (45 + 27) = 242 := by sorry

end sum_in_base5_l1746_174673


namespace intersection_parallel_line_equation_specific_line_equation_l1746_174685

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line_equation (a b c d e f g h i : ℝ) :
  (∃ x y, a * x + b * y = c ∧ d * x + e * y = f) →  -- Intersection point exists
  (∀ x y, (a * x + b * y = c ∧ d * x + e * y = f) → g * x + h * y + i = 0) →  -- Line passes through intersection
  (∃ k, ∀ x y, g * x + h * y + i = k * (g * x + h * y + 0)) →  -- Parallel to g * x + h * y + 0 = 0
  ∃ k, ∀ x y, g * x + h * y + i = k * (g * x + h * y - 27) :=
by sorry

/-- The specific case for the given problem -/
theorem specific_line_equation :
  (∃ x y, x + y = 9 ∧ 2 * x - y = 18) →
  (∀ x y, (x + y = 9 ∧ 2 * x - y = 18) → 3 * x - 2 * y + i = 0) →
  (∃ k, ∀ x y, 3 * x - 2 * y + i = k * (3 * x - 2 * y + 8)) →
  i = -27 :=
by sorry

end intersection_parallel_line_equation_specific_line_equation_l1746_174685


namespace no_real_roots_l1746_174646

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 7) - Real.sqrt (x - 5) + 2 = 0 := by
  sorry

end no_real_roots_l1746_174646


namespace expression_evaluation_l1746_174611

theorem expression_evaluation : 
  let x : ℝ := 3
  let expr := (2 * x^2 + 2*x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2*x + 1)
  expr / (x / (x + 1)) = 2 := by sorry

end expression_evaluation_l1746_174611


namespace system_consistent_iff_k_equals_four_l1746_174688

theorem system_consistent_iff_k_equals_four 
  (x y u : ℝ) (k : ℝ) : 
  (x + y = 1 ∧ k * x + y = 2 ∧ x + k * u = 3) ↔ k = 4 := by
  sorry

end system_consistent_iff_k_equals_four_l1746_174688


namespace probability_females_right_of_males_l1746_174635

theorem probability_females_right_of_males :
  let total_people : ℕ := 3 + 2
  let male_count : ℕ := 3
  let female_count : ℕ := 2
  let total_arrangements : ℕ := Nat.factorial total_people
  let favorable_arrangements : ℕ := Nat.factorial male_count * Nat.factorial female_count
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 10 := by
  sorry

end probability_females_right_of_males_l1746_174635


namespace M_properties_M_remainder_l1746_174617

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + 
        d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    1 ≤ d1 ∧ d1 ≤ 9 ∧
    1 ≤ d2 ∧ d2 ≤ 9 ∧
    1 ≤ d3 ∧ d3 ≤ 9 ∧
    1 ≤ d4 ∧ d4 ≤ 9 ∧
    1 ≤ d5 ∧ d5 ≤ 9 ∧
    1 ≤ d6 ∧ d6 ≤ 9 ∧
    1 ≤ d7 ∧ d7 ≤ 9 ∧
    1 ≤ d8 ∧ d8 ≤ 9 ∧
    1 ≤ d9 ∧ d9 ≤ 9

def M : ℕ := sorry

theorem M_properties :
  is_valid_number M ∧ 
  M % 12 = 0 ∧
  ∀ n, is_valid_number n ∧ n % 12 = 0 → n ≤ M :=
by sorry

theorem M_remainder : M % 100 = 12 :=
by sorry

end M_properties_M_remainder_l1746_174617


namespace trigonometric_identity_l1746_174640

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (α + Real.pi / 6) ^ 2 + Real.sin α * Real.cos (α + Real.pi / 6) = 3/4 := by
  sorry

end trigonometric_identity_l1746_174640


namespace negative_one_powers_equality_l1746_174687

theorem negative_one_powers_equality : -1^2022 - (-1)^2023 - (-1)^0 = -1 := by
  sorry

end negative_one_powers_equality_l1746_174687


namespace matthews_crackers_l1746_174647

theorem matthews_crackers (num_friends : ℕ) (crackers_eaten_per_friend : ℕ) 
  (h1 : num_friends = 18)
  (h2 : crackers_eaten_per_friend = 2) :
  num_friends * crackers_eaten_per_friend = 36 := by
  sorry

end matthews_crackers_l1746_174647


namespace product_mod_500_l1746_174650

theorem product_mod_500 : (1493 * 1998) % 500 = 14 := by
  sorry

end product_mod_500_l1746_174650


namespace mary_flour_calculation_l1746_174618

/-- The amount of flour needed for the recipe -/
def total_flour : ℕ := 9

/-- The amount of flour Mary has already added -/
def added_flour : ℕ := 3

/-- The remaining amount of flour Mary needs to add -/
def remaining_flour : ℕ := total_flour - added_flour

theorem mary_flour_calculation :
  remaining_flour = 6 := by
  sorry

end mary_flour_calculation_l1746_174618


namespace onion_shelf_problem_l1746_174676

/-- Given the initial conditions of onions on a shelf, prove the final number of onions. -/
theorem onion_shelf_problem (initial : ℕ) (sold : ℕ) (added : ℕ) (given_away : ℕ) : 
  initial = 98 → sold = 65 → added = 20 → given_away = 10 → 
  initial - sold + added - given_away = 43 := by
sorry

end onion_shelf_problem_l1746_174676


namespace initial_speed_is_40_l1746_174652

/-- A person's journey with varying speeds -/
def Journey (D T : ℝ) (initial_speed final_speed : ℝ) : Prop :=
  initial_speed > 0 ∧ final_speed > 0 ∧ D > 0 ∧ T > 0 ∧
  (2/3 * D) / (1/3 * T) = initial_speed ∧
  (1/3 * D) / (1/3 * T) = final_speed

/-- Theorem: Given the conditions, the initial speed is 40 kmph -/
theorem initial_speed_is_40 (D T : ℝ) :
  Journey D T initial_speed 20 → initial_speed = 40 := by
  sorry

#check initial_speed_is_40

end initial_speed_is_40_l1746_174652


namespace jackson_missed_wednesdays_l1746_174630

/-- The number of missed Wednesdays in Jackson's school year --/
def missed_wednesdays (weeks : ℕ) (total_sandwiches : ℕ) (missed_fridays : ℕ) : ℕ :=
  weeks * 2 - total_sandwiches - missed_fridays

theorem jackson_missed_wednesdays :
  missed_wednesdays 36 69 2 = 1 := by
  sorry

end jackson_missed_wednesdays_l1746_174630


namespace smallest_number_divisible_l1746_174615

theorem smallest_number_divisible (n : ℕ) : n = 34 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 10 = 2 * k ∧ m - 10 = 6 * k ∧ m - 10 = 12 * k ∧ m - 10 = 24 * k)) ∧
  (∃ k : ℕ, n - 10 = 2 * k ∧ n - 10 = 6 * k ∧ n - 10 = 12 * k ∧ n - 10 = 24 * k) ∧
  n > 10 :=
by sorry

#check smallest_number_divisible

end smallest_number_divisible_l1746_174615


namespace community_service_arrangements_l1746_174665

def arrange_people (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem community_service_arrangements : 
  arrange_people 6 4 + arrange_people 6 3 + arrange_people 6 2 = 50 := by
  sorry

end community_service_arrangements_l1746_174665


namespace probability_white_ball_l1746_174672

/-- The probability of drawing a white ball from a bag with specified numbers of colored balls. -/
theorem probability_white_ball (white red black : ℕ) : 
  white = 3 → red = 4 → black = 5 → (white : ℚ) / (white + red + black) = 1/4 := by
  sorry

end probability_white_ball_l1746_174672


namespace dice_probability_l1746_174658

/-- The number of dice --/
def n : ℕ := 7

/-- The number of sides on each die --/
def sides : ℕ := 12

/-- The number of favorable outcomes on each die (numbers less than 6) --/
def favorable : ℕ := 5

/-- The number of dice we want to show a favorable outcome --/
def k : ℕ := 3

/-- The probability of exactly k out of n dice showing a number less than 6 --/
def probability : ℚ := (n.choose k) * (favorable / sides) ^ k * ((sides - favorable) / sides) ^ (n - k)

theorem dice_probability : probability = 10504375 / 373248 := by
  sorry

end dice_probability_l1746_174658


namespace find_x_l1746_174607

theorem find_x (x y z : ℝ) 
  (hxy : x * y / (x + y) = 4)
  (hxz : x * z / (x + z) = 9)
  (hyz : y * z / (y + z) = 16)
  (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hdist : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  x = 384 / 21 := by
sorry

end find_x_l1746_174607


namespace wall_decoration_thumbtack_fraction_l1746_174645

theorem wall_decoration_thumbtack_fraction :
  let total_decorations : ℕ := 50 * 3 / 2
  let nailed_decorations : ℕ := 50
  let remaining_decorations : ℕ := total_decorations - nailed_decorations
  let sticky_strip_decorations : ℕ := 15
  let thumbtack_decorations : ℕ := remaining_decorations - sticky_strip_decorations
  (thumbtack_decorations : ℚ) / remaining_decorations = 2 / 5 :=
by sorry

end wall_decoration_thumbtack_fraction_l1746_174645


namespace phil_remaining_books_pages_l1746_174605

def book_pages : List Nat := [120, 150, 80, 200, 90, 180, 75, 190, 110, 160, 130, 170, 100, 140, 210]

def misplaced_indices : List Nat := [1, 5, 9, 14]  -- 0-based indices

def remaining_pages : Nat := book_pages.sum - (misplaced_indices.map (λ i => book_pages.get! i)).sum

theorem phil_remaining_books_pages :
  remaining_pages = 1305 := by sorry

end phil_remaining_books_pages_l1746_174605


namespace rectangle_side_length_l1746_174664

/-- If a rectangle has area 4a²b³ and one side 2ab³, then the other side is 2a -/
theorem rectangle_side_length (a b : ℝ) (area : ℝ) (side1 : ℝ) :
  area = 4 * a^2 * b^3 → side1 = 2 * a * b^3 → area / side1 = 2 * a :=
by sorry

end rectangle_side_length_l1746_174664


namespace black_squares_10th_row_l1746_174639

def stair_step_squares (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else stair_step_squares (n - 1) + 2^(n - 1)

def black_squares (n : ℕ) : ℕ :=
  (stair_step_squares n - 1) / 2

theorem black_squares_10th_row :
  black_squares 10 = 511 := by
  sorry

end black_squares_10th_row_l1746_174639


namespace binomial_recursion_l1746_174610

theorem binomial_recursion (n k : ℕ) (h1 : k ≤ n) (h2 : ¬(n = 0 ∧ k = 0)) :
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) := by
  sorry

end binomial_recursion_l1746_174610


namespace min_value_a_solution_set_l1746_174643

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- Theorem for the minimum value of a
theorem min_value_a :
  ∃ (a : ℝ), ∀ (x : ℝ), f x a ≥ a ∧ (∃ (x₀ : ℝ), f x₀ a = a) ∧ a = 2 :=
sorry

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set :
  ∃ (a : ℝ), a = 2 ∧ {x : ℝ | f x a ≤ 5} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 11/2} :=
sorry

end min_value_a_solution_set_l1746_174643


namespace inequality_proof_l1746_174654

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_sum : 1/x + 1/y + 1/z = 2) : 8 * (x - 1) * (y - 1) * (z - 1) ≤ 1 := by
  sorry

end inequality_proof_l1746_174654


namespace complex_multiplication_l1746_174668

theorem complex_multiplication :
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := 2 - 3 * Complex.I
  z₁ * z₂ = 7 - 4 * Complex.I := by
sorry

end complex_multiplication_l1746_174668


namespace ambiguous_dates_count_l1746_174669

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The minimum number of days in each month -/
def min_days_per_month : ℕ := 12

/-- The number of days with ambiguous date interpretation -/
def ambiguous_days : ℕ := months_in_year * min_days_per_month - months_in_year

theorem ambiguous_dates_count :
  ambiguous_days = 132 :=
sorry

end ambiguous_dates_count_l1746_174669


namespace percentage_division_equality_l1746_174677

theorem percentage_division_equality : 
  (208 / 100 * 1265) / 6 = 438.53333333333336 := by sorry

end percentage_division_equality_l1746_174677


namespace divisible_by_seven_l1746_174678

/-- The number of repeated digits -/
def n : ℕ := 50

/-- The number formed by n eights followed by x followed by n nines -/
def f (x : ℕ) : ℕ :=
  8 * (10^(2*n + 1) - 1) / 9 + x * 10^n + 9 * (10^n - 1) / 9

/-- The main theorem -/
theorem divisible_by_seven (x : ℕ) : 7 ∣ f x ↔ x = 0 := by sorry

end divisible_by_seven_l1746_174678


namespace distribute_five_into_three_l1746_174655

/-- The number of ways to distribute n distinct objects into k indistinguishable containers --/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 5 distinct objects into 3 indistinguishable containers
    results in 51 different arrangements --/
theorem distribute_five_into_three :
  distribute 5 3 = 51 := by
  sorry

end distribute_five_into_three_l1746_174655


namespace partnership_profit_l1746_174609

/-- The total profit of a partnership business given C's share and percentage -/
theorem partnership_profit (c_share : ℕ) (c_percentage : ℕ) (total_profit : ℕ) : 
  c_share = 60000 → c_percentage = 25 → total_profit = 240000 := by
  sorry

end partnership_profit_l1746_174609


namespace holdens_class_results_l1746_174642

/-- Proves the number of students who received an A in Mr. Holden's class exam
    and the number of students who did not receive an A in Mr. Holden's class quiz -/
theorem holdens_class_results (kennedy_total : ℕ) (kennedy_a : ℕ) (holden_total : ℕ)
    (h1 : kennedy_total = 20)
    (h2 : kennedy_a = 8)
    (h3 : holden_total = 30)
    (h4 : (kennedy_a : ℚ) / kennedy_total = (holden_a : ℚ) / holden_total)
    (h5 : (holden_total - holden_a : ℚ) / holden_total = 2 * (holden_not_a_quiz : ℚ) / holden_total) :
    holden_a = 12 ∧ holden_not_a_quiz = 9 := by
  sorry

#check holdens_class_results

end holdens_class_results_l1746_174642


namespace coral_reef_number_conversion_l1746_174601

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to hexadecimal --/
def decimal_to_hex (n : ℕ) : String := sorry

theorem coral_reef_number_conversion :
  let octal_num := 732
  let decimal_num := octal_to_decimal octal_num
  decimal_num = 474 ∧ decimal_to_hex decimal_num = "1DA" := by sorry

end coral_reef_number_conversion_l1746_174601


namespace restaurant_gratuity_l1746_174686

/-- Calculate the gratuity for a restaurant bill -/
theorem restaurant_gratuity (price1 price2 price3 : ℕ) (tip_percentage : ℚ) : 
  price1 = 10 → price2 = 13 → price3 = 17 → tip_percentage = 1/10 →
  (price1 + price2 + price3 : ℚ) * tip_percentage = 4 := by
  sorry

end restaurant_gratuity_l1746_174686


namespace calculate_expression_l1746_174679

theorem calculate_expression : 15 * 28 + 42 * 15 + 15^2 = 1275 := by
  sorry

end calculate_expression_l1746_174679


namespace gross_profit_calculation_l1746_174670

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 91 →
  gross_profit_percentage = 1.6 →
  ∃ (cost : ℝ), sales_price = cost + gross_profit_percentage * cost ∧
                 gross_profit_percentage * cost = 56 := by
  sorry

end gross_profit_calculation_l1746_174670


namespace equation_equivalence_l1746_174603

theorem equation_equivalence (a b : ℝ) (h : a + 2 * b + 2 = Real.sqrt 2) : 
  4 * a + 8 * b + 5 = 4 * Real.sqrt 2 - 3 := by
  sorry

end equation_equivalence_l1746_174603


namespace target_line_properties_l1746_174637

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 6 * y + 4 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := 2 * x + y = 5

/-- The intersection point of l₁ and l₂ -/
def intersection_point : ℝ × ℝ := (2, 1)

/-- The equation of the line we're proving -/
def target_line (x y : ℝ) : Prop := x - 2 * y = 0

/-- Theorem stating that the target_line passes through the intersection point and is perpendicular to l₂ -/
theorem target_line_properties :
  (target_line (intersection_point.1) (intersection_point.2)) ∧
  (∀ x y : ℝ, l₂ x y → ∀ x' y' : ℝ, target_line x' y' →
    (y' - intersection_point.2) * (x - intersection_point.1) = 
    -(x' - intersection_point.1) * (y - intersection_point.2)) :=
sorry

end target_line_properties_l1746_174637


namespace committee_selection_with_president_l1746_174631

/-- The number of ways to choose a committee with a required member -/
def choose_committee_with_required (n m k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: Choosing a 5-person committee from 12 people with at least one being the president -/
theorem committee_selection_with_president :
  choose_committee_with_required 12 1 5 = 330 := by
  sorry

end committee_selection_with_president_l1746_174631


namespace last_digit_product_l1746_174620

/-- The last digit of (3^65 * 6^n * 7^71) is 4 for any non-negative integer n. -/
theorem last_digit_product (n : ℕ) : (3^65 * 6^n * 7^71) % 10 = 4 := by
  sorry

end last_digit_product_l1746_174620


namespace odd_function_half_period_zero_l1746_174649

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the smallest positive period T
variable (T : ℝ)

-- Define the oddness property of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the periodicity property of f with period T
def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Define that T is the smallest positive period
def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ has_period f T ∧ ∀ S, 0 < S ∧ S < T → ¬(has_period f S)

-- State the theorem
theorem odd_function_half_period_zero
  (h_odd : is_odd f)
  (h_period : is_smallest_positive_period f T) :
  f (-T/2) = 0 :=
sorry

end odd_function_half_period_zero_l1746_174649


namespace shaded_half_l1746_174634

/-- Represents a square divided into smaller squares with specific shading -/
structure DividedSquare where
  /-- The number of smaller squares the large square is divided into -/
  num_divisions : Nat
  /-- Whether a diagonal is drawn in one of the smaller squares -/
  has_diagonal : Bool
  /-- The number of quarters of a smaller square that are additionally shaded -/
  additional_shaded_quarters : Nat

/-- Calculates the fraction of the large square that is shaded -/
def shaded_fraction (s : DividedSquare) : Rat :=
  sorry

/-- Theorem stating that for a specific configuration, the shaded fraction is 1/2 -/
theorem shaded_half (s : DividedSquare) 
  (h1 : s.num_divisions = 4) 
  (h2 : s.has_diagonal = true)
  (h3 : s.additional_shaded_quarters = 2) : 
  shaded_fraction s = 1/2 :=
sorry

end shaded_half_l1746_174634


namespace locus_of_point_P_l1746_174612

/-- The locus of points P(x, y) such that the product of slopes of AP and BP is -1/4,
    where A(-2, 0) and B(2, 0) are fixed points. -/
theorem locus_of_point_P (x y : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  (y / (x + 2)) * (y / (x - 2)) = -1/4 ↔ x^2 / 4 + y^2 = 1 :=
sorry

end locus_of_point_P_l1746_174612


namespace unique_f_two_l1746_174604

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x + y) + x * f y - 2 * x * y - x + 2

theorem unique_f_two (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∃! z : ℝ, f 2 = z ∧ z = 4 := by sorry

end unique_f_two_l1746_174604


namespace game_result_l1746_174659

def score_function (n : ℕ) : ℕ :=
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else if n % 2 = 1 ∧ n % 3 ≠ 0 then 1
  else 0

def allie_rolls : List ℕ := [5, 2, 6, 1, 3]
def betty_rolls : List ℕ := [6, 4, 1, 2, 5]

theorem game_result :
  (List.sum (List.map score_function allie_rolls)) *
  (List.sum (List.map score_function betty_rolls)) = 391 := by
  sorry

end game_result_l1746_174659


namespace distance_to_origin_problem_l1746_174666

theorem distance_to_origin_problem (a : ℝ) : 
  (|a| = 2) → (a - 2 = 0 ∨ a - 2 = -4) := by
  sorry

end distance_to_origin_problem_l1746_174666


namespace quadruplet_solution_l1746_174600

theorem quadruplet_solution (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ + x₂ = x₃^2 + x₄^2 + 6*x₃*x₄) ∧
  (x₁ + x₃ = x₂^2 + x₄^2 + 6*x₂*x₄) ∧
  (x₁ + x₄ = x₂^2 + x₃^2 + 6*x₂*x₃) ∧
  (x₂ + x₃ = x₁^2 + x₄^2 + 6*x₁*x₄) ∧
  (x₂ + x₄ = x₁^2 + x₃^2 + 6*x₁*x₃) ∧
  (x₃ + x₄ = x₁^2 + x₂^2 + 6*x₁*x₂) →
  (∃ c : ℝ, (x₁ = c ∧ x₂ = c ∧ x₃ = c ∧ x₄ = -3*c) ∨
            (x₁ = c ∧ x₂ = c ∧ x₃ = c ∧ x₄ = 1 - 3*c)) :=
by sorry


end quadruplet_solution_l1746_174600


namespace fraction_problem_l1746_174656

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 6) = 3 / 4 → x = 9 / 4 := by
  sorry

end fraction_problem_l1746_174656


namespace garden_perimeter_l1746_174689

-- Define the garden shape
structure Garden where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ

-- Define the conditions
def is_valid_garden (g : Garden) : Prop :=
  g.a + g.b + g.c = 3 ∧
  g.a ≥ 0 ∧ g.b ≥ 0 ∧ g.c ≥ 0 ∧ g.x ≥ 0

-- Calculate the perimeter
def perimeter (g : Garden) : ℝ :=
  3 + 5 + g.a + g.x + g.b + 4 + g.c + (4 + (5 - g.x))

-- Theorem statement
theorem garden_perimeter (g : Garden) (h : is_valid_garden g) : perimeter g = 24 := by
  sorry

end garden_perimeter_l1746_174689


namespace binomial_square_condition_l1746_174616

theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (3*x + b)^2) → a = 25 := by
  sorry

end binomial_square_condition_l1746_174616


namespace fold_three_to_negative_three_fold_seven_to_negative_five_fold_points_with_distance_l1746_174691

-- Define a folding operation
def fold (m : ℝ) (x : ℝ) : ℝ := 2 * m - x

-- Theorem 1
theorem fold_three_to_negative_three :
  fold 0 3 = -3 :=
sorry

-- Theorem 2
theorem fold_seven_to_negative_five :
  fold 1 7 = -5 :=
sorry

-- Theorem 3
theorem fold_points_with_distance (m : ℝ) (h : m > 0) :
  ∃ (a b : ℝ), a < b ∧ b - a = m ∧ fold ((a + b) / 2) a = b ∧ a = -(1/2) * m + 1 ∧ b = (1/2) * m + 1 :=
sorry

end fold_three_to_negative_three_fold_seven_to_negative_five_fold_points_with_distance_l1746_174691


namespace final_time_sum_l1746_174627

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define a function to add time
def addTime (start : Time) (elapsedHours elapsedMinutes elapsedSeconds : Nat) : Time :=
  sorry

-- Define a function to calculate the sum of time components
def sumTimeComponents (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

-- Theorem statement
theorem final_time_sum (startTime : Time) 
  (h1 : startTime.hours = 3) 
  (h2 : startTime.minutes = 0) 
  (h3 : startTime.seconds = 0) : 
  let finalTime := addTime startTime 240 58 30
  sumTimeComponents finalTime = 91 :=
sorry

end final_time_sum_l1746_174627


namespace function_nature_l1746_174644

theorem function_nature (n : ℕ) (h : 30 * n = 30 * n) :
  let f : ℝ → ℝ := fun x ↦ x ^ n
  (f 1)^2 + (f (-1))^2 = 2 * ((f 1) + (f (-1)) - 1) →
  ∀ x : ℝ, f (-x) = f x :=
by sorry

end function_nature_l1746_174644


namespace f_composition_one_third_l1746_174682

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 8^x

-- State the theorem
theorem f_composition_one_third : f (f (1/3)) = 1/8 := by
  sorry

end f_composition_one_third_l1746_174682


namespace max_sum_arithmetic_sequence_max_sum_value_max_sum_at_12_max_sum_is_144_l1746_174623

/-- The maximum sum of the first n terms of an arithmetic sequence with a_1 = 23 and d = -2 -/
theorem max_sum_arithmetic_sequence : ℕ → ℝ :=
  fun n => -n^2 + 24*n

/-- The maximum value of the sum of the first n terms is 144 -/
theorem max_sum_value : ∃ (n : ℕ), ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

/-- The value of n that maximizes the sum is 12 -/
theorem max_sum_at_12 : ∃ (n : ℕ), n = 12 ∧ ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

/-- The maximum sum value is 144 -/
theorem max_sum_is_144 : ∃ (n : ℕ), max_sum_arithmetic_sequence n = 144 ∧ ∀ (m : ℕ), max_sum_arithmetic_sequence n ≥ max_sum_arithmetic_sequence m :=
by
  sorry

end max_sum_arithmetic_sequence_max_sum_value_max_sum_at_12_max_sum_is_144_l1746_174623


namespace shaded_area_calculation_l1746_174622

/-- Given a square carpet with shaded squares, calculate the total shaded area -/
theorem shaded_area_calculation (carpet_side : ℝ) (S T : ℝ) 
  (h1 : carpet_side = 12)
  (h2 : carpet_side / S = 4)
  (h3 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end shaded_area_calculation_l1746_174622


namespace time_with_family_l1746_174683

/-- Given a 24-hour day, if a person spends 1/3 of the day sleeping, 
    1/6 of the day in school, 1/12 of the day making assignments, 
    then the remaining time spent with family is 10 hours. -/
theorem time_with_family (total_hours : ℝ) 
  (sleep_fraction : ℝ) (school_fraction : ℝ) (assignment_fraction : ℝ) :
  total_hours = 24 →
  sleep_fraction = 1/3 →
  school_fraction = 1/6 →
  assignment_fraction = 1/12 →
  total_hours - (sleep_fraction + school_fraction + assignment_fraction) * total_hours = 10 := by
  sorry

end time_with_family_l1746_174683


namespace triangle_side_sum_l1746_174661

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) (h5 : 9 = a.sin * 18) :
  18 + 9 * Real.sqrt 3 = 18 + b.sin * 18 := by
  sorry

end triangle_side_sum_l1746_174661


namespace quadratic_roots_problem_l1746_174667

theorem quadratic_roots_problem (α β : ℝ) (h1 : α^2 - α - 2021 = 0)
                                         (h2 : β^2 - β - 2021 = 0)
                                         (h3 : α > β) : 
  let A := α^2 - 2*β^2 + 2*α*β + 3*β + 7
  ⌊A⌋ = -5893 := by sorry

end quadratic_roots_problem_l1746_174667


namespace simplify_expression_l1746_174653

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) - (1 / 4) = 11 / 20 := by
  sorry

end simplify_expression_l1746_174653


namespace red_ball_probability_l1746_174613

theorem red_ball_probability (w r : ℕ+) 
  (h1 : r > w)
  (h2 : r < 2 * w)
  (h3 : 2 * w + 3 * r = 60) :
  (r : ℚ) / (w + r) = 4 / 7 := by
  sorry

end red_ball_probability_l1746_174613


namespace vector_c_value_l1746_174608

/-- Given two planar vectors a and b, returns true if they are parallel -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_c_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, m)
  let c : ℝ × ℝ := (-7, 14)
  are_parallel a b →
  (3 • a.1 + 2 • b.1 + c.1 = 0 ∧ 3 • a.2 + 2 • b.2 + c.2 = 0) →
  c = (-7, 14) := by
  sorry


end vector_c_value_l1746_174608


namespace smallest_valid_number_l1746_174619

def is_valid (N : ℕ) : Prop :=
  ∀ k ∈ Finset.range 9, (N + k + 2) % (k + 2) = 0

theorem smallest_valid_number :
  ∃ N : ℕ, is_valid N ∧ ∀ M : ℕ, M < N → ¬ is_valid M :=
by sorry

end smallest_valid_number_l1746_174619


namespace difference_of_squares_l1746_174602

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end difference_of_squares_l1746_174602


namespace box_dimensions_l1746_174606

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17)
  (h2 : a + b = 13)
  (h3 : 2 * (b + c) = 40)
  (h4 : a < b)
  (h5 : b < c) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
  sorry

end box_dimensions_l1746_174606


namespace max_a_min_b_for_sin_inequality_l1746_174698

theorem max_a_min_b_for_sin_inequality 
  (h : ∀ x ∈ Set.Ioo 0 (π/2), a * x < Real.sin x ∧ Real.sin x < b * x) :
  (∀ a' : ℝ, (∀ x ∈ Set.Ioo 0 (π/2), a' * x < Real.sin x) → a' ≤ 2/π) ∧
  (∀ b' : ℝ, (∀ x ∈ Set.Ioo 0 (π/2), Real.sin x < b' * x) → b' ≥ 1) := by
sorry

end max_a_min_b_for_sin_inequality_l1746_174698


namespace system_solution_existence_l1746_174684

theorem system_solution_existence (a : ℝ) : 
  (∃ b x y : ℝ, x^2 + y^2 + 2*a*(a + y - x) = 49 ∧ 
                y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)) ↔ 
  -24 ≤ a ∧ a ≤ 24 := by
sorry

end system_solution_existence_l1746_174684
