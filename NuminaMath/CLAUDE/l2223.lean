import Mathlib

namespace NUMINAMATH_CALUDE_prob_even_and_greater_than_10_l2223_222313

/-- Represents a wheel with even and odd numbers -/
structure Wheel where
  evenCount : ℕ
  oddCount : ℕ

/-- Calculates the probability of selecting an even number from a wheel -/
def probEven (w : Wheel) : ℚ :=
  w.evenCount / (w.evenCount + w.oddCount)

/-- Calculates the probability of selecting an odd number from a wheel -/
def probOdd (w : Wheel) : ℚ :=
  w.oddCount / (w.evenCount + w.oddCount)

/-- The wheels used in the problem -/
def wheelA : Wheel := ⟨3, 5⟩
def wheelB : Wheel := ⟨2, 6⟩

/-- The probability that the sum of selected numbers is even -/
def probEvenSum : ℚ :=
  probEven wheelA * probEven wheelB + probOdd wheelA * probOdd wheelB

/-- The conditional probability that an even sum is greater than 10 -/
def probGreaterThan10GivenEven : ℚ := 1/3

/-- The main theorem to prove -/
theorem prob_even_and_greater_than_10 :
  probEvenSum * probGreaterThan10GivenEven = 3/16 := by
  sorry


end NUMINAMATH_CALUDE_prob_even_and_greater_than_10_l2223_222313


namespace NUMINAMATH_CALUDE_equation_solution_l2223_222330

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4 ↔ 
  x = (-3 + Real.sqrt 13) / 4 ∨ x = (-3 - Real.sqrt 13) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2223_222330


namespace NUMINAMATH_CALUDE_purchase_total_l2223_222335

/-- The total amount spent on a vacuum cleaner and dishwasher after applying a coupon -/
theorem purchase_total (vacuum_cost dishwasher_cost coupon_value : ℕ) : 
  vacuum_cost = 250 → 
  dishwasher_cost = 450 → 
  coupon_value = 75 → 
  vacuum_cost + dishwasher_cost - coupon_value = 625 := by
sorry

end NUMINAMATH_CALUDE_purchase_total_l2223_222335


namespace NUMINAMATH_CALUDE_veranda_area_l2223_222397

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) :
  room_length = 20 ∧ room_width = 12 ∧ veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 144 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l2223_222397


namespace NUMINAMATH_CALUDE_triangle_similarity_implies_pc_length_l2223_222395

/-- Triangle ABC with sides AB, BC, and CA -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- Point P on the extension of BC -/
def P : Type := Unit

/-- The length of PC -/
def PC (t : Triangle) (p : P) : ℝ := sorry

/-- Similarity of triangles PAB and PCA -/
def similar_triangles (t : Triangle) (p : P) : Prop := sorry

theorem triangle_similarity_implies_pc_length 
  (t : Triangle) 
  (p : P) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 9) 
  (h3 : t.CA = 7) 
  (h4 : similar_triangles t p) : 
  PC t p = 1.5 := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_implies_pc_length_l2223_222395


namespace NUMINAMATH_CALUDE_range_of_a_l2223_222354

def p (a : ℝ) : Prop := ∀ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 2 < 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  a ∈ Set.Icc (-Real.sqrt 8) (-1) ∪ Set.Ioo (Real.sqrt 8) 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2223_222354


namespace NUMINAMATH_CALUDE_min_value_shifted_function_l2223_222369

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + 5 - c

theorem min_value_shifted_function 
  (c : ℝ) 
  (h : ∃ (m : ℝ), ∀ (x : ℝ), f x c ≥ m ∧ ∃ (x₀ : ℝ), f x₀ c = m) 
  (h_min : ∃ (x₀ : ℝ), f x₀ c = 2) :
  ∃ (m : ℝ), (∀ (x : ℝ), f (x - 3) c ≥ m) ∧ (∃ (x₁ : ℝ), f (x₁ - 3) c = m) ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_shifted_function_l2223_222369


namespace NUMINAMATH_CALUDE_equation_solution_and_condition_l2223_222362

theorem equation_solution_and_condition :
  ∃ x : ℝ,
    (8 * x^(1/3) - 4 * (x / x^(2/3)) = 12 + 2 * x^(1/3)) ∧
    (x ≥ Real.sqrt 144) ∧
    (x = 216) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_condition_l2223_222362


namespace NUMINAMATH_CALUDE_age_difference_l2223_222321

/-- Given the ages of three people A, B, and C, prove that A is 2 years older than B. -/
theorem age_difference (A B C : ℕ) : 
  B = 18 →
  B = 2 * C →
  A + B + C = 47 →
  A = B + 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2223_222321


namespace NUMINAMATH_CALUDE_three_planes_max_regions_l2223_222343

/-- The maximum number of regions into which n planes can divide 3D space -/
def maxRegions (n : ℕ) : ℕ := sorry

/-- Theorem: Three planes can divide 3D space into at most 8 regions -/
theorem three_planes_max_regions :
  maxRegions 3 = 8 := by sorry

end NUMINAMATH_CALUDE_three_planes_max_regions_l2223_222343


namespace NUMINAMATH_CALUDE_square_difference_l2223_222339

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2223_222339


namespace NUMINAMATH_CALUDE_server_performance_l2223_222363

/-- Represents the number of multiplications a server can perform per second -/
def multiplications_per_second : ℕ := 5000

/-- Represents the number of seconds in half an hour -/
def seconds_in_half_hour : ℕ := 1800

/-- Represents the total number of multiplications in half an hour -/
def total_multiplications : ℕ := multiplications_per_second * seconds_in_half_hour

/-- Theorem stating that the server performs 9 million multiplications in half an hour -/
theorem server_performance : total_multiplications = 9000000 := by
  sorry

end NUMINAMATH_CALUDE_server_performance_l2223_222363


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2223_222372

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (4 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2223_222372


namespace NUMINAMATH_CALUDE_current_waiting_room_count_l2223_222331

/-- The number of people in the interview room -/
def interview_room_count : ℕ := 5

/-- The number of people currently in the waiting room -/
def waiting_room_count : ℕ := 22

/-- The condition that if three more people arrive in the waiting room,
    the number becomes five times the number of people in the interview room -/
axiom waiting_room_condition :
  waiting_room_count + 3 = 5 * interview_room_count

theorem current_waiting_room_count :
  waiting_room_count = 22 :=
sorry

end NUMINAMATH_CALUDE_current_waiting_room_count_l2223_222331


namespace NUMINAMATH_CALUDE_second_project_depth_l2223_222364

/-- Represents the dimensions of a digging project -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project -/
def volume (p : DiggingProject) : ℝ :=
  p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject :=
  { depth := 100, length := 25, breadth := 30 }

/-- The second digging project with unknown depth -/
def project2 (depth : ℝ) : DiggingProject :=
  { depth := depth, length := 20, breadth := 50 }

theorem second_project_depth :
  ∃ d : ℝ, volume project1 = volume (project2 d) ∧ d = 75 := by
  sorry

end NUMINAMATH_CALUDE_second_project_depth_l2223_222364


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l2223_222324

theorem no_linear_term_condition (a b : ℝ) :
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + a * b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l2223_222324


namespace NUMINAMATH_CALUDE_root_in_interval_l2223_222377

-- Define the function f(x) = x³ - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 1.5 > 0) → ∃ x, x ∈ Set.Ioo 1 1.5 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2223_222377


namespace NUMINAMATH_CALUDE_function_inequality_l2223_222310

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_pos : ∀ x, f x > 0)
  (h_ineq : ∀ x, f x < x * deriv f x) :
  2 * f 1 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2223_222310


namespace NUMINAMATH_CALUDE_lawyer_percentage_l2223_222323

theorem lawyer_percentage (total : ℝ) (h1 : total > 0) : 
  let women_ratio : ℝ := 0.9
  let women_lawyer_prob : ℝ := 0.54
  let women_count : ℝ := women_ratio * total
  let lawyer_ratio : ℝ := women_lawyer_prob / women_ratio
  lawyer_ratio = 0.6 := by sorry

end NUMINAMATH_CALUDE_lawyer_percentage_l2223_222323


namespace NUMINAMATH_CALUDE_coin_collection_average_l2223_222350

theorem coin_collection_average (a₁ d n : ℕ) (h1 : a₁ = 10) (h2 : d = 10) (h3 : n = 7) :
  let sequence := fun i => a₁ + (i - 1) * d
  (sequence 1 + sequence n) / 2 = 40 := by sorry

end NUMINAMATH_CALUDE_coin_collection_average_l2223_222350


namespace NUMINAMATH_CALUDE_fifth_subject_score_l2223_222374

theorem fifth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℚ) :
  s1 = 50 →
  s2 = 60 →
  s3 = 70 →
  s4 = 80 →
  avg = 68 →
  (s1 + s2 + s3 + s4 : ℚ) / 4 + 80 / 5 = avg :=
by sorry

end NUMINAMATH_CALUDE_fifth_subject_score_l2223_222374


namespace NUMINAMATH_CALUDE_divisible_by_six_ratio_l2223_222349

theorem divisible_by_six_ratio (n : ℕ) : n = 120 →
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / n = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_ratio_l2223_222349


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2223_222378

/-- A homogeneous polynomial of degree n in two variables -/
noncomputable def HomogeneousPolynomial (n : ℕ) := (ℝ → ℝ → ℝ)

/-- The property of being homogeneous of degree n -/
def IsHomogeneous (P : HomogeneousPolynomial n) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

/-- The second condition from the problem -/
def SatisfiesCondition2 (P : HomogeneousPolynomial n) : Prop :=
  ∀ (a b c : ℝ), P (a + b) c + P (b + c) a + P (c + a) b = 0

/-- The third condition from the problem -/
def SatisfiesCondition3 (P : HomogeneousPolynomial n) : Prop :=
  P 1 0 = 1

/-- The theorem statement -/
theorem polynomial_characterization (n : ℕ) (P : HomogeneousPolynomial n)
  (h1 : IsHomogeneous P)
  (h2 : SatisfiesCondition2 P)
  (h3 : SatisfiesCondition3 P) :
  ∀ (x y : ℝ), P x y = (x + y)^(n - 1) * (x - 2*y) :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2223_222378


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l2223_222319

/-- Given a cylinder with height 4 and base area 9π, its lateral area is 24π. -/
theorem cylinder_lateral_area (h : ℝ) (base_area : ℝ) :
  h = 4 → base_area = 9 * Real.pi → 2 * Real.pi * (Real.sqrt (base_area / Real.pi)) * h = 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l2223_222319


namespace NUMINAMATH_CALUDE_negation_equivalence_l2223_222389

theorem negation_equivalence : 
  (¬(∀ x : ℝ, |x| < 2 → x < 2)) ↔ (∀ x : ℝ, |x| ≥ 2 → x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2223_222389


namespace NUMINAMATH_CALUDE_rectangle_area_l2223_222314

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2223_222314


namespace NUMINAMATH_CALUDE_point_not_on_line_l2223_222315

theorem point_not_on_line (m k : ℝ) (h1 : m * k > 0) :
  ¬(∃ (x y : ℝ), x = 2000 ∧ y = 0 ∧ y = m * x + k) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l2223_222315


namespace NUMINAMATH_CALUDE_count_specific_coin_toss_sequences_l2223_222396

def coin_toss_sequences (n : ℕ) (th ht tt hh : ℕ) : ℕ :=
  Nat.choose 4 2 * Nat.choose 8 3 * Nat.choose 5 4 * Nat.choose 11 5

theorem count_specific_coin_toss_sequences :
  coin_toss_sequences 15 2 3 4 5 = 775360 := by
  sorry

end NUMINAMATH_CALUDE_count_specific_coin_toss_sequences_l2223_222396


namespace NUMINAMATH_CALUDE_four_true_propositions_l2223_222357

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle and side length for a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
def side_length (t : Triangle) (v : Fin 3) : ℝ := sorry

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties for a quadrilateral
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

-- The four propositions
theorem four_true_propositions :
  (∀ t : Triangle, angle t 2 > angle t 1 → side_length t 0 > side_length t 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a = 0 ∨ b ≠ 0) ∧
  (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0) ∧
  (∀ q : Quadrilateral, diagonals_bisect q → is_parallelogram q) :=
sorry

end NUMINAMATH_CALUDE_four_true_propositions_l2223_222357


namespace NUMINAMATH_CALUDE_magnitude_of_b_l2223_222303

/-- Given vectors a and b in ℝ², prove that |b| = √2 under the given conditions -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  a = (-Real.sqrt 3, 1) →
  (a.1 + 2 * b.1) * a.1 + (a.2 + 2 * b.2) * a.2 = 0 →
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 0 →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_b_l2223_222303


namespace NUMINAMATH_CALUDE_first_load_pieces_l2223_222385

theorem first_load_pieces (total : ℕ) (num_small_loads : ℕ) (pieces_per_small_load : ℕ) 
    (h1 : total = 47)
    (h2 : num_small_loads = 5)
    (h3 : pieces_per_small_load = 6) : 
  total - (num_small_loads * pieces_per_small_load) = 17 := by
  sorry

end NUMINAMATH_CALUDE_first_load_pieces_l2223_222385


namespace NUMINAMATH_CALUDE_miles_difference_l2223_222327

/-- Given that Gervais drove an average of 315 miles for 3 days and Henri drove a total of 1,250 miles,
    prove that Henri drove 305 miles farther than Gervais. -/
theorem miles_difference (gervais_avg_daily : ℕ) (gervais_days : ℕ) (henri_total : ℕ) : 
  gervais_avg_daily = 315 → gervais_days = 3 → henri_total = 1250 → 
  henri_total - (gervais_avg_daily * gervais_days) = 305 := by
sorry

end NUMINAMATH_CALUDE_miles_difference_l2223_222327


namespace NUMINAMATH_CALUDE_solution_when_k_gt_neg_one_no_solution_when_k_eq_neg_one_solution_when_k_lt_neg_one_k_upper_bound_l2223_222316

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (1 - k)*x + 2 - k

/-- Theorem stating the solution for f(x) < 2 when k > -1 -/
theorem solution_when_k_gt_neg_one (k : ℝ) (x : ℝ) (h : k > -1) :
  f k x < 2 ↔ -1 < x ∧ x < k :=
sorry

/-- Theorem stating there's no solution for f(x) < 2 when k = -1 -/
theorem no_solution_when_k_eq_neg_one (x : ℝ) :
  ¬(f (-1) x < 2) :=
sorry

/-- Theorem stating the solution for f(x) < 2 when k < -1 -/
theorem solution_when_k_lt_neg_one (k : ℝ) (x : ℝ) (h : k < -1) :
  f k x < 2 ↔ k < x ∧ x < -1 :=
sorry

/-- Theorem stating the upper bound of k when f(n) + 11 ≥ 0 for all natural numbers n -/
theorem k_upper_bound (k : ℝ) (h : ∀ (n : ℕ), f k n + 11 ≥ 0) :
  k ≤ 25/4 :=
sorry

end NUMINAMATH_CALUDE_solution_when_k_gt_neg_one_no_solution_when_k_eq_neg_one_solution_when_k_lt_neg_one_k_upper_bound_l2223_222316


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l2223_222311

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem
theorem M_intersect_N_eq_open_interval : M ∩ N = {x | -2 < x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_interval_l2223_222311


namespace NUMINAMATH_CALUDE_smallest_quadratic_nonresidue_l2223_222325

theorem smallest_quadratic_nonresidue (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ x : Nat, x < Nat.sqrt p + 1 ∧ x > 0 ∧ ¬ ∃ y : Nat, (y ^ 2) % p = x % p :=
sorry

end NUMINAMATH_CALUDE_smallest_quadratic_nonresidue_l2223_222325


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l2223_222371

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 64) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l2223_222371


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_l2223_222304

/-- Given a truncated pyramid with base areas S₁ and S₂ (where S₁ < S₂) and volume V,
    the volume of the complete pyramid is (V * S₂ * √S₂) / (S₂ * √S₂ - S₁ * √S₁) -/
theorem truncated_pyramid_volume (S₁ S₂ V : ℝ) (h₁ : 0 < S₁) (h₂ : S₁ < S₂) (h₃ : 0 < V) :
  let complete_volume := (V * S₂ * Real.sqrt S₂) / (S₂ * Real.sqrt S₂ - S₁ * Real.sqrt S₁)
  ∃ (h : ℝ), h > 0 ∧ complete_volume = (1 / 3) * S₂ * h :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_l2223_222304


namespace NUMINAMATH_CALUDE_total_selection_schemes_l2223_222320

/-- The number of elective courses in each category (physical education and art) -/
def num_courses_per_category : ℕ := 4

/-- The minimum number of courses a student can choose -/
def min_courses : ℕ := 2

/-- The maximum number of courses a student can choose -/
def max_courses : ℕ := 3

/-- The number of categories (physical education and art) -/
def num_categories : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The total number of different course selection schemes is 64 -/
theorem total_selection_schemes : 
  (choose num_courses_per_category 1 * choose num_courses_per_category 1) + 
  (choose num_courses_per_category 2 * choose num_courses_per_category 1 + 
   choose num_courses_per_category 1 * choose num_courses_per_category 2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_selection_schemes_l2223_222320


namespace NUMINAMATH_CALUDE_min_sum_of_product_1800_l2223_222302

theorem min_sum_of_product_1800 (a b c : ℕ+) (h : a * b * c = 1800) :
  (∀ x y z : ℕ+, x * y * z = 1800 → a + b + c ≤ x + y + z) ∧ a + b + c = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1800_l2223_222302


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2223_222307

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2223_222307


namespace NUMINAMATH_CALUDE_percentage_of_36_l2223_222345

theorem percentage_of_36 : (33 + 1 / 3 : ℚ) / 100 * 36 = 12 := by sorry

end NUMINAMATH_CALUDE_percentage_of_36_l2223_222345


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l2223_222332

/-- Given that 0.5% of a is 85 paise and 0.75% of b is 150 paise, 
    prove that the ratio of a to b is 17:20 -/
theorem ratio_a_to_b (a b : ℚ) 
  (ha : (5 / 1000) * a = 85 / 100) 
  (hb : (75 / 10000) * b = 150 / 100) : 
  a / b = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l2223_222332


namespace NUMINAMATH_CALUDE_orange_boxes_theorem_l2223_222384

/-- Given 56 oranges that need to be stored in boxes, with each box containing 7 oranges,
    prove that the number of boxes required is 8. -/
theorem orange_boxes_theorem (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 56) (h2 : oranges_per_box = 7) :
  total_oranges / oranges_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_theorem_l2223_222384


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l2223_222333

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  let digits := n.digits 8
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the theorem
theorem pirate_treasure_sum :
  let silk := 5267
  let stones := 6712
  let spices := 327
  base8ToBase10 silk + base8ToBase10 stones + base8ToBase10 spices = 6488 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_sum_l2223_222333


namespace NUMINAMATH_CALUDE_horner_method_for_f_l2223_222334

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_method_for_f :
  f 2 = horner [2, 3, 0, 5, -4] 2 ∧ horner [2, 3, 0, 5, -4] 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l2223_222334


namespace NUMINAMATH_CALUDE_last_digit_n_power_9999_minus_5555_l2223_222375

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_n_power_9999_minus_5555 (n : ℕ) : 
  last_digit (n^9999 - n^5555) = 0 :=
sorry

end NUMINAMATH_CALUDE_last_digit_n_power_9999_minus_5555_l2223_222375


namespace NUMINAMATH_CALUDE_x_squared_plus_x_is_quadratic_binomial_l2223_222390

/-- A quadratic binomial is a polynomial of degree 2 with two terms. -/
def is_quadratic_binomial (p : Polynomial ℝ) : Prop :=
  p.degree = 2 ∧ p.support.card = 2

/-- x^2 + x is a quadratic binomial -/
theorem x_squared_plus_x_is_quadratic_binomial :
  is_quadratic_binomial (X^2 + X : Polynomial ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_x_is_quadratic_binomial_l2223_222390


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l2223_222348

/-- Given a cubic function f(x) with an extreme value at x = 1, prove that a + b = -7 -/
theorem extreme_value_cubic (a b : ℝ) : 
  let f := fun x : ℝ ↦ x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧ 
  (∃ (ε : ℝ), ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  f 1 = 10 →
  a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l2223_222348


namespace NUMINAMATH_CALUDE_kylie_picked_220_apples_l2223_222392

/-- The number of apples Kylie picked in the first hour -/
def first_hour_apples : ℕ := 66

/-- The number of apples Kylie picked in the second hour -/
def second_hour_apples : ℕ := 2 * first_hour_apples

/-- The number of apples Kylie picked in the third hour -/
def third_hour_apples : ℕ := first_hour_apples / 3

/-- The total number of apples Kylie picked -/
def total_apples : ℕ := first_hour_apples + second_hour_apples + third_hour_apples

/-- Theorem stating that the total number of apples Kylie picked is 220 -/
theorem kylie_picked_220_apples : total_apples = 220 := by
  sorry

end NUMINAMATH_CALUDE_kylie_picked_220_apples_l2223_222392


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l2223_222301

/-- A function f is monotonic on an interval [a,b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The main theorem stating the range of 'a' for which the given function
    is monotonic on the interval [1,3]. -/
theorem monotonic_function_a_range :
  ∀ a : ℝ,
  (IsMonotonic (fun x => (1/3) * x^3 + a * x^2 + 5 * x + 6) 1 3) →
  (a ≤ -3 ∨ a ≥ -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l2223_222301


namespace NUMINAMATH_CALUDE_f_zero_eq_two_l2223_222370

/-- The function f(x) with parameter a -/
def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

/-- Theorem: If f represents a straight line, then f(0) = 2 -/
theorem f_zero_eq_two (a : ℝ) (h : ∀ x y : ℝ, f a x - f a y = (f a 1 - f a 0) * (x - y)) : 
  f a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_eq_two_l2223_222370


namespace NUMINAMATH_CALUDE_total_colors_needed_l2223_222388

/-- Represents the number of moons for each planet in the solar system -/
def moons : Fin 8 → ℕ
  | 0 => 0  -- Mercury
  | 1 => 0  -- Venus
  | 2 => 1  -- Earth
  | 3 => 2  -- Mars
  | 4 => 79 -- Jupiter
  | 5 => 82 -- Saturn
  | 6 => 27 -- Uranus
  | 7 => 14 -- Neptune

/-- The number of planets in the solar system -/
def num_planets : ℕ := 8

/-- The number of people coloring -/
def num_people : ℕ := 3

/-- The total number of celestial bodies (planets and moons) -/
def total_bodies : ℕ := num_planets + (Finset.sum Finset.univ moons)

/-- Theorem stating the total number of colors needed -/
theorem total_colors_needed : num_people * total_bodies = 639 := by
  sorry


end NUMINAMATH_CALUDE_total_colors_needed_l2223_222388


namespace NUMINAMATH_CALUDE_triangle_property_l2223_222387

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.b + t.c = 2 * t.a * Real.sin (t.C + π/6)) : 
  t.A = π/3 ∧ 1 < (t.b^2 + t.c^2) / t.a^2 ∧ (t.b^2 + t.c^2) / t.a^2 ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l2223_222387


namespace NUMINAMATH_CALUDE_infinite_solutions_of_diophantine_equation_l2223_222398

theorem infinite_solutions_of_diophantine_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ × ℕ)), Set.Infinite S ∧
    ∀ (x y z t : ℕ), (x, y, z, t) ∈ S → x^2 + y^2 = 5*(z^2 + t^2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_of_diophantine_equation_l2223_222398


namespace NUMINAMATH_CALUDE_sin_2x_value_l2223_222347

theorem sin_2x_value (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l2223_222347


namespace NUMINAMATH_CALUDE_peter_speed_proof_l2223_222336

/-- Peter's speed in miles per hour -/
def peter_speed : ℝ := 5

/-- Juan's speed in miles per hour -/
def juan_speed : ℝ := peter_speed + 3

/-- Time traveled in hours -/
def time : ℝ := 1.5

/-- Total distance between Juan and Peter after traveling -/
def total_distance : ℝ := 19.5

theorem peter_speed_proof :
  peter_speed * time + juan_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_peter_speed_proof_l2223_222336


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_l2223_222340

theorem infinite_perfect_squares (k : ℕ+) :
  ∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ n ∈ S, ∃ m : ℕ+, (n * 2^k.val : ℤ) - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_l2223_222340


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l2223_222346

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let rectangle1_height : ℝ := 6
  let rectangle1_width : ℝ := 1
  let rectangle2_height : ℝ := 3
  let rectangle2_width : ℝ := 3
  let volume1 : ℝ := π * rectangle1_height^2 * rectangle1_width
  let volume2 : ℝ := π * rectangle2_height^2 * rectangle2_width
  let total_volume : ℝ := volume1 + volume2
  total_volume = 63 * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l2223_222346


namespace NUMINAMATH_CALUDE_probability_of_specific_tile_arrangement_l2223_222366

theorem probability_of_specific_tile_arrangement :
  let total_tiles : ℕ := 6
  let x_tiles : ℕ := 4
  let o_tiles : ℕ := 2
  let specific_arrangement := [true, true, false, true, false, true]
  
  (x_tiles + o_tiles = total_tiles) →
  (List.length specific_arrangement = total_tiles) →
  
  (probability_of_arrangement : ℚ) =
    (x_tiles.choose 2 * o_tiles.choose 1 * x_tiles.choose 1 * o_tiles.choose 1 * x_tiles.choose 1) /
    total_tiles.factorial →
  
  probability_of_arrangement = 1 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_tile_arrangement_l2223_222366


namespace NUMINAMATH_CALUDE_mouse_meiosis_observation_l2223_222356

/-- Available materials for mouse cell meiosis observation --/
inductive Material
  | MouseKidney
  | MouseTestis
  | MouseLiver
  | SudanIIIStain
  | GentianVioletSolution
  | JanusGreenBStain
  | DissociationFixativeSolution

/-- Types of cells produced during meiosis --/
inductive DaughterCell
  | Spermatogonial
  | SecondarySpermatocyte
  | Spermatid

/-- Theorem for correct mouse cell meiosis observation procedure --/
theorem mouse_meiosis_observation 
  (available_materials : List Material)
  (meiosis_occurs_in_gonads : Bool)
  (spermatogonial_cells_undergo_mitosis_and_meiosis : Bool) :
  (MouseTestis ∈ available_materials) →
  (DissociationFixativeSolution ∈ available_materials) →
  (GentianVioletSolution ∈ available_materials) →
  meiosis_occurs_in_gonads →
  spermatogonial_cells_undergo_mitosis_and_meiosis →
  (correct_tissue = MouseTestis) ∧
  (post_hypotonic_solution = DissociationFixativeSolution) ∧
  (staining_solution = GentianVioletSolution) ∧
  (daughter_cells = [DaughterCell.Spermatogonial, DaughterCell.SecondarySpermatocyte, DaughterCell.Spermatid]) := by
  sorry

end NUMINAMATH_CALUDE_mouse_meiosis_observation_l2223_222356


namespace NUMINAMATH_CALUDE_eve_gift_cost_is_135_l2223_222308

/-- The cost of Eve's gifts for her nieces --/
def eve_gift_cost : ℝ :=
  let hand_mitts : ℝ := 14
  let apron : ℝ := 16
  let utensils : ℝ := 10
  let knife : ℝ := 2 * utensils
  let cost_per_niece : ℝ := hand_mitts + apron + utensils + knife
  let total_cost : ℝ := 3 * cost_per_niece
  let discount_rate : ℝ := 0.25
  let discounted_cost : ℝ := total_cost * (1 - discount_rate)
  discounted_cost

theorem eve_gift_cost_is_135 : eve_gift_cost = 135 := by
  sorry

end NUMINAMATH_CALUDE_eve_gift_cost_is_135_l2223_222308


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l2223_222355

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1010_is_10 :
  binary_to_decimal [true, false, true, false] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l2223_222355


namespace NUMINAMATH_CALUDE_solution_to_equation_l2223_222365

theorem solution_to_equation : ∃! x : ℝ, (x - 3)^3 = (1/27)⁻¹ := by
  use 6
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2223_222365


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2223_222318

theorem imaginary_part_of_complex_expression (z : ℂ) (h : z = 3 + 4*I) : 
  Complex.im (z + Complex.abs z / z) = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2223_222318


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2223_222376

/-- The set of real numbers a for which the system of equations has at least one solution -/
def SolutionSet : Set ℝ :=
  {a | ∃ x y, x - 1 = a * (y^3 - 1) ∧
               2 * x / (|y^3| + y^3) = Real.sqrt x ∧
               y > 0 ∧
               x ≥ 0}

/-- Theorem stating that the SolutionSet is equal to the union of three intervals -/
theorem solution_set_characterization :
  SolutionSet = {a | a < 0} ∪ {a | 0 ≤ a ∧ a ≤ 1} ∪ {a | a > 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2223_222376


namespace NUMINAMATH_CALUDE_right_triangle_polynomial_roots_l2223_222351

theorem right_triangle_polynomial_roots (B : ℝ) : 
  (∃ u v w : ℝ, 
    (u^3 - 14*u^2 + B*u - 84 = 0) ∧ 
    (v^3 - 14*v^2 + B*v - 84 = 0) ∧ 
    (w^3 - 14*w^2 + B*w - 84 = 0) ∧
    (u^2 + v^2 = w^2 ∨ u^2 + w^2 = v^2 ∨ v^2 + w^2 = u^2)) →
  B = 62 := by sorry

end NUMINAMATH_CALUDE_right_triangle_polynomial_roots_l2223_222351


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2223_222394

theorem cube_volume_from_surface_area :
  ∀ s : ℝ, 6 * s^2 = 150 → s^3 = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2223_222394


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2223_222393

/-- Given two digits A and B in base d > 7, if AB + BA = 202 in base d, then A - B = 2 in base d -/
theorem digit_difference_in_base_d (d : ℕ) (A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  (A * d + B) + (B * d + A) = 2 * d^2 + 2 →
  A - B = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2223_222393


namespace NUMINAMATH_CALUDE_product_mod_seven_l2223_222306

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2223_222306


namespace NUMINAMATH_CALUDE_carbonated_water_percent_in_specific_mixture_l2223_222342

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (solution1_volume_percent : ℝ)

/-- Calculates the percentage of carbonated water in a mixture -/
def carbonated_water_percent_in_mixture (m : Mixture) : ℝ :=
  m.solution1.carbonated_water_percent * m.solution1_volume_percent +
  m.solution2.carbonated_water_percent * (1 - m.solution1_volume_percent)

/-- Theorem stating that the percentage of carbonated water in the specific mixture is 67.5% -/
theorem carbonated_water_percent_in_specific_mixture :
  let solution1 : Solution := ⟨0.8⟩
  let solution2 : Solution := ⟨0.55⟩
  let mixture : Mixture := ⟨solution1, solution2, 0.5⟩
  carbonated_water_percent_in_mixture mixture = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_carbonated_water_percent_in_specific_mixture_l2223_222342


namespace NUMINAMATH_CALUDE_min_value_expression_l2223_222382

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≥ 6 / (2 * Real.sqrt 3 + Real.sqrt 6) ∧
  (∃ x₀ > 0, (x₀^2 + 3 - Real.sqrt (x₀^4 + 9)) / x₀ = 6 / (2 * Real.sqrt 3 + Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2223_222382


namespace NUMINAMATH_CALUDE_factor_proof_l2223_222338

theorem factor_proof :
  (∃ n : ℤ, 28 = 4 * n) ∧ (∃ m : ℤ, 162 = 9 * m) := by sorry

end NUMINAMATH_CALUDE_factor_proof_l2223_222338


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_four_l2223_222386

theorem no_linear_term_implies_m_equals_four :
  ∀ m : ℝ, (∀ x : ℝ, 2*x^2 + m*x = 4*x + 2) →
  (∀ x : ℝ, ∃ a b c : ℝ, a*x^2 + c = 0 ∧ 2*x^2 + m*x = 4*x + 2) →
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_four_l2223_222386


namespace NUMINAMATH_CALUDE_equality_of_expressions_l2223_222367

theorem equality_of_expressions (x : ℝ) (hx : x > 0) :
  x^(x+1) + x^(x+1) = 2*x^(x+1) ∧
  x^(x+1) + x^(x+1) ≠ x^(2*x+2) ∧
  x^(x+1) + x^(x+1) ≠ (2*x)^(x+1) ∧
  x^(x+1) + x^(x+1) ≠ (2*x)^(2*x+2) :=
by sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l2223_222367


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2223_222383

def M (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def P (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (M a) ∩ (P a) = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2223_222383


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2223_222300

def M : Set ℤ := {0, 1, 2, 3, 4}
def N : Set ℤ := {-2, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2223_222300


namespace NUMINAMATH_CALUDE_markeesha_friday_sales_l2223_222353

/-- Proves that Markeesha sold 30 boxes on Friday given the conditions of the problem -/
theorem markeesha_friday_sales : ∀ (friday : ℕ), 
  (∃ (saturday sunday : ℕ),
    saturday = 2 * friday ∧
    sunday = saturday - 15 ∧
    friday + saturday + sunday = 135) →
  friday = 30 := by
sorry

end NUMINAMATH_CALUDE_markeesha_friday_sales_l2223_222353


namespace NUMINAMATH_CALUDE_quadratic_from_means_l2223_222328

theorem quadratic_from_means (α β : ℝ) : 
  (α + β) / 2 = 8 → 
  (α * β) = 15^2 → 
  ∀ x, x^2 - 16*x + 225 = 0 ↔ (x = α ∨ x = β) := by
sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l2223_222328


namespace NUMINAMATH_CALUDE_alice_class_size_l2223_222358

/-- The number of students in Alice's white water rafting class -/
def num_students : ℕ := 40

/-- The number of instructors, including Alice -/
def num_instructors : ℕ := 10

/-- The number of life vests Alice has on hand -/
def vests_on_hand : ℕ := 20

/-- The percentage of students bringing their own life vests -/
def percent_students_with_vests : ℚ := 1/5

/-- The additional number of life vests Alice needs to get -/
def additional_vests_needed : ℕ := 22

theorem alice_class_size :
  num_students = 40 ∧
  (num_students + num_instructors) * (1 - percent_students_with_vests) =
    vests_on_hand + additional_vests_needed :=
by sorry

end NUMINAMATH_CALUDE_alice_class_size_l2223_222358


namespace NUMINAMATH_CALUDE_function_supremum_m_range_l2223_222359

/-- The supremum of the given function for positive real x and y is 25/4 -/
theorem function_supremum : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) ≤ 25/4) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) = 25/4) :=
by sorry

/-- The range of m for which the inequality always holds is (25, +∞) -/
theorem m_range (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 
    (4*x^4 + 17*x^2*y + 4*y^2) / (x^4 + 2*x^2*y + y^2) < m/4) ↔ 
  m > 25 :=
by sorry

end NUMINAMATH_CALUDE_function_supremum_m_range_l2223_222359


namespace NUMINAMATH_CALUDE_flywheel_rotation_l2223_222312

-- Define the angular displacement function
def φ (t : ℝ) : ℝ := 8 * t - 0.5 * t^2

-- Define the angular velocity function
def ω (t : ℝ) : ℝ := 8 - t

theorem flywheel_rotation (t : ℝ) :
  -- 1. The angular velocity is the derivative of the angular displacement
  (deriv φ) t = ω t ∧
  -- 2. The angular velocity at t = 3 seconds is 5 rad/s
  ω 3 = 5 ∧
  -- 3. The flywheel stops rotating at t = 8 seconds
  ω 8 = 0 := by
  sorry


end NUMINAMATH_CALUDE_flywheel_rotation_l2223_222312


namespace NUMINAMATH_CALUDE_frequency_of_fifth_group_l2223_222373

theorem frequency_of_fifth_group 
  (total_students : ℕ) 
  (group1 group2 group3 group4 : ℕ) 
  (h1 : total_students = 40)
  (h2 : group1 = 12)
  (h3 : group2 = 10)
  (h4 : group3 = 6)
  (h5 : group4 = 8) :
  total_students - (group1 + group2 + group3 + group4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_fifth_group_l2223_222373


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2223_222341

theorem sin_cos_identity (x : ℝ) : 
  Real.sin (3 * x - Real.pi / 4) = Real.cos (3 * x - 3 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2223_222341


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2223_222399

/-- Represents the simple interest calculation problem --/
theorem simple_interest_problem (P T : ℝ) (h1 : P = 2500) (h2 : T = 5) : 
  let SI := P - 2000
  let R := (SI * 100) / (P * T)
  R = 4 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2223_222399


namespace NUMINAMATH_CALUDE_octal_subtraction_l2223_222337

def octal_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_octal (n : ℕ) : ℕ := sorry

theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 5374 - octal_to_decimal 2645) = 1527 := by sorry

end NUMINAMATH_CALUDE_octal_subtraction_l2223_222337


namespace NUMINAMATH_CALUDE_chocolate_chip_cookies_l2223_222391

theorem chocolate_chip_cookies (cookies_per_bag : ℕ) (baggies : ℕ) (oatmeal_cookies : ℕ) :
  cookies_per_bag = 5 →
  baggies = 7 →
  oatmeal_cookies = 2 →
  cookies_per_bag * baggies - oatmeal_cookies = 33 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookies_l2223_222391


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2223_222309

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) :
  x^3 + y^3 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2223_222309


namespace NUMINAMATH_CALUDE_repeating_block_length_l2223_222361

/-- The least number of digits in the repeating block of the decimal expansion of 7/13 -/
def least_repeating_digits_7_13 : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 7/13

theorem repeating_block_length :
  least_repeating_digits_7_13 = 6 ∧
  ∃ (n : ℕ) (d : ℕ+), fraction * 10^least_repeating_digits_7_13 = n / d ∧
  fraction * 10^least_repeating_digits_7_13 - fraction = ↑(n - 7) / 13 :=
sorry

end NUMINAMATH_CALUDE_repeating_block_length_l2223_222361


namespace NUMINAMATH_CALUDE_strictly_increasing_inverse_sum_identity_l2223_222317

theorem strictly_increasing_inverse_sum_identity 
  (f : ℝ → ℝ) 
  (h_incr : ∀ x y, x < y → f x < f y) 
  (h_inv : Function.Bijective f) 
  (h_sum : ∀ x, f x + (Function.invFun f) x = 2 * x) : 
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_inverse_sum_identity_l2223_222317


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2223_222380

theorem isosceles_triangle_perimeter
  (equilateral_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h1 : equilateral_perimeter = 60)
  (h2 : isosceles_base = 10)
  : ℝ := by
  sorry

#check isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2223_222380


namespace NUMINAMATH_CALUDE_min_value_theorem_l2223_222352

theorem min_value_theorem (a : ℝ) (h : a > 1) :
  a + 1 / (a - 1) ≥ 3 ∧ (a + 1 / (a - 1) = 3 ↔ a = 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2223_222352


namespace NUMINAMATH_CALUDE_cos_alpha_plus_17pi_12_l2223_222368

theorem cos_alpha_plus_17pi_12 (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_17pi_12_l2223_222368


namespace NUMINAMATH_CALUDE_root_product_expression_l2223_222344

theorem root_product_expression (p q r s : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + r = 0) → 
  (β^2 + p*β + r = 0) → 
  (γ^2 + q*γ + s = 0) → 
  (δ^2 + q*δ + s = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = (p-q)^4 * s^2 + 2*(p-q)^3 * s * (r-s) + (p-q)^2 * (r-s)^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l2223_222344


namespace NUMINAMATH_CALUDE_x_over_y_equals_negative_one_fourth_l2223_222381

theorem x_over_y_equals_negative_one_fourth (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y)^5 + x^5 + 4 * x + y = 0) : x / y = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_equals_negative_one_fourth_l2223_222381


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l2223_222326

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back --/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back --/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 15)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  oranges_to_put_back fs = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l2223_222326


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l2223_222360

theorem log_expression_equals_two :
  Real.log 4 + Real.log 5 * Real.log 20 + (Real.log 5)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l2223_222360


namespace NUMINAMATH_CALUDE_cos_sum_plus_cos_diff_l2223_222329

theorem cos_sum_plus_cos_diff (x y : ℝ) : 
  Real.cos (x + y) + Real.cos (x - y) = 2 * Real.cos x * Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_plus_cos_diff_l2223_222329


namespace NUMINAMATH_CALUDE_product_47_33_l2223_222305

theorem product_47_33 : 47 * 33 = 1551 := by
  sorry

end NUMINAMATH_CALUDE_product_47_33_l2223_222305


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_condition_l2223_222379

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the relation for a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the intersection relation for lines
variable (line_intersect : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_sufficient_condition
  (α β : Plane) (m n l₁ l₂ : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_α : line_in_plane n α)
  (h_l₁_in_β : line_in_plane l₁ β)
  (h_l₂_in_β : line_in_plane l₂ β)
  (h_l₁_l₂_intersect : line_intersect l₁ l₂)
  (h_m_parallel_l₁ : line_parallel m l₁)
  (h_n_parallel_l₂ : line_parallel n l₂) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_condition_l2223_222379


namespace NUMINAMATH_CALUDE_problem_solution_l2223_222322

theorem problem_solution : ∃ x : ℝ, 
  (0.6 * x = 0.3 * (125 ^ (1/3 : ℝ)) + 27) ∧ x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2223_222322
