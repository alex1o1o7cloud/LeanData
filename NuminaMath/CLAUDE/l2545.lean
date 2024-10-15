import Mathlib

namespace NUMINAMATH_CALUDE_service_center_location_l2545_254537

/-- Represents a highway with exits and a service center -/
structure Highway where
  third_exit : ℝ
  tenth_exit : ℝ
  service_center : ℝ

/-- Theorem: Given a highway with the third exit at milepost 50 and the tenth exit at milepost 170,
    a service center located two-thirds of the way from the third exit to the tenth exit
    is at milepost 130. -/
theorem service_center_location (h : Highway)
  (h_third : h.third_exit = 50)
  (h_tenth : h.tenth_exit = 170)
  (h_service : h.service_center = h.third_exit + 2 / 3 * (h.tenth_exit - h.third_exit)) :
  h.service_center = 130 := by
  sorry

end NUMINAMATH_CALUDE_service_center_location_l2545_254537


namespace NUMINAMATH_CALUDE_new_years_party_assignments_l2545_254506

/-- The number of ways to assign teachers to classes -/
def assignTeachers (totalTeachers : ℕ) (numClasses : ℕ) (maxPerClass : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of assignments for the given conditions -/
theorem new_years_party_assignments :
  assignTeachers 6 2 4 = 50 := by sorry

end NUMINAMATH_CALUDE_new_years_party_assignments_l2545_254506


namespace NUMINAMATH_CALUDE_combination_equality_l2545_254542

theorem combination_equality (x : ℕ) : (Nat.choose 9 x = Nat.choose 9 (2*x - 3)) → (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_combination_equality_l2545_254542


namespace NUMINAMATH_CALUDE_ball_distribution_l2545_254594

theorem ball_distribution (n : ℕ) (k : ℕ) :
  -- Part (a): No empty boxes
  (Nat.choose (n + k - 1) (k - 1) = Nat.choose 19 5 → n = 20 ∧ k = 6) ∧
  -- Part (b): Some boxes can be empty
  (Nat.choose (n + k - 1) (k - 1) = Nat.choose 25 5 → n = 20 ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_l2545_254594


namespace NUMINAMATH_CALUDE_jennifer_dogs_count_l2545_254523

/-- The number of dogs Jennifer has -/
def number_of_dogs : ℕ := 2

/-- Time in minutes to groom each dog -/
def grooming_time_per_dog : ℕ := 20

/-- Number of days Jennifer grooms her dogs -/
def grooming_days : ℕ := 30

/-- Total time in hours Jennifer spends grooming in 30 days -/
def total_grooming_time_hours : ℕ := 20

theorem jennifer_dogs_count :
  number_of_dogs * grooming_time_per_dog * grooming_days = total_grooming_time_hours * 60 :=
by sorry

end NUMINAMATH_CALUDE_jennifer_dogs_count_l2545_254523


namespace NUMINAMATH_CALUDE_c_range_l2545_254596

def p (c : ℝ) : Prop := c^2 < c

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

def range_of_c (c : ℝ) : Prop :=
  (c > -1/2 ∧ c ≤ 0) ∨ (c ≥ 1/2 ∧ c < 1)

theorem c_range (c : ℝ) :
  (p c ∨ q c) ∧ ¬(p c ∧ q c) → range_of_c c :=
sorry

end NUMINAMATH_CALUDE_c_range_l2545_254596


namespace NUMINAMATH_CALUDE_jack_books_left_l2545_254565

/-- The number of books left in Jack's classics section -/
def books_left (authors : ℕ) (books_per_author : ℕ) (lent_books : ℕ) (misplaced_books : ℕ) : ℕ :=
  authors * books_per_author - (lent_books + misplaced_books)

theorem jack_books_left :
  books_left 10 45 17 8 = 425 := by
  sorry

end NUMINAMATH_CALUDE_jack_books_left_l2545_254565


namespace NUMINAMATH_CALUDE_tax_reduction_l2545_254549

theorem tax_reduction (T C X : ℝ) (h1 : T > 0) (h2 : C > 0) (h3 : X > 0) : 
  (T * (1 - X / 100) * (C * 1.2) = 0.84 * (T * C)) → X = 30 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_l2545_254549


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2545_254520

theorem system_solution_ratio (x y a b : ℝ) 
  (h1 : 6 * x - 4 * y = a)
  (h2 : 6 * y - 9 * x = b)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hb : b ≠ 0) :
  a / b = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2545_254520


namespace NUMINAMATH_CALUDE_maya_lifting_improvement_l2545_254519

theorem maya_lifting_improvement (america_initial : ℕ) (america_peak : ℕ) : 
  america_initial = 240 →
  america_peak = 300 →
  (america_peak / 2 : ℕ) - (america_initial / 4 : ℕ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_maya_lifting_improvement_l2545_254519


namespace NUMINAMATH_CALUDE_choose_captains_l2545_254508

theorem choose_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_captains_l2545_254508


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2545_254589

def A : Set ℤ := {x : ℤ | x^2 - 4*x ≤ 0}
def B : Set ℤ := {x : ℤ | -1 ≤ x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2545_254589


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2545_254538

theorem increasing_function_inequality (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_deriv : ∀ x ∈ Set.Ioo 0 1, DifferentiableAt ℝ f x ∧ deriv f x > 0) :
  f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l2545_254538


namespace NUMINAMATH_CALUDE_product_consecutive_integers_square_l2545_254528

theorem product_consecutive_integers_square (x : ℤ) :
  ∃ (y : ℤ), x * (x + 1) * (x + 2) = y^2 ↔ x = 0 ∨ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_consecutive_integers_square_l2545_254528


namespace NUMINAMATH_CALUDE_antelopes_count_l2545_254502

/-- Represents the count of animals on a safari --/
structure SafariCount where
  antelopes : ℕ
  rabbits : ℕ
  hyenas : ℕ
  wild_dogs : ℕ
  leopards : ℕ

/-- Conditions for the safari animal count --/
def safari_conditions (count : SafariCount) : Prop :=
  count.rabbits = count.antelopes + 34 ∧
  count.hyenas = count.antelopes + count.rabbits - 42 ∧
  count.wild_dogs = count.hyenas + 50 ∧
  count.leopards * 2 = count.rabbits ∧
  count.antelopes + count.rabbits + count.hyenas + count.wild_dogs + count.leopards = 605

/-- The theorem stating that the number of antelopes is 80 --/
theorem antelopes_count (count : SafariCount) :
  safari_conditions count → count.antelopes = 80 := by
  sorry

end NUMINAMATH_CALUDE_antelopes_count_l2545_254502


namespace NUMINAMATH_CALUDE_percentage_change_relation_l2545_254599

theorem percentage_change_relation (n c : ℝ) (hn : n > 0) (hc : c > 0) :
  (∀ x : ℝ, x > 0 → x * (1 + n / 100) * (1 - c / 100) = x) →
  n^2 / c^2 = (100 + n) / (100 - c) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_relation_l2545_254599


namespace NUMINAMATH_CALUDE_largest_B_divisible_by_three_l2545_254556

def seven_digit_number (B : ℕ) : ℕ := 4000000 + B * 100000 + 68251

theorem largest_B_divisible_by_three :
  ∀ B : ℕ, B ≤ 9 →
    (seven_digit_number B % 3 = 0) →
    B ≤ 7 ∧
    seven_digit_number 7 % 3 = 0 ∧
    (∀ C : ℕ, C > 7 → C ≤ 9 → seven_digit_number C % 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_B_divisible_by_three_l2545_254556


namespace NUMINAMATH_CALUDE_range_of_cubic_function_l2545_254539

def f (x : ℝ) := x^3

theorem range_of_cubic_function :
  Set.range (fun x => f x) = Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_cubic_function_l2545_254539


namespace NUMINAMATH_CALUDE_ant_colony_problem_l2545_254576

theorem ant_colony_problem (x y : ℕ) :
  x + y = 40 →
  64 * x + 729 * y = 8748 →
  64 * x = 1984 :=
by
  sorry

end NUMINAMATH_CALUDE_ant_colony_problem_l2545_254576


namespace NUMINAMATH_CALUDE_shaded_semicircle_perimeter_l2545_254583

/-- The perimeter of a shaded region in a semicircle -/
theorem shaded_semicircle_perimeter (r : ℝ) (h : r = 2) :
  let arc_length := π * r / 2
  let radii_length := 2 * r
  arc_length + radii_length = π + 4 := by
  sorry


end NUMINAMATH_CALUDE_shaded_semicircle_perimeter_l2545_254583


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l2545_254540

/-- Calculates the time difference at the finish line between two runners in a race -/
theorem race_finish_time_difference 
  (race_distance : ℝ) 
  (alice_speed : ℝ) 
  (bob_speed : ℝ) 
  (h1 : race_distance = 15) 
  (h2 : alice_speed = 7) 
  (h3 : bob_speed = 9) : 
  bob_speed * race_distance - alice_speed * race_distance = 30 := by
  sorry

#check race_finish_time_difference

end NUMINAMATH_CALUDE_race_finish_time_difference_l2545_254540


namespace NUMINAMATH_CALUDE_product_sum_equality_l2545_254535

theorem product_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 31) :
  c + 1 / b = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l2545_254535


namespace NUMINAMATH_CALUDE_function_periodicity_l2545_254557

open Real

theorem function_periodicity 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_a : a > 0) 
  (h_f : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∀ x : ℝ, f (x + 2 * a) = f x := by
sorry

end NUMINAMATH_CALUDE_function_periodicity_l2545_254557


namespace NUMINAMATH_CALUDE_inequality_proof_l2545_254552

theorem inequality_proof (p : ℝ) (x y z v : ℝ) 
  (hp : p ≥ 2) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y)^p + (z + v)^p + (x + z)^p + (y + v)^p ≤ 
  x^p + y^p + z^p + v^p + (x + y + z + v)^p :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2545_254552


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2545_254554

theorem inequality_solution_set (x : ℝ) :
  (1 / x < 2 ∧ 1 / x > -3) ↔ (x > 1 / 2 ∨ x < -1 / 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2545_254554


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2545_254524

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2545_254524


namespace NUMINAMATH_CALUDE_cube_root_three_equation_l2545_254578

theorem cube_root_three_equation (s : ℝ) : s = 1 / (2 - (3 : ℝ)^(1/3)) → s = 2 + (3 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_three_equation_l2545_254578


namespace NUMINAMATH_CALUDE_sugar_percentage_in_kola_solution_l2545_254573

/-- Calculates the percentage of sugar in a kola solution after adding ingredients -/
theorem sugar_percentage_in_kola_solution
  (initial_volume : ℝ)
  (initial_water_percent : ℝ)
  (initial_kola_percent : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percent = 88)
  (h3 : initial_kola_percent = 5)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8) :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_sugar_volume := initial_sugar_percent / 100 * initial_volume
  let final_sugar_volume := initial_sugar_volume + added_sugar
  let final_volume := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_percent := final_sugar_volume / final_volume * 100
  final_sugar_percent = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sugar_percentage_in_kola_solution_l2545_254573


namespace NUMINAMATH_CALUDE_marbles_lost_l2545_254568

theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) 
  (h1 : initial = 19) 
  (h2 : current = 8) 
  (h3 : lost = initial - current) : lost = 11 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l2545_254568


namespace NUMINAMATH_CALUDE_closest_ratio_is_27_26_l2545_254586

/-- The admission fee for adults -/
def adult_fee : ℕ := 30

/-- The admission fee for children -/
def child_fee : ℕ := 15

/-- The total amount collected -/
def total_collected : ℕ := 2400

/-- Represents the number of adults and children at the exhibition -/
structure Attendance where
  adults : ℕ
  children : ℕ
  adults_nonzero : adults > 0
  children_nonzero : children > 0
  total_correct : adult_fee * adults + child_fee * children = total_collected

/-- The ratio of adults to children -/
def attendance_ratio (a : Attendance) : ℚ :=
  a.adults / a.children

/-- Checks if a given ratio is closest to 1 among all possible attendances -/
def is_closest_to_one (r : ℚ) : Prop :=
  ∀ a : Attendance, |attendance_ratio a - 1| ≥ |r - 1|

/-- The main theorem stating that 27/26 is the ratio closest to 1 -/
theorem closest_ratio_is_27_26 :
  is_closest_to_one (27 / 26) :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_is_27_26_l2545_254586


namespace NUMINAMATH_CALUDE_consistent_production_rate_l2545_254593

/-- Represents the rate of paint drum production -/
structure PaintProduction where
  days : ℕ
  drums : ℕ

/-- Calculates the daily production rate -/
def dailyRate (p : PaintProduction) : ℚ :=
  p.drums / p.days

theorem consistent_production_rate : 
  let scenario1 : PaintProduction := ⟨3, 18⟩
  let scenario2 : PaintProduction := ⟨60, 360⟩
  dailyRate scenario1 = dailyRate scenario2 ∧ dailyRate scenario1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_consistent_production_rate_l2545_254593


namespace NUMINAMATH_CALUDE_pyramid_frustum_volume_l2545_254597

/-- Calculate the volume of a pyramid frustum given the dimensions of the original and smaller pyramids --/
theorem pyramid_frustum_volume
  (base_edge_original : ℝ)
  (altitude_original : ℝ)
  (base_edge_smaller : ℝ)
  (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 18)
  (h_altitude_original : altitude_original = 12)
  (h_base_edge_smaller : base_edge_smaller = 12)
  (h_altitude_smaller : altitude_smaller = 8) :
  (1/3 * base_edge_original^2 * altitude_original) - (1/3 * base_edge_smaller^2 * altitude_smaller) = 912 := by
  sorry

#check pyramid_frustum_volume

end NUMINAMATH_CALUDE_pyramid_frustum_volume_l2545_254597


namespace NUMINAMATH_CALUDE_third_to_second_ratio_l2545_254559

/-- The heights of four buildings satisfy certain conditions -/
structure BuildingHeights where
  h1 : ℝ  -- Height of the tallest building
  h2 : ℝ  -- Height of the second tallest building
  h3 : ℝ  -- Height of the third tallest building
  h4 : ℝ  -- Height of the fourth tallest building
  tallest : h1 = 100
  second_tallest : h2 = h1 / 2
  fourth_tallest : h4 = h3 / 5
  total_height : h1 + h2 + h3 + h4 = 180

/-- The ratio of the third tallest to the second tallest building is 1:2 -/
theorem third_to_second_ratio (b : BuildingHeights) : b.h3 / b.h2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_third_to_second_ratio_l2545_254559


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2545_254598

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 7 → 
  c % 2 = 0 →
  c > (b - a) →
  c < (b + a) →
  a + b + c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2545_254598


namespace NUMINAMATH_CALUDE_function_form_l2545_254503

/-- Given a function g: ℝ → ℝ satisfying certain conditions, prove it has a specific form. -/
theorem function_form (g : ℝ → ℝ) 
  (h1 : g 2 = 2)
  (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  ∀ x : ℝ, g x = (5^x - 3^x) / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_form_l2545_254503


namespace NUMINAMATH_CALUDE_first_question_percentage_l2545_254560

/-- The percentage of students who answered the first question correctly -/
def first_question_correct : ℝ := sorry

/-- The percentage of students who answered the second question correctly -/
def second_question_correct : ℝ := 35

/-- The percentage of students who answered neither question correctly -/
def neither_correct : ℝ := 20

/-- The percentage of students who answered both questions correctly -/
def both_correct : ℝ := 30

/-- Theorem stating that the percentage of students who answered the first question correctly is 75% -/
theorem first_question_percentage :
  first_question_correct = 75 :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l2545_254560


namespace NUMINAMATH_CALUDE_permutation_inequality_l2545_254544

theorem permutation_inequality (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : 
  a₁ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₂ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₃ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₄ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₅ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₆ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧
  a₅ ≠ a₆ →
  (a₁ + 1) / 2 * (a₂ + 2) / 2 * (a₃ + 3) / 2 * (a₄ + 4) / 2 * (a₅ + 5) / 2 * (a₆ + 6) / 2 < 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequality_l2545_254544


namespace NUMINAMATH_CALUDE_ln_is_elite_elite_bound_exists_nonincreasing_elite_sufficient_condition_elite_l2545_254507

/-- Definition of an "elite" function -/
def IsElite (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ + x₂) < f x₁ + f x₂

/-- Statement 1: ln(1+x) is an "elite" function -/
theorem ln_is_elite : IsElite (fun x => Real.log (1 + x)) := sorry

/-- Statement 2: For "elite" functions, f(n) < nf(1) for n ≥ 2 -/
theorem elite_bound (f : ℝ → ℝ) (hf : IsElite f) :
  ∀ n : ℕ, n ≥ 2 → f n < n * f 1 := sorry

/-- Statement 3: Existence of an "elite" function that is not strictly increasing -/
theorem exists_nonincreasing_elite :
  ∃ f : ℝ → ℝ, IsElite f ∧ ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) / (x₁ - x₂) ≤ 0 := sorry

/-- Statement 4: A sufficient condition for a function to be "elite" -/
theorem sufficient_condition_elite (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → x₂ * f x₁ < x₁ * f x₂) : 
  IsElite f := sorry

end NUMINAMATH_CALUDE_ln_is_elite_elite_bound_exists_nonincreasing_elite_sufficient_condition_elite_l2545_254507


namespace NUMINAMATH_CALUDE_youtube_ad_time_l2545_254513

/-- Calculates the time spent watching ads on Youtube --/
def time_watching_ads (videos_per_day : ℕ) (video_duration : ℕ) (total_time : ℕ) : ℕ :=
  total_time - (videos_per_day * video_duration)

/-- Theorem: The time spent watching ads is 3 minutes --/
theorem youtube_ad_time :
  time_watching_ads 2 7 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_youtube_ad_time_l2545_254513


namespace NUMINAMATH_CALUDE_symmetry_implies_f_3_equals_1_l2545_254525

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_about_y_equals_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f (x - 1) = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_f_3_equals_1
  (h_sym : symmetric_about_y_equals_x f g)
  (h_g : g 1 = 2) :
  f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_f_3_equals_1_l2545_254525


namespace NUMINAMATH_CALUDE_square_of_sum_l2545_254562

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2545_254562


namespace NUMINAMATH_CALUDE_ratio_10_20_percent_l2545_254575

/-- The percent value of a ratio a:b is defined as (a/b) * 100 -/
def percent_value (a b : ℚ) : ℚ := (a / b) * 100

/-- The ratio 10:20 expressed as a percent is 50% -/
theorem ratio_10_20_percent : percent_value 10 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ratio_10_20_percent_l2545_254575


namespace NUMINAMATH_CALUDE_moon_permutations_eq_twelve_l2545_254558

/-- The number of distinct permutations of the letters in "MOON" -/
def moon_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

theorem moon_permutations_eq_twelve :
  moon_permutations = 12 := by
  sorry

#eval moon_permutations

end NUMINAMATH_CALUDE_moon_permutations_eq_twelve_l2545_254558


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2545_254582

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_s : 0 < s) (pos_t : 0 < t) (pos_u : 0 < u)
  (sum_eq_8 : p + q + r + s + t + u = 8) :
  2/p + 4/q + 9/r + 16/s + 25/t + 36/u ≥ 98 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2545_254582


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l2545_254547

-- Define the displacement function
def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t + 1

-- State the theorem
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 25 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l2545_254547


namespace NUMINAMATH_CALUDE_unfolded_paper_has_four_crosses_l2545_254580

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents a fold on the paper -/
inductive Fold
  | LeftRight
  | TopBottom

/-- Represents a cross pattern of holes -/
structure Cross :=
  (center : ℝ × ℝ)
  (size : ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (paper : Paper)
  (folds : List Fold)
  (cross : Cross)

/-- Represents the unfolded paper with crosses -/
structure UnfoldedPaper :=
  (paper : Paper)
  (crosses : List Cross)

/-- Function to unfold the paper -/
def unfold (fp : FoldedPaper) : UnfoldedPaper :=
  sorry

/-- Main theorem: Unfolding results in four crosses, one in each quadrant -/
theorem unfolded_paper_has_four_crosses (fp : FoldedPaper) 
  (h1 : fp.folds = [Fold.LeftRight, Fold.TopBottom])
  (h2 : fp.cross.center.1 > fp.paper.side / 2 ∧ fp.cross.center.2 > fp.paper.side / 2) :
  let up := unfold fp
  (up.crosses.length = 4) ∧ 
  (∀ q : ℕ, q < 4 → ∃ c ∈ up.crosses, 
    (c.center.1 < up.paper.side / 2 ↔ q % 2 = 0) ∧
    (c.center.2 < up.paper.side / 2 ↔ q < 2)) :=
  sorry

end NUMINAMATH_CALUDE_unfolded_paper_has_four_crosses_l2545_254580


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2545_254516

/-- The lateral area of a cone with base radius 3 and slant height 5 is 15π -/
theorem cone_lateral_area :
  let base_radius : ℝ := 3
  let slant_height : ℝ := 5
  let lateral_area := π * base_radius * slant_height
  lateral_area = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2545_254516


namespace NUMINAMATH_CALUDE_g_difference_l2545_254510

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

-- State the theorem
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2545_254510


namespace NUMINAMATH_CALUDE_negative_comparison_l2545_254530

theorem negative_comparison : -2023 > -2024 := by
  sorry

end NUMINAMATH_CALUDE_negative_comparison_l2545_254530


namespace NUMINAMATH_CALUDE_common_difference_is_negative_three_l2545_254515

def arithmetic_sequence (n : ℕ) : ℤ := 2 - 3 * n

theorem common_difference_is_negative_three :
  ∃ d : ℤ, ∀ n : ℕ, arithmetic_sequence (n + 1) - arithmetic_sequence n = d ∧ d = -3 :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_three_l2545_254515


namespace NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l2545_254588

/-- Proves that Jayson's mom was 28 when he was born, given the conditions -/
theorem jaysons_mom_age_at_birth (jayson_age : ℕ) (dad_age : ℕ) (mom_age : ℕ) 
  (h1 : jayson_age = 10)
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l2545_254588


namespace NUMINAMATH_CALUDE_four_people_three_rooms_l2545_254514

/-- The number of ways to distribute n people into k non-empty rooms -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem four_people_three_rooms :
  distribute 4 3 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_four_people_three_rooms_l2545_254514


namespace NUMINAMATH_CALUDE_second_prime_range_l2545_254546

theorem second_prime_range (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  15 < p * q ∧ p * q ≤ 70 ∧ 2 < p ∧ p < 6 ∧ p * q = 69 → q = 23 := by
  sorry

end NUMINAMATH_CALUDE_second_prime_range_l2545_254546


namespace NUMINAMATH_CALUDE_two_digit_triple_reverse_difference_l2545_254595

theorem two_digit_triple_reverse_difference (A B : ℕ) : 
  A ≠ 0 → 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  2 ∣ ((30 * B + A) - (10 * B + A)) := by
sorry

end NUMINAMATH_CALUDE_two_digit_triple_reverse_difference_l2545_254595


namespace NUMINAMATH_CALUDE_f_pow_ten_l2545_254505

/-- f(n) is the number of ones that occur in the decimal representations of all the numbers from 1 to n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number k, f(10^k) = k * 10^(k-1) + 1 -/
theorem f_pow_ten (k : ℕ) : f (10^k) = k * 10^(k-1) + 1 := by sorry

end NUMINAMATH_CALUDE_f_pow_ten_l2545_254505


namespace NUMINAMATH_CALUDE_jill_minus_jake_equals_one_l2545_254590

def peach_problem (jake steven jill : ℕ) : Prop :=
  (jake + 16 = steven) ∧ 
  (steven = jill + 15) ∧ 
  (jill = 12)

theorem jill_minus_jake_equals_one :
  ∀ jake steven jill : ℕ, peach_problem jake steven jill → jill - jake = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_jill_minus_jake_equals_one_l2545_254590


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2545_254521

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y - 3 = 0) :
  ∀ z, z = 2*x + y → z ≥ 3 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + 2*x₀*y₀ - 3 = 0 ∧ 2*x₀ + y₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2545_254521


namespace NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l2545_254527

/-- The phase shift of a sine function y = A * sin(B * x + C) is -C/B -/
theorem sine_phase_shift (A B C : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ A * Real.sin (B * x + C)
  let phase_shift := -C / B
  ∀ x, f (x + phase_shift) = A * Real.sin (B * x)
  := by sorry

/-- The phase shift of y = 3 * sin(4x + π/4) is -π/16 -/
theorem specific_sine_phase_shift : 
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (4 * x + π/4)
  let phase_shift := -π/16
  ∀ x, f (x + phase_shift) = 3 * Real.sin (4 * x)
  := by sorry

end NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l2545_254527


namespace NUMINAMATH_CALUDE_a_equals_2a_is_valid_assignment_l2545_254563

/-- Definition of a valid assignment statement -/
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String),
    stmt = var ++ " = " ++ expr ∧
    var.length > 0 ∧
    (∀ c, c ∈ var.data → c.isAlpha)

/-- The statement "a = 2*a" is a valid assignment -/
theorem a_equals_2a_is_valid_assignment :
  is_valid_assignment "a = 2*a" := by
  sorry

#check a_equals_2a_is_valid_assignment

end NUMINAMATH_CALUDE_a_equals_2a_is_valid_assignment_l2545_254563


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l2545_254545

theorem quadratic_is_perfect_square (x : ℝ) : 
  ∃ (a : ℝ), x^2 - 20*x + 100 = (x + a)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l2545_254545


namespace NUMINAMATH_CALUDE_five_solutions_for_f_f_eq_seven_l2545_254522

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 6 else x + 5

theorem five_solutions_for_f_f_eq_seven :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 7 :=
sorry

end NUMINAMATH_CALUDE_five_solutions_for_f_f_eq_seven_l2545_254522


namespace NUMINAMATH_CALUDE_joe_remaining_money_l2545_254566

def joe_pocket_money : ℚ := 450

def chocolate_fraction : ℚ := 1/9
def fruit_fraction : ℚ := 2/5

def remaining_money : ℚ := joe_pocket_money - (chocolate_fraction * joe_pocket_money) - (fruit_fraction * joe_pocket_money)

theorem joe_remaining_money :
  remaining_money = 220 :=
by sorry

end NUMINAMATH_CALUDE_joe_remaining_money_l2545_254566


namespace NUMINAMATH_CALUDE_pirate_treasure_ratio_l2545_254534

theorem pirate_treasure_ratio : 
  let total_gold : ℕ := 3500
  let num_chests : ℕ := 5
  let total_silver : ℕ := 500
  let coins_per_chest : ℕ := 1000
  let gold_per_chest : ℕ := total_gold / num_chests
  let silver_per_chest : ℕ := total_silver / num_chests
  let bronze_per_chest : ℕ := coins_per_chest - gold_per_chest - silver_per_chest
  bronze_per_chest = 2 * silver_per_chest :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_ratio_l2545_254534


namespace NUMINAMATH_CALUDE_distance_to_focus_l2545_254577

/-- Given a parabola y^2 = 2x and a point P(m, 2) on the parabola,
    the distance from P to the focus of the parabola is 5/2 -/
theorem distance_to_focus (m : ℝ) (h : 2^2 = 2*m) : 
  let P : ℝ × ℝ := (m, 2)
  let F : ℝ × ℝ := (1/2, 0)
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l2545_254577


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l2545_254574

def f (a x : ℝ) : ℝ := |x + 1| + |x - a|

theorem solution_set_implies_a_value (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, f a x ≥ 5 ↔ x ≤ -2 ∨ x > 3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l2545_254574


namespace NUMINAMATH_CALUDE_square_area_in_circle_l2545_254591

/-- Given a circle with radius 1 and a square with two vertices on the circle
    and one edge passing through the center, prove the area of the square is 4/5 -/
theorem square_area_in_circle (circle_radius : ℝ) (square_side : ℝ) : 
  circle_radius = 1 →
  ∃ (x : ℝ), square_side = 2 * x ∧ 
  x ^ 2 + (2 * x) ^ 2 = circle_radius ^ 2 →
  square_side ^ 2 = 4 / 5 := by
  sorry

#check square_area_in_circle

end NUMINAMATH_CALUDE_square_area_in_circle_l2545_254591


namespace NUMINAMATH_CALUDE_fraction_replacement_l2545_254536

theorem fraction_replacement (x : ℚ) :
  ((5 / 2 / x * 5 / 2) / (5 / 2 * x / (5 / 2))) = 25 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_replacement_l2545_254536


namespace NUMINAMATH_CALUDE_trees_in_column_l2545_254533

/-- Proves the number of trees in one column of Jack's grove --/
theorem trees_in_column (trees_per_row : ℕ) (cleaning_time_per_tree : ℕ) (total_cleaning_time : ℕ) 
  (h1 : trees_per_row = 4)
  (h2 : cleaning_time_per_tree = 3)
  (h3 : total_cleaning_time = 60)
  (h4 : total_cleaning_time / cleaning_time_per_tree = trees_per_row * (total_cleaning_time / cleaning_time_per_tree / trees_per_row)) :
  total_cleaning_time / cleaning_time_per_tree / trees_per_row = 5 := by
  sorry

#check trees_in_column

end NUMINAMATH_CALUDE_trees_in_column_l2545_254533


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2545_254584

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Points where a perpendicular from the right focus intersects the hyperbola -/
def intersection_points (h : Hyperbola a b) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The inscribed circle of a triangle -/
def inscribed_circle (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The eccentricity of the hyperbola is (1 + √5) / 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let (A, B) := intersection_points h
  let F₁ := left_focus h
  let (_, _, r) := inscribed_circle A B F₁
  r = a →
  eccentricity h = (1 + Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2545_254584


namespace NUMINAMATH_CALUDE_cave_depth_calculation_l2545_254570

theorem cave_depth_calculation (total_depth remaining_distance : ℕ) 
  (h1 : total_depth = 974)
  (h2 : remaining_distance = 386) :
  total_depth - remaining_distance = 588 := by
sorry

end NUMINAMATH_CALUDE_cave_depth_calculation_l2545_254570


namespace NUMINAMATH_CALUDE_min_value_of_function_l2545_254509

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let f := fun x => 4 * x + 2 / x
  (∀ y > 0, f y ≥ 4 * Real.sqrt 2) ∧ (∃ y > 0, f y = 4 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2545_254509


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l2545_254543

/-- Given two lines with the same non-zero y-intercept, where the first line has slope 12
    and x-intercept (u, 0), and the second line has slope 8 and x-intercept (v, 0),
    prove that the ratio of u to v is 2/3. -/
theorem ratio_of_x_intercepts (b : ℝ) (u v : ℝ) (h1 : b ≠ 0)
    (h2 : 12 * u + b = 0) (h3 : 8 * v + b = 0) : u / v = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l2545_254543


namespace NUMINAMATH_CALUDE_mrs_copper_class_size_l2545_254579

theorem mrs_copper_class_size :
  ∀ (initial_jellybeans : ℕ) 
    (absent_children : ℕ) 
    (jellybeans_per_child : ℕ) 
    (remaining_jellybeans : ℕ),
  initial_jellybeans = 100 →
  absent_children = 2 →
  jellybeans_per_child = 3 →
  remaining_jellybeans = 34 →
  ∃ (total_children : ℕ),
    total_children = 
      (initial_jellybeans - remaining_jellybeans) / jellybeans_per_child + absent_children ∧
    total_children = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_copper_class_size_l2545_254579


namespace NUMINAMATH_CALUDE_angle_A_value_side_a_range_l2545_254504

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

theorem angle_A_value (t : AcuteTriangle) :
  Real.cos (2 * t.A) - Real.cos (2 * t.B) + 2 * Real.cos (π/6 - t.B) * Real.cos (π/6 + t.B) = 0 →
  t.A = π/3 := by sorry

theorem side_a_range (t : AcuteTriangle) :
  t.b = Real.sqrt 3 → t.b ≤ t.a → t.A = π/3 →
  t.a ≥ Real.sqrt 3 ∧ t.a < 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_side_a_range_l2545_254504


namespace NUMINAMATH_CALUDE_prime_factors_of_x_l2545_254517

theorem prime_factors_of_x (x : ℕ) 
  (h1 : x % 44 = 0 ∧ x / 44 = 432)
  (h2 : x % 31 = 5)
  (h3 : ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ x = a^3 * b^2 * c) :
  ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 7 ∧ Prime a ∧ Prime b ∧ Prime c ∧ x = a^3 * b^2 * c :=
by sorry

end NUMINAMATH_CALUDE_prime_factors_of_x_l2545_254517


namespace NUMINAMATH_CALUDE_prob_more_twos_than_fives_correct_l2545_254529

def num_dice : ℕ := 5
def num_sides : ℕ := 6

def prob_more_twos_than_fives : ℚ := 2721 / 7776

theorem prob_more_twos_than_fives_correct :
  let total_outcomes := num_sides ^ num_dice
  let equal_twos_and_fives := 2334
  (1 / 2) * (1 - equal_twos_and_fives / total_outcomes) = prob_more_twos_than_fives :=
by sorry

end NUMINAMATH_CALUDE_prob_more_twos_than_fives_correct_l2545_254529


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2545_254531

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 17) 
  (eq2 : x + 4 * y = 23) : 
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2545_254531


namespace NUMINAMATH_CALUDE_part_one_part_two_l2545_254501

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem part_one :
  (∀ x : ℝ, y m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2
theorem part_two :
  (∀ x ∈ Set.Icc 1 3, y m x < -m + 5) ↔ m ∈ Set.Iio (6/7) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2545_254501


namespace NUMINAMATH_CALUDE_system_solution_l2545_254511

def solution_set : Set (ℝ × ℝ) :=
  {(-3/Real.sqrt 5, 1/Real.sqrt 5), (-3/Real.sqrt 5, -1/Real.sqrt 5),
   (3/Real.sqrt 5, -1/Real.sqrt 5), (3/Real.sqrt 5, 1/Real.sqrt 5)}

theorem system_solution :
  ∀ x y : ℝ, (x^2 + y^2 ≤ 2 ∧
    81*x^4 - 18*x^2*y^2 + y^4 - 360*x^2 - 40*y^2 + 400 = 0) ↔
  (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2545_254511


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2545_254567

/-- Two arithmetic sequences and their sum sequences -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum sequence for a
  T : ℕ → ℚ  -- Sum sequence for b

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequencePair)
  (h_sum_ratio : ∀ n : ℕ, seq.S n / seq.T n = 2 * n / (3 * n + 1)) :
  seq.a 10 / seq.b 10 = 19 / 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2545_254567


namespace NUMINAMATH_CALUDE_T_100_value_l2545_254526

/-- The original sequence a_n -/
def a (n : ℕ) : ℕ := 2^(n-1)

/-- The number of inserted terms between a_k and a_{k+1} -/
def inserted_count (k : ℕ) : ℕ := k

/-- The value of inserted terms between a_k and a_{k+1} -/
def inserted_value (k : ℕ) : ℤ := (-1)^k * k

/-- The sum of the first n terms of the new sequence b_n -/
noncomputable def T (n : ℕ) : ℤ := sorry

/-- The theorem to prove -/
theorem T_100_value : T 100 = 8152 := by sorry

end NUMINAMATH_CALUDE_T_100_value_l2545_254526


namespace NUMINAMATH_CALUDE_smallest_natural_with_remainder_one_l2545_254569

theorem smallest_natural_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 3 = 1 ∧ 
  n % 5 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 3 = 1 ∧ m % 5 = 1 → n ≤ m) ∧
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_natural_with_remainder_one_l2545_254569


namespace NUMINAMATH_CALUDE_circle_c_equation_l2545_254550

/-- A circle C with center on y = x^2, passing through origin, and intercepting 8 units on y-axis -/
structure CircleC where
  a : ℝ
  center : ℝ × ℝ
  center_on_parabola : center.2 = center.1^2
  passes_through_origin : (0 - center.1)^2 + (0 - center.2)^2 = (4 + center.1)^2
  intercepts_8_on_yaxis : (0 - center.1)^2 + (4 - center.2)^2 = (4 + center.1)^2

/-- The equation of circle C is either (x-2)^2 + (y-4)^2 = 20 or (x+2)^2 + (y-4)^2 = 20 -/
theorem circle_c_equation (c : CircleC) :
  ((λ (x y : ℝ) => (x - 2)^2 + (y - 4)^2 = 20) = λ (x y : ℝ) => (x - c.center.1)^2 + (y - c.center.2)^2 = (4 + c.a)^2) ∨
  ((λ (x y : ℝ) => (x + 2)^2 + (y - 4)^2 = 20) = λ (x y : ℝ) => (x - c.center.1)^2 + (y - c.center.2)^2 = (4 + c.a)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_c_equation_l2545_254550


namespace NUMINAMATH_CALUDE_power_sum_of_i_l2545_254532

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^66 + i^103 = -1 - i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l2545_254532


namespace NUMINAMATH_CALUDE_max_n_for_factorization_l2545_254518

theorem max_n_for_factorization : 
  (∃ (n : ℤ), ∀ (x : ℝ), ∃ (A B : ℤ), 
    6 * x^2 + n * x + 144 = (6 * x + A) * (x + B)) ∧
  (∀ (m : ℤ), m > 865 → 
    ¬∃ (A B : ℤ), ∀ (x : ℝ), 6 * x^2 + m * x + 144 = (6 * x + A) * (x + B)) :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_factorization_l2545_254518


namespace NUMINAMATH_CALUDE_system_solution_difference_l2545_254585

theorem system_solution_difference (x y : ℝ) : 
  1012 * x + 1016 * y = 1020 →
  1014 * x + 1018 * y = 1022 →
  x - y = 1.09 := by
sorry

end NUMINAMATH_CALUDE_system_solution_difference_l2545_254585


namespace NUMINAMATH_CALUDE_school_population_relation_l2545_254592

theorem school_population_relation 
  (X : ℝ) -- Total number of students
  (p : ℝ) -- Percentage of boys that 90 students represent
  (h1 : X > 0) -- Assumption that the school has a positive number of students
  (h2 : 0 < p ∧ p < 100) -- Assumption that p is a valid percentage
  : 90 = p / 100 * 0.5 * X := by
  sorry

end NUMINAMATH_CALUDE_school_population_relation_l2545_254592


namespace NUMINAMATH_CALUDE_girls_together_arrangements_girls_separate_arrangements_l2545_254553

/-- The number of boys in the lineup -/
def num_boys : ℕ := 4

/-- The number of girls in the lineup -/
def num_girls : ℕ := 3

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of ways to arrange the lineup with girls together -/
def arrangements_girls_together : ℕ := 720

/-- The number of ways to arrange the lineup with no two girls together -/
def arrangements_girls_separate : ℕ := 1440

/-- Theorem stating the number of arrangements with girls together -/
theorem girls_together_arrangements :
  (num_girls.factorial * (num_boys + 1).factorial) = arrangements_girls_together := by sorry

/-- Theorem stating the number of arrangements with no two girls together -/
theorem girls_separate_arrangements :
  (num_boys.factorial * (Nat.choose (num_boys + 1) num_girls) * num_girls.factorial) = arrangements_girls_separate := by sorry

end NUMINAMATH_CALUDE_girls_together_arrangements_girls_separate_arrangements_l2545_254553


namespace NUMINAMATH_CALUDE_block_with_t_hole_difference_l2545_254571

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the dimensions and position of a T-shaped hole -/
structure THole where
  height : ℕ
  length : ℕ
  width : ℕ
  distanceFromFront : ℕ

/-- Calculates the number of cubes needed to create a block with a T-shaped hole -/
def cubesNeededWithHole (block : BlockDimensions) (hole : THole) : ℕ :=
  block.length * block.width * block.depth - (hole.height * hole.length + hole.width - 1)

/-- Theorem stating that a 7x7x6 block with the given T-shaped hole requires 3 fewer cubes -/
theorem block_with_t_hole_difference :
  let block := BlockDimensions.mk 7 7 6
  let hole := THole.mk 1 3 2 3
  block.length * block.width * block.depth - cubesNeededWithHole block hole = 3 := by
  sorry


end NUMINAMATH_CALUDE_block_with_t_hole_difference_l2545_254571


namespace NUMINAMATH_CALUDE_cos_a2_a12_equals_half_l2545_254541

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem cos_a2_a12_equals_half
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_a2_a12_equals_half_l2545_254541


namespace NUMINAMATH_CALUDE_sin_tan_greater_than_square_l2545_254555

theorem sin_tan_greater_than_square (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) : 
  Real.sin x * Real.tan x > x^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_greater_than_square_l2545_254555


namespace NUMINAMATH_CALUDE_quadratic_curve_coefficient_l2545_254548

theorem quadratic_curve_coefficient (p q y1 y2 : ℝ) : 
  (y1 = p + q + 5) →
  (y2 = p - q + 5) →
  (y1 + y2 = 14) →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_curve_coefficient_l2545_254548


namespace NUMINAMATH_CALUDE_john_soap_cost_l2545_254551

/-- The amount of money John spent on soap -/
def soap_cost (num_bars : ℕ) (weight_per_bar : ℚ) (price_per_pound : ℚ) : ℚ :=
  num_bars * weight_per_bar * price_per_pound

/-- Proof that John spent $15 on soap -/
theorem john_soap_cost :
  soap_cost 20 (3/2) (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_soap_cost_l2545_254551


namespace NUMINAMATH_CALUDE_triangle_height_theorem_l2545_254587

-- Define the triangle's properties
def triangle_area : ℝ := 48  -- in square decimeters
def triangle_base : ℝ := 6   -- in meters

-- Convert base from meters to decimeters
def triangle_base_dm : ℝ := triangle_base * 10

-- Define the theorem
theorem triangle_height_theorem :
  ∃ (height : ℝ), 
    (triangle_base_dm * height / 2 = triangle_area) ∧ 
    (height = 1.6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_theorem_l2545_254587


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2545_254572

theorem triangle_angle_measure (D E F : ℝ) : 
  D + E + F = 180 →  -- Sum of angles in a triangle is 180°
  E = F →            -- Angle E is congruent to Angle F
  F = 3 * D →        -- Angle F is three times Angle D
  E = 540 / 7 :=     -- Measure of Angle E
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2545_254572


namespace NUMINAMATH_CALUDE_torn_sheets_count_l2545_254561

/-- Represents a book with consecutively numbered pages. -/
structure Book where
  first_torn_page : Nat
  last_torn_page : Nat

/-- Checks if two numbers have the same digits. -/
def same_digits (a b : Nat) : Prop :=
  sorry

/-- Calculates the number of torn sheets given a Book. -/
def torn_sheets (book : Book) : Nat :=
  (book.last_torn_page - book.first_torn_page + 1) / 2

/-- The main theorem stating the number of torn sheets. -/
theorem torn_sheets_count (book : Book) :
    book.first_torn_page = 185
  → same_digits book.first_torn_page book.last_torn_page
  → Even book.last_torn_page
  → book.last_torn_page > book.first_torn_page
  → torn_sheets book = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l2545_254561


namespace NUMINAMATH_CALUDE_jesse_room_area_l2545_254512

/-- The area of a rectangular room -/
def room_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of Jesse's room is 96 square feet -/
theorem jesse_room_area :
  room_area 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l2545_254512


namespace NUMINAMATH_CALUDE_final_number_calculation_l2545_254500

theorem final_number_calculation : ∃ (n : ℕ), n = 5 ∧ (3 * ((2 * n) + 9) = 57) := by
  sorry

end NUMINAMATH_CALUDE_final_number_calculation_l2545_254500


namespace NUMINAMATH_CALUDE_property_rent_calculation_l2545_254564

theorem property_rent_calculation (purchase_price : ℝ) (maintenance_rate : ℝ) 
  (annual_tax : ℝ) (target_return_rate : ℝ) (monthly_rent : ℝ) : 
  purchase_price = 12000 ∧ 
  maintenance_rate = 0.15 ∧ 
  annual_tax = 400 ∧ 
  target_return_rate = 0.06 ∧ 
  monthly_rent = 109.80 →
  monthly_rent * 12 * (1 - maintenance_rate) = 
    purchase_price * target_return_rate + annual_tax :=
by
  sorry

#check property_rent_calculation

end NUMINAMATH_CALUDE_property_rent_calculation_l2545_254564


namespace NUMINAMATH_CALUDE_distribution_problem_l2545_254581

theorem distribution_problem (total_amount : ℕ) (first_group : ℕ) (difference : ℕ) (second_group : ℕ) :
  total_amount = 5040 →
  first_group = 14 →
  difference = 80 →
  (total_amount / first_group) = (total_amount / second_group + difference) →
  second_group = 18 := by
sorry

end NUMINAMATH_CALUDE_distribution_problem_l2545_254581
