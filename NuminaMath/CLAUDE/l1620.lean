import Mathlib

namespace largest_positive_integer_for_binary_operation_l1620_162087

theorem largest_positive_integer_for_binary_operation
  (x : ℝ) (h : x > -8) :
  (∀ n : ℕ+, n - 5 * n < x) ∧
  (∀ m : ℕ+, m > 2 → ¬(m - 5 * m < x)) :=
sorry

end largest_positive_integer_for_binary_operation_l1620_162087


namespace degree_of_minus_five_x_four_y_l1620_162046

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (coeff : ℤ) (x_exp y_exp : ℕ) : ℕ :=
  x_exp + y_exp

/-- The monomial -5x^4y has degree 5 -/
theorem degree_of_minus_five_x_four_y :
  degree_of_monomial (-5) 4 1 = 5 := by
  sorry

end degree_of_minus_five_x_four_y_l1620_162046


namespace min_value_theorem_l1620_162008

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 14 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 14 := by
sorry

end min_value_theorem_l1620_162008


namespace bird_nest_difference_l1620_162031

def number_of_birds : ℕ := 6
def number_of_nests : ℕ := 3

theorem bird_nest_difference :
  number_of_birds - number_of_nests = 3 := by
  sorry

end bird_nest_difference_l1620_162031


namespace original_people_count_l1620_162050

theorem original_people_count (initial_count : ℕ) : 
  (initial_count / 3 : ℚ) = 18 →
  initial_count = 54 := by
  sorry

end original_people_count_l1620_162050


namespace ellipse_constants_correct_l1620_162076

def ellipse_constants (f₁ f₂ p : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem ellipse_constants_correct :
  let f₁ : ℝ × ℝ := (3, 3)
  let f₂ : ℝ × ℝ := (3, 9)
  let p : ℝ × ℝ := (16, -2)
  let (a, b, h, k) := ellipse_constants f₁ f₂ p
  (h = 3 ∧
   k = 6 ∧
   a = (Real.sqrt 194 + Real.sqrt 290) / 2 ∧
   b = Real.sqrt ((Real.sqrt 194 + Real.sqrt 290)^2 / 4 - 9)) ∧
  (a > 0 ∧ b > 0) ∧
  ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 2 * a := by
  sorry

end ellipse_constants_correct_l1620_162076


namespace largest_four_digit_negative_congruent_to_one_mod_23_l1620_162073

theorem largest_four_digit_negative_congruent_to_one_mod_23 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -1011 :=
by sorry

end largest_four_digit_negative_congruent_to_one_mod_23_l1620_162073


namespace fair_apple_distribution_l1620_162002

/-- Represents the work done by each girl -/
structure WorkDistribution :=
  (anna : ℕ)
  (varya : ℕ)
  (sveta : ℕ)

/-- Represents the fair distribution of apples -/
structure AppleDistribution :=
  (anna : ℚ)
  (varya : ℚ)
  (sveta : ℚ)

/-- Calculates the fair distribution of apples based on work done -/
def calculateAppleDistribution (work : WorkDistribution) (totalApples : ℕ) : AppleDistribution :=
  let totalWork := work.anna + work.varya + work.sveta
  { anna := (work.anna : ℚ) / totalWork * totalApples,
    varya := (work.varya : ℚ) / totalWork * totalApples,
    sveta := (work.sveta : ℚ) / totalWork * totalApples }

/-- The main theorem to prove -/
theorem fair_apple_distribution 
  (work : WorkDistribution) 
  (h_work : work = ⟨20, 35, 45⟩) 
  (totalApples : ℕ) 
  (h_apples : totalApples = 10) :
  calculateAppleDistribution work totalApples = ⟨2, (7:ℚ)/2, (9:ℚ)/2⟩ := by
  sorry

#check fair_apple_distribution

end fair_apple_distribution_l1620_162002


namespace new_person_weight_l1620_162068

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 35 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 55 :=
by sorry

end new_person_weight_l1620_162068


namespace symmetric_normal_distribution_l1620_162038

/-- Represents a normally distributed population with a given mean -/
structure NormalPopulation where
  mean : ℝ
  size : ℕ
  above_threshold : ℕ
  threshold : ℝ

/-- 
Given a normally distributed population with mean 75,
if 960 out of 1200 individuals score at least 60,
then 240 individuals score above 90.
-/
theorem symmetric_normal_distribution 
  (pop : NormalPopulation)
  (h_mean : pop.mean = 75)
  (h_size : pop.size = 1200)
  (h_threshold : pop.threshold = 60)
  (h_above_threshold : pop.above_threshold = 960) :
  pop.size - pop.above_threshold = 240 :=
sorry

end symmetric_normal_distribution_l1620_162038


namespace tower_of_hanoi_l1620_162016

/-- Minimal number of moves required to solve the Tower of Hanoi problem with n discs -/
def hanoi_moves (n : ℕ) : ℕ :=
  2^n - 1

/-- The Tower of Hanoi theorem -/
theorem tower_of_hanoi (n : ℕ) : 
  hanoi_moves n = 2^n - 1 := by
  sorry

#eval hanoi_moves 64

end tower_of_hanoi_l1620_162016


namespace initial_speed_correct_l1620_162082

/-- The initial walking speed that satisfies the given conditions. -/
def initial_speed (distance : ℝ) (miss_time : ℝ) (early_time : ℝ) (faster_speed : ℝ) : ℝ :=
  3

/-- Theorem stating that the initial_speed function returns the correct speed. -/
theorem initial_speed_correct
  (distance : ℝ)
  (miss_time : ℝ)
  (early_time : ℝ)
  (faster_speed : ℝ)
  (h1 : distance = 2.2)
  (h2 : miss_time = 12 / 60)
  (h3 : early_time = 10 / 60)
  (h4 : faster_speed = 6) :
  let v := initial_speed distance miss_time early_time faster_speed
  distance / v - miss_time = distance / faster_speed + early_time :=
by sorry

end initial_speed_correct_l1620_162082


namespace triangle_similarity_problem_l1620_162017

theorem triangle_similarity_problem (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 10) 
  (h3 : AD > 0) (h4 : (1 : ℝ) / 3 * AD = (DC + CB) * 2 / 3) (h5 : (3 : ℝ) / 4 * AD = 21.375) : 
  ∃ FC : ℝ, FC = 14.625 := by
  sorry

end triangle_similarity_problem_l1620_162017


namespace car_travel_distance_l1620_162041

/-- Proves that a car traveling at a constant rate of 3 kilometers every 4 minutes
    will cover 90 kilometers in 2 hours. -/
theorem car_travel_distance (rate : ℝ) (time : ℝ) : 
  rate = 3 / 4 → time = 120 → rate * time = 90 := by
  sorry

end car_travel_distance_l1620_162041


namespace quadratic_inequality_range_l1620_162043

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) → 0 ≤ a ∧ a ≤ 4 := by
  sorry

end quadratic_inequality_range_l1620_162043


namespace age_difference_l1620_162033

/-- Given that the total age of A and B is 20 years more than the total age of B and C,
    prove that C is 20 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 20) : A = C + 20 := by
  sorry

end age_difference_l1620_162033


namespace price_reduction_proof_l1620_162096

theorem price_reduction_proof (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.4
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_reduction := original_price - price_after_second
  let reduction_percentage := total_reduction / original_price * 100
  reduction_percentage = 58 := by
sorry

end price_reduction_proof_l1620_162096


namespace candy_given_to_haley_l1620_162000

def initial_candy : ℕ := 15
def remaining_candy : ℕ := 9

theorem candy_given_to_haley : initial_candy - remaining_candy = 6 := by
  sorry

end candy_given_to_haley_l1620_162000


namespace abduls_numbers_l1620_162091

theorem abduls_numbers (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (eq1 : a + (b + c + d) / 3 = 17)
  (eq2 : b + (a + c + d) / 3 = 21)
  (eq3 : c + (a + b + d) / 3 = 23)
  (eq4 : d + (a + b + c) / 3 = 29) :
  max a (max b (max c d)) = 21 := by
  sorry

end abduls_numbers_l1620_162091


namespace inequality_solution_l1620_162003

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l1620_162003


namespace angle_quadrant_for_defined_log_l1620_162052

theorem angle_quadrant_for_defined_log (θ : Real) :
  (∃ x, x = Real.log (Real.cos θ * Real.tan θ)) →
  (0 ≤ θ ∧ θ < Real.pi) :=
sorry

end angle_quadrant_for_defined_log_l1620_162052


namespace fraction_modification_l1620_162048

theorem fraction_modification (a b c d x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a^2 + x) / (b^2 + x) = c / d) 
  (h4 : c ≠ d) : 
  x = (a^2 * d - b^2 * c) / (c - d) := by
sorry

end fraction_modification_l1620_162048


namespace all_integers_are_integers_l1620_162023

theorem all_integers_are_integers (n : ℕ) (a : Fin n → ℕ+) 
  (h : ∀ i j : Fin n, i ≠ j → 
    (((a i).val + (a j).val) / 2 : ℚ).den = 1 ∨ 
    (((a i).val * (a j).val : ℕ).sqrt : ℚ).den = 1) : 
  ∀ i : Fin n, (a i).val = (a i : ℕ) := by
sorry

end all_integers_are_integers_l1620_162023


namespace max_negative_integers_l1620_162056

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    (∃ (neg_set : Finset ℤ),
      neg_set ⊆ {a, b, c, d, e, f} ∧
      neg_set.card = neg_count ∧
      ∀ x ∈ neg_set, x < 0) ∧
    ∀ (other_neg_set : Finset ℤ),
      other_neg_set ⊆ {a, b, c, d, e, f} →
      (∀ x ∈ other_neg_set, x < 0) →
      other_neg_set.card ≤ neg_count :=
by sorry

end max_negative_integers_l1620_162056


namespace x_eq_2_sufficient_not_necessary_l1620_162057

-- Define the vectors a and b as functions of x
def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 1, 3]

-- Define what it means for two vectors to be parallel
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

-- State the theorem
theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, parallel (a x) (b x) → x = 2) :=
by sorry

end x_eq_2_sufficient_not_necessary_l1620_162057


namespace equal_intercept_line_correct_l1620_162029

/-- A line passing through point (2, 3) with equal intercepts on both axes -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y - 5 = 0

theorem equal_intercept_line_correct :
  -- The line passes through (2, 3)
  equal_intercept_line 2 3 ∧
  -- The line has equal intercepts on both axes
  ∃ a : ℝ, a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a :=
by
  sorry

end equal_intercept_line_correct_l1620_162029


namespace complex_number_coordinates_l1620_162072

theorem complex_number_coordinates (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 1 - Complex.I
  (∃ b : ℝ, z₁ / z₂ = Complex.I * b) → z₁ = 1 + Complex.I :=
by sorry

end complex_number_coordinates_l1620_162072


namespace floor_equation_solution_l1620_162040

theorem floor_equation_solution (x : ℝ) :
  ⌊x * (⌊x⌋ - 1)⌋ = 8 ↔ 4 ≤ x ∧ x < 4.5 :=
by sorry

end floor_equation_solution_l1620_162040


namespace parallel_line_correct_perpendicular_line_correct_l1620_162022

-- Define the original line
def original_line (x y : ℝ) : Prop := y = 3 * x - 4

-- Define the point P₀
def P₀ : ℝ × ℝ := (1, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := y = 3 * x - 1

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := y = -1/3 * x + 7/3

-- Theorem for the parallel line
theorem parallel_line_correct :
  (parallel_line P₀.1 P₀.2) ∧
  (∀ x y z : ℝ, original_line x y → original_line (x + 1) z → z - y = 3) :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line_correct :
  (perpendicular_line P₀.1 P₀.2) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, original_line x₁ y₁ → perpendicular_line x₂ y₂ →
    (y₂ - y₁) = -(1/3) * (x₂ - x₁)) :=
sorry

end parallel_line_correct_perpendicular_line_correct_l1620_162022


namespace complex_magnitude_l1620_162071

theorem complex_magnitude (z : ℂ) (h : z + (Real.exp 1) / z + Real.pi = 0) :
  Complex.abs z = Real.sqrt (Real.exp 1) := by
  sorry

end complex_magnitude_l1620_162071


namespace sum_of_primes_divisible_by_60_l1620_162015

theorem sum_of_primes_divisible_by_60 (p q r s : ℕ) : 
  Prime p → Prime q → Prime r → Prime s →
  5 < p → p < q → q < r → r < s → s < p + 10 →
  60 ∣ (p + q + r + s) := by
sorry

end sum_of_primes_divisible_by_60_l1620_162015


namespace water_mixture_percentage_l1620_162092

theorem water_mixture_percentage 
  (initial_mixture : ℝ) 
  (water_added : ℝ) 
  (final_percentage : ℝ) :
  initial_mixture = 20 →
  water_added = 4 →
  final_percentage = 25 →
  (initial_mixture * (initial_percentage / 100) + water_added) / (initial_mixture + water_added) * 100 = final_percentage →
  initial_percentage = 10 :=
by
  sorry

end water_mixture_percentage_l1620_162092


namespace carnival_ride_wait_time_l1620_162063

/-- Proves that the wait time for the giant slide is 15 minutes given the carnival ride conditions. -/
theorem carnival_ride_wait_time 
  (total_time : ℕ) 
  (roller_coaster_wait : ℕ) 
  (tilt_a_whirl_wait : ℕ) 
  (roller_coaster_rides : ℕ) 
  (tilt_a_whirl_rides : ℕ) 
  (giant_slide_rides : ℕ) 
  (h1 : total_time = 4 * 60)  -- 4 hours in minutes
  (h2 : roller_coaster_wait = 30)
  (h3 : tilt_a_whirl_wait = 60)
  (h4 : roller_coaster_rides = 4)
  (h5 : tilt_a_whirl_rides = 1)
  (h6 : giant_slide_rides = 4)
  : ∃ (giant_slide_wait : ℕ), 
    giant_slide_wait * giant_slide_rides = 
      total_time - (roller_coaster_wait * roller_coaster_rides + tilt_a_whirl_wait * tilt_a_whirl_rides) ∧
    giant_slide_wait = 15 :=
by sorry

end carnival_ride_wait_time_l1620_162063


namespace volume_between_concentric_spheres_l1620_162021

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4  -- radius of smaller sphere
  let r₂ : ℝ := 9  -- radius of larger sphere
  let v₁ := (4/3) * π * r₁^3  -- volume of smaller sphere
  let v₂ := (4/3) * π * r₂^3  -- volume of larger sphere
  v₂ - v₁ = (2656/3) * π :=
by sorry

end volume_between_concentric_spheres_l1620_162021


namespace sum_first_15_odd_integers_l1620_162018

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2*n + 1) = 225 := by
  sorry

end sum_first_15_odd_integers_l1620_162018


namespace unique_number_doubled_plus_thirteen_l1620_162007

theorem unique_number_doubled_plus_thirteen : ∃! x : ℝ, 2 * x + 13 = 89 := by
  sorry

end unique_number_doubled_plus_thirteen_l1620_162007


namespace twelfth_root_of_unity_l1620_162062

theorem twelfth_root_of_unity : 
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ 
  (Complex.tan (π / 6) + Complex.I) / (Complex.tan (π / 6) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 12)) := by
  sorry

end twelfth_root_of_unity_l1620_162062


namespace sum_of_squares_representation_specific_sum_of_squares_2009_l1620_162065

theorem sum_of_squares_representation (n : ℕ) :
  ∃ (a b c d : ℕ), 2 * n^2 + 2 * (n + 1)^2 = a^2 + b^2 ∧ 2 * n^2 + 2 * (n + 1)^2 = c^2 + d^2 ∧ (a ≠ c ∨ b ≠ d) := by
  sorry

-- Specific case for n = 2009
theorem specific_sum_of_squares_2009 :
  ∃ (a b c d : ℕ), 2 * 2009^2 + 2 * 2010^2 = a^2 + b^2 ∧ 2 * 2009^2 + 2 * 2010^2 = c^2 + d^2 ∧ (a ≠ c ∨ b ≠ d) := by
  sorry

end sum_of_squares_representation_specific_sum_of_squares_2009_l1620_162065


namespace odd_times_even_is_odd_l1620_162034

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem odd_times_even_is_odd
  (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry

#check odd_times_even_is_odd

end odd_times_even_is_odd_l1620_162034


namespace g_of_2_eq_5_l1620_162026

def g (x : ℝ) : ℝ := x^3 - x^2 + 1

theorem g_of_2_eq_5 : g 2 = 5 := by sorry

end g_of_2_eq_5_l1620_162026


namespace parabola_vertex_l1620_162070

/-- The vertex of the parabola y = 1/2 * (x + 1)^2 - 1/2 is (-1, -1/2) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := λ x ↦ (1/2 : ℝ) * (x + 1)^2 - 1/2
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1/2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end parabola_vertex_l1620_162070


namespace sin_sum_to_product_l1620_162075

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end sin_sum_to_product_l1620_162075


namespace letter_distribution_l1620_162061

/-- Represents the number of letters in each pocket -/
def pocket1_letters : ℕ := 5
def pocket2_letters : ℕ := 4

/-- Represents the number of mailboxes -/
def num_mailboxes : ℕ := 4

/-- The number of ways to take one letter from two pockets -/
def ways_to_take_one_letter : ℕ := pocket1_letters + pocket2_letters

/-- The number of ways to take one letter from each pocket -/
def ways_to_take_one_from_each : ℕ := pocket1_letters * pocket2_letters

/-- The number of ways to put all letters into mailboxes -/
def ways_to_put_in_mailboxes : ℕ := num_mailboxes ^ (pocket1_letters + pocket2_letters)

theorem letter_distribution :
  (ways_to_take_one_letter = 9) ∧
  (ways_to_take_one_from_each = 20) ∧
  (ways_to_put_in_mailboxes = 262144) := by
  sorry

end letter_distribution_l1620_162061


namespace area_triangle_BCD_l1620_162045

/-- Given a triangle ABC and a point D on the line AC extended, 
    prove that the area of triangle BCD can be calculated. -/
theorem area_triangle_BCD 
  (area_ABC : ℝ) 
  (length_AC : ℝ) 
  (length_CD : ℝ) 
  (h_area_ABC : area_ABC = 36)
  (h_length_AC : length_AC = 9)
  (h_length_CD : length_CD = 33) :
  ∃ (area_BCD : ℝ), area_BCD = 132 := by
sorry

end area_triangle_BCD_l1620_162045


namespace square_root_of_one_sixty_fourth_l1620_162035

theorem square_root_of_one_sixty_fourth : Real.sqrt (1 / 64) = 1 / 8 := by
  sorry

end square_root_of_one_sixty_fourth_l1620_162035


namespace vector_equality_implies_m_value_l1620_162014

theorem vector_equality_implies_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![m, 2]
  let b : Fin 2 → ℝ := ![2, -3]
  (‖a + b‖ = ‖a - b‖) → m = 3 := by
  sorry

end vector_equality_implies_m_value_l1620_162014


namespace square_area_ratio_l1620_162089

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := 2 * s₂ * Real.sqrt 2
  (s₁ ^ 2) / (s₂ ^ 2) = 8 := by
sorry

end square_area_ratio_l1620_162089


namespace least_three_digit_multiple_of_eight_l1620_162098

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, n ≥ 100 ∧ n < 104 → n % 8 ≠ 0) ∧ 
  104 % 8 = 0 ∧ 
  104 ≥ 100 ∧ 
  104 < 1000 :=
sorry

end least_three_digit_multiple_of_eight_l1620_162098


namespace vanessa_video_files_vanessa_video_files_proof_l1620_162019

theorem vanessa_video_files 
  (initial_music_files : ℕ) 
  (deleted_files : ℕ) 
  (remaining_files : ℕ) : ℕ :=
  let initial_total_files := remaining_files + deleted_files
  let initial_video_files := initial_total_files - initial_music_files
  initial_video_files

-- Proof
theorem vanessa_video_files_proof 
  (initial_music_files : ℕ) 
  (deleted_files : ℕ) 
  (remaining_files : ℕ) 
  (h1 : initial_music_files = 13) 
  (h2 : deleted_files = 10) 
  (h3 : remaining_files = 33) : 
  vanessa_video_files initial_music_files deleted_files remaining_files = 30 := by
  sorry

end vanessa_video_files_vanessa_video_files_proof_l1620_162019


namespace circle_equation_l1620_162058

/-- The circle C with center (a, b) in the second quadrant -/
structure Circle where
  a : ℝ
  b : ℝ
  h1 : a < 0
  h2 : b > 0

/-- The line 3x+y-5=0 -/
def line1 (x y : ℝ) : Prop := 3*x + y - 5 = 0

/-- The line 2x-3y+4=0 -/
def line2 (x y : ℝ) : Prop := 2*x - 3*y + 4 = 0

/-- The line 3x-4y+5=0 -/
def line3 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

/-- The x-axis -/
def xAxis (y : ℝ) : Prop := y = 0

/-- The intersection point of line1 and line2 -/
def intersectionPoint : ℝ × ℝ := (1, 2)

/-- Circle C passes through the intersection point -/
def passesThroughIntersection (C : Circle) : Prop :=
  let (x, y) := intersectionPoint
  (x - C.a)^2 + (y - C.b)^2 = C.b^2

/-- Circle C is tangent to line3 -/
def tangentToLine3 (C : Circle) : Prop :=
  |3*C.a - 4*C.b + 5| / 5 = C.b

/-- Circle C is tangent to x-axis -/
def tangentToXAxis (C : Circle) : Prop :=
  C.b = C.b

theorem circle_equation (C : Circle) 
  (h1 : passesThroughIntersection C)
  (h2 : tangentToLine3 C)
  (h3 : tangentToXAxis C) :
  ∀ (x y : ℝ), (x + 5)^2 + (y - 10)^2 = 100 ↔ (x - C.a)^2 + (y - C.b)^2 = C.b^2 :=
sorry

end circle_equation_l1620_162058


namespace product_of_primes_factors_l1620_162099

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def nth_prime (n : ℕ) : ℕ := sorry

def sum_of_products (k : ℕ) : ℕ := sorry

theorem product_of_primes_factors (k : ℕ) (h : k > 4) :
  ∃ (factors : List ℕ), 
    (∀ f ∈ factors, is_prime f) ∧ 
    (factors.length ≥ 2 * k) ∧
    (factors.prod = sum_of_products k + 1) :=
sorry

end product_of_primes_factors_l1620_162099


namespace olivia_made_45_dollars_l1620_162011

/-- The amount of money Olivia made selling chocolate bars -/
def olivia_earnings (bar_price : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * bar_price

/-- Proof that Olivia made $45 -/
theorem olivia_made_45_dollars :
  olivia_earnings 5 15 6 = 45 := by
  sorry

end olivia_made_45_dollars_l1620_162011


namespace line_parabola_properties_l1620_162028

/-- A line represented by the equation y = ax + b -/
structure Line where
  a : ℝ
  b : ℝ

/-- A parabola represented by the equation y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The theorem stating the properties of the line and parabola -/
theorem line_parabola_properties (l : Line) (p : Parabola)
    (h1 : l.a > 0)
    (h2 : p.a > 0)
    (h3 : l.b = p.b) :
    (∃ x y : ℝ, x = 0 ∧ y = l.b ∧ y = l.a * x + l.b ∧ y = p.a * x^2 + p.b) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → l.a * x₁ + l.b < l.a * x₂ + l.b) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → p.a * x₁^2 + p.b < p.a * x₂^2 + p.b) :=
  sorry


end line_parabola_properties_l1620_162028


namespace union_of_A_and_B_l1620_162074

-- Define the universe as the set of real numbers
def U : Set ℝ := Set.univ

-- Define sets A and B as open intervals
def A : Set ℝ := Set.Ioo 1 3
def B : Set ℝ := Set.Ioo 2 4

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Ioo 1 4 := by sorry

end union_of_A_and_B_l1620_162074


namespace gcd_lcm_sum_75_7350_l1620_162094

theorem gcd_lcm_sum_75_7350 : Nat.gcd 75 7350 + Nat.lcm 75 7350 = 3225 := by sorry

end gcd_lcm_sum_75_7350_l1620_162094


namespace equation_equivalence_l1620_162049

theorem equation_equivalence (x : ℝ) : x^2 + 4*x + 2 = 0 ↔ (x + 2)^2 = 2 := by
  sorry

end equation_equivalence_l1620_162049


namespace parking_methods_count_l1620_162060

/-- Represents the number of parking spaces --/
def parking_spaces : ℕ := 6

/-- Represents the number of cars to be parked --/
def cars : ℕ := 3

/-- Calculates the number of available slots for parking --/
def available_slots : ℕ := parking_spaces - cars + 1

/-- Calculates the number of ways to park cars --/
def parking_methods : ℕ := available_slots * (available_slots - 1) * (available_slots - 2)

/-- Theorem stating that the number of parking methods is 24 --/
theorem parking_methods_count : parking_methods = 24 := by sorry

end parking_methods_count_l1620_162060


namespace min_perimeter_is_16_l1620_162095

/-- A regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)
  (sideLength : ℝ)

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ :=
  p.sides * p.sideLength

/-- The configuration of polygons surrounding the triangle -/
structure PolygonConfiguration :=
  (p1 : RegularPolygon)
  (p2 : RegularPolygon)
  (p3 : RegularPolygon)

/-- The total perimeter of the configuration, excluding shared edges -/
def totalPerimeter (c : PolygonConfiguration) : ℝ :=
  perimeter c.p1 + perimeter c.p2 + perimeter c.p3 - 3 * 2

/-- The theorem stating the minimum perimeter -/
theorem min_perimeter_is_16 :
  ∃ (c : PolygonConfiguration),
    (c.p1.sideLength = 2 ∧ c.p2.sideLength = 2 ∧ c.p3.sideLength = 2) ∧
    (c.p1 = c.p2 ∨ c.p1 = c.p3 ∨ c.p2 = c.p3) ∧
    (∀ (d : PolygonConfiguration),
      (d.p1.sideLength = 2 ∧ d.p2.sideLength = 2 ∧ d.p3.sideLength = 2) →
      (d.p1 = d.p2 ∨ d.p1 = d.p3 ∨ d.p2 = d.p3) →
      totalPerimeter c ≤ totalPerimeter d) ∧
    totalPerimeter c = 16 :=
by sorry

end min_perimeter_is_16_l1620_162095


namespace space_shuttle_speed_conversion_l1620_162009

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * (60 * 60)

theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 2 = 7200 := by
  sorry

end space_shuttle_speed_conversion_l1620_162009


namespace quadratic_sum_l1620_162027

theorem quadratic_sum (a b : ℝ) : 
  a^2 - 2*a + 8 = 24 →
  b^2 - 2*b + 8 = 24 →
  a ≥ b →
  3*a + 2*b = 5 + Real.sqrt 17 := by
sorry

end quadratic_sum_l1620_162027


namespace quadratic_point_relationship_l1620_162054

/-- Given a quadratic function f(x) = -3x² + 2, prove that for points
    A(-1, y₁), B(1, y₂), and C(2, y₃) on its graph, y₁ = y₂ > y₃ holds. -/
theorem quadratic_point_relationship : ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -3 * (-1)^2 + 2) →
  (y₂ = -3 * 1^2 + 2) →
  (y₃ = -3 * 2^2 + 2) →
  (y₁ = y₂ ∧ y₁ > y₃) :=
by sorry

end quadratic_point_relationship_l1620_162054


namespace unique_digit_sum_count_l1620_162013

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Count of numbers in [1, 1000] with a given digit sum -/
def count_numbers_with_digit_sum (sum : ℕ) : ℕ := sorry

theorem unique_digit_sum_count :
  ∃! n : ℕ, n ∈ Finset.range 28 ∧ count_numbers_with_digit_sum n = 10 :=
sorry

end unique_digit_sum_count_l1620_162013


namespace shoebox_surface_area_l1620_162066

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a rectangular prism with dimensions 12 cm × 5 cm × 3 cm is 222 square centimeters -/
theorem shoebox_surface_area :
  surface_area 12 5 3 = 222 := by
  sorry

end shoebox_surface_area_l1620_162066


namespace notebook_cost_is_three_l1620_162047

/-- The cost of each notebook given the total spent, costs of other items, and number of notebooks. -/
def notebook_cost (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℚ) : ℚ :=
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

/-- Theorem stating that the cost of each notebook is $3 given the problem conditions. -/
theorem notebook_cost_is_three :
  notebook_cost 32 15 1 1 5 = 3 := by
  sorry

end notebook_cost_is_three_l1620_162047


namespace range_of_a_l1620_162053

theorem range_of_a (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 1) (ha : ∃ a : ℝ, a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  ∃ a : ℝ, a ∈ Set.Ioo 0 (7 / 27) ∨ a = 7 / 27 := by
  sorry

end range_of_a_l1620_162053


namespace solution_set_inequality_l1620_162024

/-- Given a > 0, prove that the solution set of |((2x - 3 - 2a) / (x - a))| ≤ 1 is {x | a + 1 ≤ x ≤ a + 3} -/
theorem solution_set_inequality (a : ℝ) (ha : a > 0) :
  {x : ℝ | |((2 * x - 3 - 2 * a) / (x - a))| ≤ 1} = {x : ℝ | a + 1 ≤ x ∧ x ≤ a + 3} := by
  sorry

end solution_set_inequality_l1620_162024


namespace max_sequence_length_l1620_162037

/-- Definition of the recurrence relation -/
def satisfies_recurrence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k, 2 ≤ k → k < n → a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1

/-- The maximum length of a sequence satisfying the recurrence relation is 4 -/
theorem max_sequence_length :
  ∀ n : ℕ, n > 0 →
    (∃ a : ℕ → ℕ, (∀ i, i ≤ n → a i > 0) ∧ satisfies_recurrence a n) →
    n ≤ 4 :=
sorry

end max_sequence_length_l1620_162037


namespace algebraic_expression_value_l1620_162005

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a^2 - 5*a + 2 = 0) (h3 : b^2 - 5*b + 2 = 0) : 
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13/2 := by
  sorry

end algebraic_expression_value_l1620_162005


namespace pipeline_equation_l1620_162067

/-- Represents the equation for a pipeline project with increased construction speed --/
theorem pipeline_equation (x : ℝ) (h : x > 0) :
  (35 / x) - (35 / ((1 + 0.2) * x)) = 7 :=
sorry

end pipeline_equation_l1620_162067


namespace renaldo_distance_l1620_162086

theorem renaldo_distance :
  ∀ (r : ℝ),
  (r + (1/3 * r + 7) = 27) →
  r = 15 := by
sorry

end renaldo_distance_l1620_162086


namespace rectangle_shading_l1620_162042

theorem rectangle_shading (total_rectangles : ℕ) 
  (h1 : total_rectangles = 12) : 
  (2 : ℚ) / 3 * (3 : ℚ) / 4 * total_rectangles = 6 := by
  sorry

end rectangle_shading_l1620_162042


namespace half_plus_five_equals_thirteen_l1620_162069

theorem half_plus_five_equals_thirteen (n : ℝ) : (1/2 : ℝ) * n + 5 = 13 → n = 16 := by
  sorry

end half_plus_five_equals_thirteen_l1620_162069


namespace hershel_goldfish_count_l1620_162093

/-- The number of goldfish Hershel had initially -/
def initial_goldfish : ℕ := 15

theorem hershel_goldfish_count :
  let initial_betta : ℕ := 10
  let bexley_betta : ℕ := (2 * initial_betta) / 5
  let bexley_goldfish : ℕ := initial_goldfish / 3
  let total_fish : ℕ := initial_betta + bexley_betta + initial_goldfish + bexley_goldfish
  let remaining_fish : ℕ := total_fish / 2
  remaining_fish = 17 :=
by sorry

end hershel_goldfish_count_l1620_162093


namespace unique_solution_for_equation_l1620_162036

/-- Given that x is prime and y is odd, prove that x^2 + y = 2007 has only one solution -/
theorem unique_solution_for_equation (x y : ℕ) :
  Prime x → Odd y → (x^2 + y = 2007) → ∃! (x y : ℕ), Prime x ∧ Odd y ∧ x^2 + y = 2007 := by
  sorry

end unique_solution_for_equation_l1620_162036


namespace trigonometric_simplification_l1620_162079

theorem trigonometric_simplification (α : Real) 
  (h1 : π/2 < α ∧ α < π) : 
  (Real.sqrt (1 + 2 * Real.sin (5 * π - α) * Real.cos (α - π))) / 
  (Real.sin (α - 3 * π / 2) - Real.sqrt (1 - Real.sin (3 * π / 2 + α) ^ 2)) = -1 := by
  sorry

end trigonometric_simplification_l1620_162079


namespace number_division_problem_l1620_162080

theorem number_division_problem (x : ℝ) : (x + 17) / 5 = 25 → x / 6 = 18 := by
  sorry

end number_division_problem_l1620_162080


namespace dosage_for_package_l1620_162010

-- Define the dosage function
def dosage (x : ℝ) : ℝ := 10 * x + 10

-- Define the weight range
def valid_weight (x : ℝ) : Prop := 5 ≤ x ∧ x ≤ 50

-- Define the safe dosage range for a 300 mg package
def safe_dosage (y : ℝ) : Prop := 250 ≤ y ∧ y ≤ 300

-- Theorem statement
theorem dosage_for_package (x : ℝ) (h1 : 24 ≤ x) (h2 : x ≤ 29) (h3 : valid_weight x) :
  safe_dosage (dosage x) :=
sorry

end dosage_for_package_l1620_162010


namespace min_value_expression_l1620_162081

theorem min_value_expression (x : ℝ) (h : x > 0) :
  x^3 + 12*x + 81/x^4 ≥ 24 ∧
  ∀ ε > 0, ∃ y > 0, x^3 + 12*x + 81/x^4 < 24 + ε :=
sorry

end min_value_expression_l1620_162081


namespace fountain_distance_is_30_l1620_162020

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℕ :=
  total_distance / num_trips

/-- Theorem stating that the water fountain is 30 feet from Mrs. Hilt's desk -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end fountain_distance_is_30_l1620_162020


namespace inequality_proof_l1620_162083

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt ((1 + a^2 * b) / (1 + a * b)) + Real.sqrt ((1 + b^2 * c) / (1 + b * c)) + Real.sqrt ((1 + c^2 * a) / (1 + c * a)) ≥ 3 := by
  sorry

end inequality_proof_l1620_162083


namespace part_one_part_two_l1620_162088

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem for part I
theorem part_one (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) → a = 2 := by
  sorry

-- Theorem for part II
theorem part_two (a : ℝ) : (∀ x, x ∈ A a → x ∈ B) → a ≤ 0 ∨ a ≥ 4 := by
  sorry

end part_one_part_two_l1620_162088


namespace barnyard_owls_count_l1620_162085

/-- The number of hoot sounds one barnyard owl makes per minute. -/
def hoots_per_owl : ℕ := 5

/-- The total number of hoots heard per minute. -/
def total_hoots : ℕ := 20 - 5

/-- The number of barnyard owls making the noise. -/
def num_owls : ℕ := total_hoots / hoots_per_owl

theorem barnyard_owls_count : num_owls = 3 := by
  sorry

end barnyard_owls_count_l1620_162085


namespace cos_five_pi_thirds_l1620_162001

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by
  sorry

end cos_five_pi_thirds_l1620_162001


namespace balloon_ascent_rate_l1620_162078

/-- The rate of descent of the balloon in feet per minute -/
def descent_rate : ℝ := 10

/-- The duration of the first ascent in minutes -/
def first_ascent_duration : ℝ := 15

/-- The duration of the descent in minutes -/
def descent_duration : ℝ := 10

/-- The duration of the second ascent in minutes -/
def second_ascent_duration : ℝ := 15

/-- The maximum height reached by the balloon in feet -/
def max_height : ℝ := 1400

/-- The theorem stating the rate of ascent of the balloon -/
theorem balloon_ascent_rate :
  ∃ (ascent_rate : ℝ),
    ascent_rate * first_ascent_duration
    - descent_rate * descent_duration
    + ascent_rate * second_ascent_duration
    = max_height
    ∧ ascent_rate = 50 := by sorry

end balloon_ascent_rate_l1620_162078


namespace prob_odd_sum_is_correct_l1620_162097

/-- Represents the dartboard with given radii and scoring regions -/
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  inner_scores : Fin 3 → ℕ
  intermediate_scores : Fin 3 → ℕ
  outer_scores : Fin 3 → ℕ

/-- Calculates the probability of getting an odd sum when throwing two darts -/
def prob_odd_sum (d : Dartboard) : ℚ :=
  265 / 855

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard where
  inner_radius := 4.5
  intermediate_radius := 6.75
  outer_radius := 9
  inner_scores := ![3, 2, 2]
  intermediate_scores := ![2, 1, 1]
  outer_scores := ![1, 1, 3]

theorem prob_odd_sum_is_correct :
  prob_odd_sum problem_dartboard = 265 / 855 := by
  sorry

end prob_odd_sum_is_correct_l1620_162097


namespace exponential_inequality_supremum_l1620_162025

theorem exponential_inequality_supremum : 
  (∃ a : ℝ, (∀ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) → m < a) ∧ 
   (∀ ε > 0, ∃ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) ∧ m > a - ε)) → 
  (∃ a : ℝ, a = 1 ∧ 
   (∀ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) → m < a) ∧ 
   (∀ ε > 0, ∃ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) ∧ m > a - ε)) :=
by sorry

end exponential_inequality_supremum_l1620_162025


namespace isosceles_triangle_side_length_l1620_162044

/-- An ellipse with equation x^2 + 9y^2 = 9 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 9 * p.2^2 = 9}

/-- An isosceles triangle inscribed in the ellipse -/
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_in_ellipse : A ∈ Ellipse ∧ B ∈ Ellipse ∧ C ∈ Ellipse
  h_isosceles : dist A B = dist A C
  h_vertex_at_origin : A = (0, 1)
  h_altitude_on_y_axis : B.1 + C.1 = 0 ∧ B.2 = C.2

/-- The square of the length of the equal sides of the isosceles triangle -/
def squareLengthEqualSides (t : IsoscelesTriangle) : ℝ :=
  (dist t.A t.B)^2

/-- The main theorem -/
theorem isosceles_triangle_side_length 
  (t : IsoscelesTriangle) : squareLengthEqualSides t = 108/25 := by
  sorry


end isosceles_triangle_side_length_l1620_162044


namespace right_triangle_area_l1620_162004

theorem right_triangle_area (a b c : ℝ) (h1 : a = 18) (h2 : c = 30) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 := by
  sorry

end right_triangle_area_l1620_162004


namespace reaction_outcome_l1620_162012

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  alMoles : ℚ
  h2so4Moles : ℚ
  al2so43Moles : ℚ
  h2Moles : ℚ

/-- The balanced equation for the reaction -/
def balancedReaction : ChemicalReaction :=
  { alMoles := 2
  , h2so4Moles := 3
  , al2so43Moles := 1
  , h2Moles := 3 }

/-- The given amounts of reactants -/
def givenReactants : ChemicalReaction :=
  { alMoles := 2
  , h2so4Moles := 3
  , al2so43Moles := 0
  , h2Moles := 0 }

/-- Checks if the reaction is balanced -/
def isBalanced (r : ChemicalReaction) : Prop :=
  r.alMoles / balancedReaction.alMoles = r.h2so4Moles / balancedReaction.h2so4Moles

/-- Calculates the limiting factor of the reaction -/
def limitingFactor (r : ChemicalReaction) : ℚ :=
  min (r.alMoles / balancedReaction.alMoles) (r.h2so4Moles / balancedReaction.h2so4Moles)

/-- Calculates the products formed in the reaction -/
def productsFormed (r : ChemicalReaction) : ChemicalReaction :=
  let factor := limitingFactor r
  { alMoles := r.alMoles - factor * balancedReaction.alMoles
  , h2so4Moles := r.h2so4Moles - factor * balancedReaction.h2so4Moles
  , al2so43Moles := factor * balancedReaction.al2so43Moles
  , h2Moles := factor * balancedReaction.h2Moles }

theorem reaction_outcome :
  isBalanced givenReactants ∧
  (productsFormed givenReactants).al2so43Moles = 1 ∧
  (productsFormed givenReactants).h2Moles = 3 ∧
  (productsFormed givenReactants).alMoles = 0 ∧
  (productsFormed givenReactants).h2so4Moles = 0 :=
sorry

end reaction_outcome_l1620_162012


namespace minimum_distance_theorem_l1620_162055

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the position and speed of a character -/
structure Character where
  start : Point
  speed : ℝ

/-- The minimum distance between two characters chasing a target -/
def minDistance (char1 char2 : Character) (target : Point) : ℝ :=
  sorry

/-- Garfield's initial position and speed -/
def garfield : Character :=
  { start := { x := 0, y := 0 }, speed := 7 }

/-- Odie's initial position and speed -/
def odie : Character :=
  { start := { x := 25, y := 0 }, speed := 10 }

/-- The target point both characters are chasing -/
def target : Point :=
  { x := 9, y := 12 }

theorem minimum_distance_theorem :
  minDistance garfield odie target = 10 / Real.sqrt 149 :=
sorry

end minimum_distance_theorem_l1620_162055


namespace dogGroupings_eq_4200_l1620_162006

/-- The number of ways to divide 12 dogs into groups of 4, 5, and 3,
    with Fluffy in the 4-dog group and Nipper in the 5-dog group -/
def dogGroupings : ℕ :=
  let totalDogs : ℕ := 12
  let group1Size : ℕ := 4  -- Fluffy's group
  let group2Size : ℕ := 5  -- Nipper's group
  let group3Size : ℕ := 3
  let remainingDogs : ℕ := totalDogs - 2  -- Excluding Fluffy and Nipper

  (remainingDogs.choose (group1Size - 1)) * ((remainingDogs - (group1Size - 1)).choose (group2Size - 1))

theorem dogGroupings_eq_4200 : dogGroupings = 4200 := by
  sorry

end dogGroupings_eq_4200_l1620_162006


namespace parallel_vectors_imply_x_equals_three_l1620_162064

/-- Given vectors a, b, c in ℝ², if a + c is parallel to b + c, then the x-coordinate of c is 3. -/
theorem parallel_vectors_imply_x_equals_three :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![3, -1]
  let c : Fin 2 → ℝ := ![x, 4]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + c) = k • (b + c)) →
  x = 3 :=
by
  sorry

end parallel_vectors_imply_x_equals_three_l1620_162064


namespace units_digit_of_2137_pow_753_l1620_162077

theorem units_digit_of_2137_pow_753 : ∃ n : ℕ, 2137^753 ≡ 7 [ZMOD 10] :=
sorry

end units_digit_of_2137_pow_753_l1620_162077


namespace subset_of_intersection_eq_union_l1620_162090

theorem subset_of_intersection_eq_union {A B C : Set α} 
  (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) 
  (h : A ∩ B = B ∪ C) : C ⊆ B := by
  sorry

end subset_of_intersection_eq_union_l1620_162090


namespace difference_of_squares_l1620_162059

theorem difference_of_squares : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_l1620_162059


namespace triangle_side_length_l1620_162084

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a = Real.sqrt 3 ∧  -- Given condition
  Real.sin B = 1 / 2 ∧  -- Given condition
  C = π / 6 ∧  -- Given condition
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  a / Real.sin A = c / Real.sin C  -- Law of sines
  →
  b = 1 := by
sorry

end triangle_side_length_l1620_162084


namespace ravi_coins_l1620_162051

def coin_problem (nickels quarters dimes : ℕ) (nickel_value quarter_value dime_value : ℚ) : Prop :=
  nickels = 6 ∧
  quarters = nickels + 2 ∧
  dimes = quarters + 4 ∧
  nickel_value = 5/100 ∧
  quarter_value = 25/100 ∧
  dime_value = 10/100 ∧
  nickels * nickel_value + quarters * quarter_value + dimes * dime_value = 7/2

theorem ravi_coins :
  ∃ (nickels quarters dimes : ℕ) (nickel_value quarter_value dime_value : ℚ),
    coin_problem nickels quarters dimes nickel_value quarter_value dime_value :=
by
  sorry

#check ravi_coins

end ravi_coins_l1620_162051


namespace circular_arrangement_students_l1620_162030

/-- 
Given a circular arrangement of students, if the 10th and 40th positions 
are opposite each other, then the total number of students is 62.
-/
theorem circular_arrangement_students (n : ℕ) : 
  (∃ (a b : ℕ), a = 10 ∧ b = 40 ∧ a < b ∧ b - a = n - (b - a)) → n = 62 :=
by sorry

end circular_arrangement_students_l1620_162030


namespace sqrt_meaningful_range_l1620_162039

theorem sqrt_meaningful_range (m : ℝ) : 
  (∃ (x : ℝ), x^2 = m + 4) ↔ m ≥ -4 := by
  sorry

end sqrt_meaningful_range_l1620_162039


namespace briellesClockRings_l1620_162032

/-- Calculates the number of times a clock rings in a day -/
def ringsPerDay (startHour interval : ℕ) : ℕ :=
  (24 - startHour + interval - 1) / interval

/-- Represents the clock's ringing pattern over three days -/
structure ClockPattern where
  day1Interval : ℕ
  day1Start : ℕ
  day2Interval : ℕ
  day2Start : ℕ
  day3Interval : ℕ
  day3Start : ℕ

/-- Calculates the total number of rings for the given clock pattern -/
def totalRings (pattern : ClockPattern) : ℕ :=
  ringsPerDay pattern.day1Start pattern.day1Interval +
  ringsPerDay pattern.day2Start pattern.day2Interval +
  ringsPerDay pattern.day3Start pattern.day3Interval

/-- The specific clock pattern from the problem -/
def briellesClockPattern : ClockPattern :=
  { day1Interval := 3
    day1Start := 1
    day2Interval := 4
    day2Start := 2
    day3Interval := 5
    day3Start := 3 }

theorem briellesClockRings :
  totalRings briellesClockPattern = 19 := by
  sorry


end briellesClockRings_l1620_162032
