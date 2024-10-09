import Mathlib

namespace factorial_expression_equiv_l1995_199574

theorem factorial_expression_equiv :
  6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := 
sorry

end factorial_expression_equiv_l1995_199574


namespace calculate_expression_l1995_199555

theorem calculate_expression :
  (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := 
by sorry

end calculate_expression_l1995_199555


namespace x_squared_plus_y_squared_l1995_199569

theorem x_squared_plus_y_squared (x y : ℝ) 
   (h1 : (x + y)^2 = 49) 
   (h2 : x * y = 8) 
   : x^2 + y^2 = 33 := 
by
  sorry

end x_squared_plus_y_squared_l1995_199569


namespace Cd_sum_l1995_199505

theorem Cd_sum : ∀ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 3 → (C / (x-3) + D * (x+2) = (-2 * x^2 + 8 * x + 28) / (x-3))) → 
  (C + D = 20) :=
by
  intros C D h
  sorry

end Cd_sum_l1995_199505


namespace total_birdseed_amount_l1995_199564

-- Define the birdseed amounts in the boxes
def box1_amount : ℕ := 250
def box2_amount : ℕ := 275
def box3_amount : ℕ := 225
def box4_amount : ℕ := 300
def box5_amount : ℕ := 275
def box6_amount : ℕ := 200
def box7_amount : ℕ := 150
def box8_amount : ℕ := 180

-- Define the weekly consumption of each bird
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def canary_consumption : ℕ := 25

-- Define a theorem to calculate the total birdseed that Leah has
theorem total_birdseed_amount : box1_amount + box2_amount + box3_amount + box4_amount + box5_amount + box6_amount + box7_amount + box8_amount = 1855 :=
by
  sorry

end total_birdseed_amount_l1995_199564


namespace number_of_men_in_first_group_l1995_199556

-- Definitions based on the conditions provided
def work_done (men : ℕ) (days : ℕ) (work_rate : ℝ) : ℝ :=
  men * days * work_rate

-- Given conditions
def condition1 (M : ℕ) : Prop :=
  ∃ work_rate : ℝ, work_done M 12 work_rate = 66

def condition2 : Prop :=
  ∃ work_rate : ℝ, work_done 86 8 work_rate = 189.2

-- Proof goal
theorem number_of_men_in_first_group : 
  ∀ M : ℕ, condition1 M → condition2 → M = 57 := by
  sorry

end number_of_men_in_first_group_l1995_199556


namespace exercise_books_purchasing_methods_l1995_199558

theorem exercise_books_purchasing_methods :
  ∃ (ways : ℕ), ways = 5 ∧
  (∃ (x y z : ℕ), 2 * x + 5 * y + 11 * z = 40 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) ∧
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ : ℕ),
    2 * x₁ + 5 * y₁ + 11 * z₂ = 40 ∧ x₁ ≥ 1 ∧ y₁ ≥ 1 ∧ z₁ ≥ 1 →
    2 * x₂ + 5 * y₂ + 11 * z₂ = 40 ∧ x₂ ≥ 1 ∧ y₂ ≥ 1 ∧ z₂ ≥ 1 →
    (x₁, y₁, z₁) = (x₂, y₂, z₂)) := sorry

end exercise_books_purchasing_methods_l1995_199558


namespace largest_among_five_numbers_l1995_199541

theorem largest_among_five_numbers :
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  sorry

end largest_among_five_numbers_l1995_199541


namespace max_sqrt_expr_l1995_199550

variable {x y z : ℝ}

noncomputable def f (x y z : ℝ) : ℝ := Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)

theorem max_sqrt_expr (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  f x y z ≤ 2 * Real.sqrt 3 := by
  sorry

end max_sqrt_expr_l1995_199550


namespace sum_of_prime_factors_of_91_l1995_199589

theorem sum_of_prime_factors_of_91 : 
  (¬ (91 % 2 = 0)) ∧ 
  (¬ (91 % 3 = 0)) ∧ 
  (¬ (91 % 5 = 0)) ∧ 
  (91 = 7 * 13) →
  (7 + 13 = 20) := 
by 
  intros h
  sorry

end sum_of_prime_factors_of_91_l1995_199589


namespace ratio_of_boys_to_total_l1995_199529

theorem ratio_of_boys_to_total (p_b p_g : ℝ) (h1 : p_b + p_g = 1) (h2 : p_b = (2 / 3) * p_g) :
  p_b = 2 / 5 :=
by
  sorry

end ratio_of_boys_to_total_l1995_199529


namespace range_of_a_l1995_199599

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, (0 < x) ∧ (x + 1/x < a)) ↔ a ≤ 2 :=
by {
  sorry
}

end range_of_a_l1995_199599


namespace sqrt_74_between_8_and_9_product_of_consecutive_integers_l1995_199559

theorem sqrt_74_between_8_and_9 : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9 := sorry

theorem product_of_consecutive_integers (h : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9) : 8 * 9 = 72 := by
  have h1 : 8 < Real.sqrt 74 := And.left h
  have h2 : Real.sqrt 74 < 9 := And.right h
  calc
    8 * 9 = 72 := by norm_num

end sqrt_74_between_8_and_9_product_of_consecutive_integers_l1995_199559


namespace find_C_l1995_199542

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 := 
by
  sorry

end find_C_l1995_199542


namespace lcm_of_two_numbers_l1995_199522

theorem lcm_of_two_numbers (A B : ℕ) (h1 : A * B = 62216) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2828 :=
by
  sorry

end lcm_of_two_numbers_l1995_199522


namespace quadratic_positivity_range_l1995_199513

variable (a : ℝ)

def quadratic_function (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

theorem quadratic_positivity_range :
  (∀ x, 0 < x ∧ x < 3 → quadratic_function a x > 0)
  ↔ (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) := sorry

end quadratic_positivity_range_l1995_199513


namespace diagonals_sum_pentagon_inscribed_in_circle_l1995_199585

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end diagonals_sum_pentagon_inscribed_in_circle_l1995_199585


namespace ben_has_20_mms_l1995_199545

theorem ben_has_20_mms (B_candies Ben_candies : ℕ) 
  (h1 : B_candies = 50) 
  (h2 : B_candies = Ben_candies + 30) : 
  Ben_candies = 20 := 
by
  sorry

end ben_has_20_mms_l1995_199545


namespace central_angle_of_sector_with_area_one_l1995_199586

theorem central_angle_of_sector_with_area_one (θ : ℝ):
  (1 / 2) * θ = 1 → θ = 2 :=
by
  sorry

end central_angle_of_sector_with_area_one_l1995_199586


namespace greatest_common_divisor_546_180_l1995_199596

theorem greatest_common_divisor_546_180 : 
  ∃ d, d < 70 ∧ d > 0 ∧ d ∣ 546 ∧ d ∣ 180 ∧ ∀ x, x < 70 ∧ x > 0 ∧ x ∣ 546 ∧ x ∣ 180 → x ≤ d → x = 6 :=
by
  sorry

end greatest_common_divisor_546_180_l1995_199596


namespace largest_four_digit_mod_5_l1995_199539

theorem largest_four_digit_mod_5 : ∃ (n : ℤ), n % 5 = 3 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℤ, m % 5 = 3 ∧ 1000 ≤ m ∧ m ≤ 9999 → m ≤ n :=
sorry

end largest_four_digit_mod_5_l1995_199539


namespace range_of_m_l1995_199554

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → x^2 - 2 * x - 3 > 0) → (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l1995_199554


namespace cost_of_gravelling_path_eq_630_l1995_199566

-- Define the dimensions of the grassy plot.
def length_grassy_plot : ℝ := 110
def width_grassy_plot : ℝ := 65

-- Define the width of the gravel path.
def width_gravel_path : ℝ := 2.5

-- Define the cost of gravelling per square meter in INR.
def cost_per_sqm : ℝ := 0.70

-- Compute the dimensions of the plot including the gravel path.
def length_including_path := length_grassy_plot + 2 * width_gravel_path
def width_including_path := width_grassy_plot + 2 * width_gravel_path

-- Compute the area of the plot including the gravel path.
def area_including_path := length_including_path * width_including_path

-- Compute the area of the grassy plot without the gravel path.
def area_grassy_plot := length_grassy_plot * width_grassy_plot

-- Compute the area of the gravel path alone.
def area_gravel_path := area_including_path - area_grassy_plot

-- Compute the total cost of gravelling the path.
def total_cost := area_gravel_path * cost_per_sqm

-- The theorem stating the cost of gravelling the path.
theorem cost_of_gravelling_path_eq_630 : total_cost = 630 := by
  -- Proof goes here
  sorry

end cost_of_gravelling_path_eq_630_l1995_199566


namespace ratio_is_one_half_l1995_199512

noncomputable def ratio_of_intercepts (b : ℝ) (hb : b ≠ 0) : ℝ :=
  let s := -b / 8
  let t := -b / 4
  s / t

theorem ratio_is_one_half (b : ℝ) (hb : b ≠ 0) :
  ratio_of_intercepts b hb = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l1995_199512


namespace largest_possible_m_value_l1995_199511

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_possible_m_value :
  ∃ (m x y : ℕ), is_three_digit m ∧ is_prime x ∧ is_prime y ∧ x ≠ y ∧
  x < 10 ∧ y < 10 ∧ is_prime (10 * x - y) ∧ m = x * y * (10 * x - y) ∧ m = 705 := sorry

end largest_possible_m_value_l1995_199511


namespace area_of_triangle_is_right_angled_l1995_199568

noncomputable def vector_a : ℝ × ℝ := (3, 4)
noncomputable def vector_b : ℝ × ℝ := (-4, 3)

theorem area_of_triangle_is_right_angled (h1 : vector_a = (3, 4)) (h2 : vector_b = (-4, 3)) : 
  let det := vector_a.1 * vector_b.2 - vector_a.2 * vector_b.1
  (1 / 2) * abs det = 12.5 :=
by
  sorry

end area_of_triangle_is_right_angled_l1995_199568


namespace trapezoid_cd_length_l1995_199579

noncomputable def proof_cd_length (AD BC CD : ℝ) (BD : ℝ) (angle_DBA angle_BDC : ℝ) (ratio_BC_AD : ℝ) : Prop :=
  AD > 0 ∧ BC > 0 ∧
  BD = 1 ∧
  angle_DBA = 23 ∧
  angle_BDC = 46 ∧
  ratio_BC_AD = 9 / 5 ∧
  AD / BC = 5 / 9 ∧
  CD = 4 / 5

theorem trapezoid_cd_length
  (AD BC CD : ℝ)
  (BD : ℝ := 1)
  (angle_DBA : ℝ := 23)
  (angle_BDC : ℝ := 46)
  (ratio_BC_AD : ℝ := 9 / 5)
  (h_conditions : proof_cd_length AD BC CD BD angle_DBA angle_BDC ratio_BC_AD) : CD = 4 / 5 :=
sorry

end trapezoid_cd_length_l1995_199579


namespace solution_set_of_inequality_l1995_199504

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2 * x > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l1995_199504


namespace total_pies_eq_l1995_199507

-- Definitions for the number of pies made by each person
def pinky_pies : ℕ := 147
def helen_pies : ℕ := 56
def emily_pies : ℕ := 89
def jake_pies : ℕ := 122

-- The theorem stating the total number of pies
theorem total_pies_eq : pinky_pies + helen_pies + emily_pies + jake_pies = 414 :=
by sorry

end total_pies_eq_l1995_199507


namespace function_odd_domain_of_f_range_of_f_l1995_199546

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem function_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

theorem domain_of_f : ∀ x : ℝ, true :=
by
  intro x
  trivial

theorem range_of_f : ∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  intro y
  sorry

end function_odd_domain_of_f_range_of_f_l1995_199546


namespace power_equation_l1995_199577

theorem power_equation (x a b : ℝ) (ha : 3^x = a) (hb : 5^x = b) : 45^x = a^2 * b :=
sorry

end power_equation_l1995_199577


namespace probability_C_l1995_199547

variable (pA pB pD pC : ℚ)
variable (hA : pA = 1 / 4)
variable (hB : pB = 1 / 3)
variable (hD : pD = 1 / 6)
variable (total_prob : pA + pB + pD + pC = 1)

theorem probability_C (hA : pA = 1 / 4) (hB : pB = 1 / 3) (hD : pD = 1 / 6) (total_prob : pA + pB + pD + pC = 1) : pC = 1 / 4 :=
sorry

end probability_C_l1995_199547


namespace value_of_f_5_l1995_199543

variable (a b c m : ℝ)

-- Conditions: definition of f and given value of f(-5)
def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2
axiom H1 : f a b c (-5) = m

-- Question: Prove that f(5) = -m + 4
theorem value_of_f_5 : f a b c 5 = -m + 4 :=
by
  sorry

end value_of_f_5_l1995_199543


namespace fraction_squares_sum_l1995_199518

theorem fraction_squares_sum (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := 
sorry

end fraction_squares_sum_l1995_199518


namespace min_abc_value_l1995_199510

noncomputable def minValue (a b c : ℝ) : ℝ := (a + b) / (a * b * c)

theorem min_abc_value (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
  (minValue a b c) ≥ 16 :=
by
  sorry

end min_abc_value_l1995_199510


namespace polynomial_sum_l1995_199535

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - 5 * x - 7
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 2 * x + 8

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -5 * x^3 + 11 * x^2 + x - 8 :=
  sorry

end polynomial_sum_l1995_199535


namespace gcd_228_1995_l1995_199594

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l1995_199594


namespace ratio_of_chords_l1995_199523

theorem ratio_of_chords 
  (E F G H Q : Type)
  (EQ GQ FQ HQ : ℝ)
  (h1 : EQ = 4)
  (h2 : GQ = 10)
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 5 / 2 := 
by 
  sorry

end ratio_of_chords_l1995_199523


namespace sphere_volume_increase_l1995_199598

theorem sphere_volume_increase 
  (r : ℝ) 
  (S : ℝ := 4 * Real.pi * r^2) 
  (V : ℝ := (4/3) * Real.pi * r^3)
  (k : ℝ := 2) 
  (h : 4 * S = 4 * Real.pi * (k * r)^2) : 
  ((4/3) * Real.pi * (2 * r)^3) = 8 * V := 
by
  sorry

end sphere_volume_increase_l1995_199598


namespace triangle_perimeter_l1995_199508

theorem triangle_perimeter :
  let a := 15
  let b := 10
  let c := 12
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a + b + c = 37) :=
by
  intros
  sorry

end triangle_perimeter_l1995_199508


namespace train_length_l1995_199536

/-
  Given:
  - Speed of the train is 78 km/h
  - Time to pass an electric pole is 5.0769230769230775 seconds
  We need to prove that the length of the train is 110 meters.
-/

def speed_kmph : ℝ := 78
def time_seconds : ℝ := 5.0769230769230775
def expected_length_meters : ℝ := 110

theorem train_length :
  (speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters :=
by {
  -- Proof goes here
  sorry
}

end train_length_l1995_199536


namespace total_students_l1995_199515

theorem total_students (boys girls : ℕ) (h_boys : boys = 127) (h_girls : girls = boys + 212) : boys + girls = 466 :=
by
  sorry

end total_students_l1995_199515


namespace similar_triangles_perimeter_ratio_l1995_199502

theorem similar_triangles_perimeter_ratio
  (a₁ a₂ s₁ s₂ : ℝ)
  (h₁ : a₁ / a₂ = 1 / 4)
  (h₂ : s₁ / s₂ = 1 / 2) :
  (s₁ / s₂ = 1 / 2) :=
by {
  sorry
}

end similar_triangles_perimeter_ratio_l1995_199502


namespace solve_medium_apple_cost_l1995_199544

def cost_small_apple : ℝ := 1.5
def cost_big_apple : ℝ := 3.0
def num_small_apples : ℕ := 6
def num_medium_apples : ℕ := 6
def num_big_apples : ℕ := 8
def total_cost : ℝ := 45

noncomputable def cost_medium_apple (M : ℝ) : Prop :=
  (6 * cost_small_apple) + (6 * M) + (8 * cost_big_apple) = total_cost

theorem solve_medium_apple_cost : ∃ M : ℝ, cost_medium_apple M ∧ M = 2 := by
  sorry

end solve_medium_apple_cost_l1995_199544


namespace squirrel_rise_per_circuit_l1995_199533

theorem squirrel_rise_per_circuit
  (h_post_height : ℕ := 12)
  (h_circumference : ℕ := 3)
  (h_travel_distance : ℕ := 9) :
  (h_post_height / (h_travel_distance / h_circumference) = 4) :=
  sorry

end squirrel_rise_per_circuit_l1995_199533


namespace second_more_than_third_l1995_199583

def firstChapterPages : ℕ := 35
def secondChapterPages : ℕ := 18
def thirdChapterPages : ℕ := 3

theorem second_more_than_third : secondChapterPages - thirdChapterPages = 15 := by
  sorry

end second_more_than_third_l1995_199583


namespace breadthOfRectangularPart_l1995_199576

variable (b l : ℝ)

def rectangularAreaProblem : Prop :=
  (l * b + (1 / 12) * b * l = 24 * b) ∧ (l - b = 10)

theorem breadthOfRectangularPart :
  rectangularAreaProblem b l → b = 12.15 :=
by
  intros
  sorry

end breadthOfRectangularPart_l1995_199576


namespace thickness_of_stack_l1995_199530

theorem thickness_of_stack (books : ℕ) (avg_pages_per_book : ℕ) (pages_per_inch : ℕ) (total_pages : ℕ) (thick_in_inches : ℕ)
    (h1 : books = 6)
    (h2 : avg_pages_per_book = 160)
    (h3 : pages_per_inch = 80)
    (h4 : total_pages = books * avg_pages_per_book)
    (h5 : thick_in_inches = total_pages / pages_per_inch) :
    thick_in_inches = 12 :=
by {
    -- statement without proof
    sorry
}

end thickness_of_stack_l1995_199530


namespace sphere_radius_ratio_l1995_199573

theorem sphere_radius_ratio (R1 R2 : ℝ) (m n : ℝ) (hm : 1 < m) (hn : 1 < n) 
  (h_ratio1 : (2 * π * R1 * ((2 * R1) / (m + 1))) / (4 * π * R1 * R1) = 1 / (m + 1))
  (h_ratio2 : (2 * π * R2 * ((2 * R2) / (n + 1))) / (4 * π * R2 * R2) = 1 / (n + 1)): 
  R2 / R1 = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := 
by
  sorry

end sphere_radius_ratio_l1995_199573


namespace asymptotes_and_foci_of_hyperbola_l1995_199567

def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

theorem asymptotes_and_foci_of_hyperbola :
  (∀ x y : ℝ, hyperbola x y → y = x * (3 / 4) ∨ y = x * -(3 / 4)) ∧
  (∃ x y : ℝ, (x, y) = (15, 0) ∨ (x, y) = (-15, 0)) :=
by {
  -- prove these conditions here
  sorry 
}

end asymptotes_and_foci_of_hyperbola_l1995_199567


namespace if_a_eq_b_then_a_squared_eq_b_squared_l1995_199540

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end if_a_eq_b_then_a_squared_eq_b_squared_l1995_199540


namespace production_equipment_B_l1995_199552

theorem production_equipment_B :
  ∃ (X Y : ℕ), X + Y = 4800 ∧ (50 / 80 = 5 / 8) ∧ (X / 4800 = 5 / 8) ∧ Y = 1800 :=
by
  sorry

end production_equipment_B_l1995_199552


namespace increasing_on_neg_reals_l1995_199509

variable (f : ℝ → ℝ)

def even_function : Prop := ∀ x : ℝ, f (-x) = f x

def decreasing_on_pos_reals : Prop := ∀ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2 ∧ x1 < x2) → f x1 > f x2

theorem increasing_on_neg_reals
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on_pos_reals f) :
  ∀ x1 x2 : ℝ, (x1 < 0 ∧ x2 < 0 ∧ x1 < x2) → f x1 < f x2 :=
by sorry

end increasing_on_neg_reals_l1995_199509


namespace right_angled_triangles_with_cathetus_2021_l1995_199560

theorem right_angled_triangles_with_cathetus_2021 :
  ∃ n : Nat, n = 4 ∧ ∀ (a b c : ℕ), ((a = 2021 ∧ a * a + b * b = c * c) ↔ (a = 2021 ∧ 
    ∃ m n, (m > n ∧ m > 0 ∧ n > 0 ∧ 2021 = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2))) :=
sorry

end right_angled_triangles_with_cathetus_2021_l1995_199560


namespace negation_of_p_l1995_199519

variable {x : ℝ}

def proposition_p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

theorem negation_of_p :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
sorry

end negation_of_p_l1995_199519


namespace petya_no_win_implies_draw_or_lost_l1995_199563

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end petya_no_win_implies_draw_or_lost_l1995_199563


namespace glasses_displayed_is_correct_l1995_199527

-- Definitions from the problem conditions
def tall_cupboard_capacity : Nat := 20
def wide_cupboard_capacity : Nat := 2 * tall_cupboard_capacity
def per_shelf_narrow_cupboard : Nat := 15 / 3
def usable_narrow_cupboard_capacity : Nat := 2 * per_shelf_narrow_cupboard

-- Theorem to prove that the total number of glasses displayed is 70
theorem glasses_displayed_is_correct :
  (tall_cupboard_capacity + wide_cupboard_capacity + usable_narrow_cupboard_capacity) = 70 :=
by
  sorry

end glasses_displayed_is_correct_l1995_199527


namespace ratio_adult_women_to_men_event_l1995_199524

theorem ratio_adult_women_to_men_event :
  ∀ (total_members men_ratio women_ratio children : ℕ), 
  total_members = 2000 →
  men_ratio = 30 →
  children = 200 →
  women_ratio = men_ratio →
  women_ratio / men_ratio = 1 / 1 := 
by
  intros total_members men_ratio women_ratio children
  sorry

end ratio_adult_women_to_men_event_l1995_199524


namespace workers_distribution_l1995_199572

theorem workers_distribution (x y : ℕ) (h1 : x + y = 32) (h2 : 2 * 5 * x = 6 * y) : 
  (∃ x y : ℕ, x + y = 32 ∧ 2 * 5 * x = 6 * y) :=
sorry

end workers_distribution_l1995_199572


namespace range_of_a_l1995_199578

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + 1 / 4 > 0) ↔ (Real.sqrt 5 - 3) / 2 < a ∧ a < (3 + Real.sqrt 5) / 2 :=
by
  sorry

end range_of_a_l1995_199578


namespace min_value_at_2_l1995_199562

noncomputable def f (x : ℝ) : ℝ := (2 / (x^2)) + Real.log x

theorem min_value_at_2 : (∀ x ∈ Set.Ioi (0 : ℝ), f x ≥ f 2) ∧ (∃ x ∈ Set.Ioi (0 : ℝ), f x = f 2) :=
by
  sorry

end min_value_at_2_l1995_199562


namespace minute_hand_travel_distance_l1995_199526

theorem minute_hand_travel_distance :
  ∀ (r : ℝ), r = 8 → (45 / 60) * (2 * Real.pi * r) = 12 * Real.pi :=
by
  intros r r_eq
  sorry

end minute_hand_travel_distance_l1995_199526


namespace meal_center_adults_l1995_199503

theorem meal_center_adults (cans : ℕ) (children_served : ℕ) (adults_served : ℕ) (total_children : ℕ) 
  (initial_cans : cans = 10) 
  (children_per_can : children_served = 7) 
  (adults_per_can : adults_served = 4) 
  (children_to_feed : total_children = 21) : 
  (cans - (total_children / children_served)) * adults_served = 28 := by
  have h1: 3 = total_children / children_served := by
    sorry
  have h2: 7 = cans - 3 := by
    sorry
  have h3: 28 = 7 * adults_served := by
    sorry
  have h4: adults_served = 4 := by
    sorry
  sorry

end meal_center_adults_l1995_199503


namespace find_n_l1995_199549

theorem find_n (a n : ℕ) 
  (h1 : a^2 % n = 8) 
  (h2 : a^3 % n = 25) 
  (h3 : n > 25) : 
  n = 113 := 
sorry

end find_n_l1995_199549


namespace average_percentage_difference_in_tail_sizes_l1995_199588

-- Definitions for the number of segments in each type of rattlesnake
def segments_eastern : ℕ := 6
def segments_western : ℕ := 8
def segments_southern : ℕ := 7
def segments_northern : ℕ := 9

-- Definition for percentage difference function
def percentage_difference (a : ℕ) (b : ℕ) : ℚ := ((b - a : ℚ) / b) * 100

-- Theorem statement to prove the average percentage difference
theorem average_percentage_difference_in_tail_sizes :
  (percentage_difference segments_eastern segments_western +
   percentage_difference segments_southern segments_western +
   percentage_difference segments_northern segments_western) / 3 = 16.67 := 
sorry

end average_percentage_difference_in_tail_sizes_l1995_199588


namespace number_of_customers_l1995_199590

theorem number_of_customers
  (nails_per_person : ℕ)
  (total_sounds : ℕ)
  (trimmed_nails_per_person : nails_per_person = 20)
  (produced_sounds : total_sounds = 100) :
  total_sounds / nails_per_person = 5 :=
by
  -- This is offered as a placeholder to indicate where a Lean proof goes.
  sorry

end number_of_customers_l1995_199590


namespace line_symmetric_about_y_eq_x_l1995_199570

-- Define the line equation types and the condition for symmetry
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Conditions given
variable (a b c : ℝ)
variable (h_ab_pos : a * b > 0)

-- Definition of the problem in Lean
theorem line_symmetric_about_y_eq_x (h_bisector : ∀ x y : ℝ, line_equation a b c x y ↔ line_equation b a c y x) : 
  ∀ x y : ℝ, line_equation b a c x y := by
  sorry

end line_symmetric_about_y_eq_x_l1995_199570


namespace James_pays_6_dollars_l1995_199500

-- Defining the conditions
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def friend_share : ℚ := 0.5

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack

-- Total cost calculation
def total_cost : ℚ := total_stickers * cost_per_sticker

-- James' payment calculation
def james_payment : ℚ := total_cost * friend_share

-- Theorem statement to be proven
theorem James_pays_6_dollars : james_payment = 6 := by
  sorry

end James_pays_6_dollars_l1995_199500


namespace diagonal_of_square_l1995_199575

theorem diagonal_of_square (s d : ℝ) (h_perimeter : 4 * s = 40) : d = 10 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_square_l1995_199575


namespace cubic_difference_l1995_199592

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : a^3 - b^3 = 448 :=
by
  sorry

end cubic_difference_l1995_199592


namespace derivative_of_x_ln_x_l1995_199580

noncomputable
def x_ln_x (x : ℝ) : ℝ := x * Real.log x

theorem derivative_of_x_ln_x (x : ℝ) (hx : x > 0) :
  deriv (x_ln_x) x = 1 + Real.log x :=
by
  -- Proof body, with necessary assumptions and justifications
  sorry

end derivative_of_x_ln_x_l1995_199580


namespace matrix_equation_l1995_199521

-- Definitions from conditions
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![ -1, 4], ![ -6, 3]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1  -- Identity matrix

-- Given calculation of N^2
def N_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![ -23, 8], ![ -12, -15]]

-- Goal: prove that N^2 = r*N + s*I for r = 2 and s = -21
theorem matrix_equation (r s : ℤ) (h_r : r = 2) (h_s : s = -21) : N_squared = r • N + s • I := by
  sorry

end matrix_equation_l1995_199521


namespace geometric_sequence_third_term_l1995_199548

theorem geometric_sequence_third_term (a : ℕ → ℕ) (x : ℕ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : a 3 = x) (h_geom : ∀ n, a (n + 1) = a n * r) :
  x = 9 := 
sorry

end geometric_sequence_third_term_l1995_199548


namespace ptarmigan_environmental_capacity_l1995_199593

theorem ptarmigan_environmental_capacity (predators_eradicated : Prop) (mass_deaths : Prop) : 
  (after_predator_eradication : predators_eradicated → mass_deaths) →
  (environmental_capacity_increased : Prop) → environmental_capacity_increased :=
by
  intros h1 h2
  sorry

end ptarmigan_environmental_capacity_l1995_199593


namespace percentage_decrease_l1995_199525

theorem percentage_decrease (original_price new_price : ℝ) (h₁ : original_price = 700) (h₂ : new_price = 532) : 
  ((original_price - new_price) / original_price) * 100 = 24 := by
  sorry

end percentage_decrease_l1995_199525


namespace find_a_b_transform_line_l1995_199537

theorem find_a_b_transform_line (a b : ℝ) (hA : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, a], ![b, 3]]) :
  (∀ x y : ℝ, (2 * (-(x) + a*y) - (b*x + 3*y) - 3 = 0) → (2*x - y - 3 = 0)) →
  a = 1 ∧ b = -4 :=
by {
  sorry
}

end find_a_b_transform_line_l1995_199537


namespace one_cow_one_bag_l1995_199561

def husk_eating (C B D : ℕ) : Prop :=
  C * D / B = D

theorem one_cow_one_bag (C B D n : ℕ) (h : husk_eating C B D) (hC : C = 46) (hB : B = 46) (hD : D = 46) : n = D :=
by
  rw [hC, hB, hD] at h
  sorry

end one_cow_one_bag_l1995_199561


namespace tom_speed_RB_l1995_199506

/-- Let d be the distance between B and C (in miles).
    Let 2d be the distance between R and B (in miles).
    Let v be Tom’s speed driving from R to B (in mph).
    Given conditions:
    1. Tom's speed from B to C = 20 mph.
    2. Total average speed of the whole journey = 36 mph.
    Prove that Tom's speed driving from R to B is 60 mph. -/
theorem tom_speed_RB
  (d : ℝ) (v : ℝ)
  (h1 : 20 ≠ 0)
  (h2 : 36 ≠ 0)
  (avg_speed : 3 * d / (2 * d / v + d / 20) = 36) :
  v = 60 := 
sorry

end tom_speed_RB_l1995_199506


namespace height_of_removed_player_l1995_199551

theorem height_of_removed_player (S : ℕ) (x : ℕ) (total_height_11 : S + x = 182 * 11)
  (average_height_10 : S = 181 * 10): x = 192 :=
by
  sorry

end height_of_removed_player_l1995_199551


namespace exists_points_with_small_distance_l1995_199582

theorem exists_points_with_small_distance :
  ∃ A B : ℝ × ℝ, (A.2 = A.1^4) ∧ (B.2 = B.1^4 + B.1^2 + B.1 + 1) ∧ 
  (dist A B < 1 / 100) :=
by
  sorry

end exists_points_with_small_distance_l1995_199582


namespace k_is_odd_l1995_199520

theorem k_is_odd (m n k : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_k : 0 < k) (h : 3 * m * k = (m + 3)^n + 1) : Odd k :=
by {
  sorry
}

end k_is_odd_l1995_199520


namespace units_digit_17_pow_2007_l1995_199597

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l1995_199597


namespace two_mul_seven_pow_n_plus_one_divisible_by_three_l1995_199528

-- Definition of natural numbers
variable (n : ℕ)

-- Statement of the problem in Lean
theorem two_mul_seven_pow_n_plus_one_divisible_by_three (n : ℕ) : 3 ∣ (2 * 7^n + 1) := 
sorry

end two_mul_seven_pow_n_plus_one_divisible_by_three_l1995_199528


namespace total_earnings_correct_l1995_199591

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end total_earnings_correct_l1995_199591


namespace intersection_range_l1995_199581

theorem intersection_range (k : ℝ) :
  (∃ x y : ℝ, y = k * x + k + 2 ∧ y = -2 * x + 4 ∧ x > 0 ∧ y > 0) ↔ -2/3 < k ∧ k < 2 :=
by
  sorry

end intersection_range_l1995_199581


namespace initial_percentage_increase_l1995_199516

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end initial_percentage_increase_l1995_199516


namespace smallest_value_not_defined_l1995_199553

noncomputable def smallest_undefined_x : ℝ :=
  let a := 6
  let b := -37
  let c := 5
  let discriminant := b * b - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 < x2 then x1 else x2

theorem smallest_value_not_defined :
  smallest_undefined_x = 0.1383 :=
by sorry

end smallest_value_not_defined_l1995_199553


namespace describes_random_event_proof_l1995_199532

def describes_random_event (phrase : String) : Prop :=
  match phrase with
  | "Winter turns into spring"  => False
  | "Fishing for the moon in the water" => False
  | "Seeking fish on a tree" => False
  | "Meeting unexpectedly" => True
  | _ => False

theorem describes_random_event_proof : describes_random_event "Meeting unexpectedly" = True :=
by
  sorry

end describes_random_event_proof_l1995_199532


namespace mary_total_money_l1995_199595

def num_quarters : ℕ := 21
def quarters_worth : ℚ := 0.25
def dimes_worth : ℚ := 0.10

def num_dimes (Q : ℕ) : ℕ := (Q - 7) / 2

def total_money (Q : ℕ) (D : ℕ) : ℚ :=
  Q * quarters_worth + D * dimes_worth

theorem mary_total_money : 
  total_money num_quarters (num_dimes num_quarters) = 5.95 := 
by
  sorry

end mary_total_money_l1995_199595


namespace find_t_l1995_199587

theorem find_t (s t : ℝ) (h1 : 12 * s + 7 * t = 165) (h2 : s = t + 3) : t = 6.789 := 
by 
  sorry

end find_t_l1995_199587


namespace symmetric_line_equation_y_axis_l1995_199571

theorem symmetric_line_equation_y_axis (x y : ℝ) : 
  (∃ m n : ℝ, (y = 3 * x + 1) ∧ (x + m = 0) ∧ (y = n) ∧ (n = 3 * m + 1)) → 
  y = -3 * x + 1 :=
by
  sorry

end symmetric_line_equation_y_axis_l1995_199571


namespace arccos_equivalence_l1995_199517

open Real

theorem arccos_equivalence (α : ℝ) (h₀ : α ∈ Set.Icc 0 (2 * π)) (h₁ : cos α = 1 / 3) :
  α = arccos (1 / 3) ∨ α = 2 * π - arccos (1 / 3) := 
by 
  sorry

end arccos_equivalence_l1995_199517


namespace urn_problem_l1995_199531

noncomputable def count_balls (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) : ℕ :=
initial_white + initial_black + operations

noncomputable def urn_probability (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) (final_white : ℕ) (final_black : ℕ) : ℚ :=
if final_white + final_black = count_balls initial_white initial_black operations &&
   final_white = (initial_white + (operations - (final_black - initial_black))) &&
   (final_white + final_black) = 8 then 3 / 5 else 0

theorem urn_problem :
  let initial_white := 2
  let initial_black := 1
  let operations := 4
  let final_white := 4
  let final_black := 4
  count_balls initial_white initial_black operations = 8 ∧ urn_probability initial_white initial_black operations final_white final_black = 3 / 5 :=
by
  sorry

end urn_problem_l1995_199531


namespace mike_salary_calculation_l1995_199557

theorem mike_salary_calculation
  (F : ℝ) (M : ℝ) (new_M : ℝ) (x : ℝ)
  (F_eq : F = 1000)
  (M_eq : M = x * F)
  (increase_eq : new_M = 1.40 * M)
  (new_M_val : new_M = 15400) :
  M = 11000 ∧ x = 11 :=
by
  sorry

end mike_salary_calculation_l1995_199557


namespace not_in_second_column_l1995_199534

theorem not_in_second_column : ¬∃ (n : ℕ), (1 ≤ n ∧ n ≤ 400) ∧ 3 * n + 1 = 131 :=
by sorry

end not_in_second_column_l1995_199534


namespace maximum_value_P_l1995_199514

open Classical

noncomputable def P (a b c d : ℝ) : ℝ := a * b + b * c + c * d + d * a

theorem maximum_value_P : ∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 40 → P a b c d ≤ 800 :=
by
  sorry

end maximum_value_P_l1995_199514


namespace upper_limit_of_prime_range_l1995_199538

theorem upper_limit_of_prime_range : 
  ∃ x : ℝ, (26 / 3 < 11) ∧ (11 < x) ∧ (x < 17) :=
by
  sorry

end upper_limit_of_prime_range_l1995_199538


namespace odot_computation_l1995_199501

noncomputable def op (a b : ℚ) : ℚ := 
  (a + b) / (1 + a * b)

theorem odot_computation : op 2 (op 3 (op 4 5)) = 7 / 8 := 
  by 
  sorry

end odot_computation_l1995_199501


namespace original_sum_of_money_l1995_199565

theorem original_sum_of_money (P R : ℝ) 
  (h1 : 720 = P + (P * R * 2) / 100) 
  (h2 : 1020 = P + (P * R * 7) / 100) : 
  P = 600 := 
by sorry

end original_sum_of_money_l1995_199565


namespace area_of_triangle_ABC_l1995_199584

-- Axiom statements representing the conditions
axiom medians_perpendicular (A B C D E G : Type) : Prop
axiom median_ad_length (A D : Type) : Prop
axiom median_be_length (B E : Type) : Prop

-- Main theorem statement
theorem area_of_triangle_ABC
  (A B C D E G : Type)
  (h1 : medians_perpendicular A B C D E G)
  (h2 : median_ad_length A D) -- AD = 18
  (h3 : median_be_length B E) -- BE = 24
  : ∃ (area : ℝ), area = 576 :=
sorry

end area_of_triangle_ABC_l1995_199584
