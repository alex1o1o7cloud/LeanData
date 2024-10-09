import Mathlib

namespace Sues_necklace_total_beads_l1419_141952

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l1419_141952


namespace books_before_grant_correct_l1419_141991

-- Definitions based on the given conditions
def books_purchased : ℕ := 2647
def total_books_now : ℕ := 8582

-- Definition and the proof statement
def books_before_grant : ℕ := 5935

-- Proof statement: The number of books before the grant plus the books purchased equals the total books now
theorem books_before_grant_correct :
  books_before_grant + books_purchased = total_books_now :=
by
  -- Predictably, no need to complete proof, 'sorry' is used.
  sorry

end books_before_grant_correct_l1419_141991


namespace product_of_roots_l1419_141940

-- Define the coefficients of the cubic equation
def a : ℝ := 2
def d : ℝ := 12

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := a * x^3 - 3 * x^2 - 8 * x + d

-- Prove the product of the roots is -6 using Vieta's formulas
theorem product_of_roots : -d / a = -6 := by
  sorry

end product_of_roots_l1419_141940


namespace sum_of_all_digits_divisible_by_nine_l1419_141910

theorem sum_of_all_digits_divisible_by_nine :
  ∀ (A B C D : ℕ),
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
  sorry

end sum_of_all_digits_divisible_by_nine_l1419_141910


namespace man_walking_rate_is_12_l1419_141967

theorem man_walking_rate_is_12 (M : ℝ) (woman_speed : ℝ) (time_waiting : ℝ) (catch_up_time : ℝ) 
  (woman_speed_eq : woman_speed = 12) (time_waiting_eq : time_waiting = 1 / 6) 
  (catch_up_time_eq : catch_up_time = 1 / 6): 
  (M * catch_up_time = woman_speed * time_waiting) → M = 12 := by
  intro h
  rw [woman_speed_eq, time_waiting_eq, catch_up_time_eq] at h
  sorry

end man_walking_rate_is_12_l1419_141967


namespace cost_price_of_article_l1419_141994

theorem cost_price_of_article :
  ∃ (CP : ℝ), (616 = 1.10 * (1.17 * CP)) → CP = 478.77 :=
by
  sorry

end cost_price_of_article_l1419_141994


namespace number_of_students_l1419_141908

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N) (h2 : (T - 250) / (N - 5) = 90) : N = 20 :=
sorry

end number_of_students_l1419_141908


namespace sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l1419_141943

theorem sin_and_tan_alpha_in_second_quadrant 
  (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (hcos : Real.cos α = -8 / 17) :
  Real.sin α = 15 / 17 ∧ Real.tan α = -15 / 8 := 
  sorry

theorem expression_value_for_given_tan 
  (α : ℝ) (htan : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := 
  sorry

end sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l1419_141943


namespace ukuleles_and_violins_l1419_141936

theorem ukuleles_and_violins (U V : ℕ) : 
  (4 * U + 6 * 4 + 4 * V = 40) → (U + V = 4) :=
by
  intro h
  sorry

end ukuleles_and_violins_l1419_141936


namespace identify_triangle_centers_l1419_141955

variable (P : Fin 7 → Type)
variable (I O H L G N K : Type)
variable (P1 P2 P3 P4 P5 P6 P7 : Type)
variable (cond : (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H))

theorem identify_triangle_centers :
  (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H) :=
by sorry

end identify_triangle_centers_l1419_141955


namespace findYears_l1419_141972

def totalInterest (n : ℕ) : ℕ :=
  24 * n + 70 * n

theorem findYears (n : ℕ) : totalInterest n = 350 → n = 4 := 
sorry

end findYears_l1419_141972


namespace function_properties_l1419_141920

-- Define the function and conditions
def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f (10 + x) = f (10 - x)
axiom condition2 (x : ℝ) : f (20 - x) = -f (20 + x)

-- Lean statement to encapsulate the question and expected result
theorem function_properties (x : ℝ) : (f (-x) = -f x) ∧ (f (x + 40) = f x) :=
sorry

end function_properties_l1419_141920


namespace parallel_vectors_implies_x_l1419_141909

-- a definition of the vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

-- a definition for vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- a definition for scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- a definition for vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- the theorem statement
theorem parallel_vectors_implies_x (x : ℝ) (h : 
  vector_add vector_a (vector_b x) = ⟨3, 1 + x⟩ ∧
  vector_sub (scalar_mul 2 vector_a) (vector_b x) = ⟨3, 2 - x⟩ ∧
  ∃ k : ℝ, vector_add vector_a (vector_b x) = scalar_mul k (vector_sub (scalar_mul 2 vector_a) (vector_b x))
  ) : x = 1 / 2 :=
sorry

end parallel_vectors_implies_x_l1419_141909


namespace loss_percentage_is_nine_percent_l1419_141986

theorem loss_percentage_is_nine_percent
    (C S : ℝ)
    (h1 : 15 * C = 20 * S)
    (discount_rate : ℝ := 0.10)
    (tax_rate : ℝ := 0.08) :
    (((0.9 * C) - (1.08 * S)) / C) * 100 = 9 :=
by
  sorry

end loss_percentage_is_nine_percent_l1419_141986


namespace apples_distribution_count_l1419_141957

theorem apples_distribution_count : 
  ∃ (count : ℕ), count = 249 ∧ 
  (∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ a ≤ 20) →
  (a' + 3 + b' + 3 + c' + 3 = 30 ∧ a' + b' + c' = 21) → 
  (∃ (a' b' c' : ℕ), a' + b' + c' = 21 ∧ a' ≤ 17) :=
by
  sorry

end apples_distribution_count_l1419_141957


namespace Watson_class_student_count_l1419_141918

def num_kindergartners : ℕ := 14
def num_first_graders : ℕ := 24
def num_second_graders : ℕ := 4

def total_students : ℕ := num_kindergartners + num_first_graders + num_second_graders

theorem Watson_class_student_count : total_students = 42 := 
by
    sorry

end Watson_class_student_count_l1419_141918


namespace special_number_is_square_l1419_141930

-- Define the special number format
def special_number (n : ℕ) : ℕ :=
  3 * (10^n - 1)/9 + 4

theorem special_number_is_square (n : ℕ) :
  ∃ k : ℕ, k * k = special_number n := by
  sorry

end special_number_is_square_l1419_141930


namespace initial_fliers_l1419_141911

variable (F : ℕ) -- Initial number of fliers

-- Conditions
axiom morning_send : F - (1 / 5) * F = (4 / 5) * F
axiom afternoon_send : (4 / 5) * F - (1 / 4) * ((4 / 5) * F) = (3 / 5) * F
axiom final_count : (3 / 5) * F = 600

theorem initial_fliers : F = 1000 := by
  sorry

end initial_fliers_l1419_141911


namespace compound_interest_amount_l1419_141905

theorem compound_interest_amount:
  let SI := (5250 * 4 * 2) / 100
  let CI := 2 * SI
  let P := 420 / 0.21 
  CI = P * ((1 + 0.1) ^ 2 - 1) →
  SI = 210 →
  CI = 420 →
  P = 2000 :=
by
  sorry

end compound_interest_amount_l1419_141905


namespace band_section_student_count_l1419_141925

theorem band_section_student_count :
  (0.5 * 500) + (0.12 * 500) + (0.23 * 500) + (0.08 * 500) = 465 :=
by 
  sorry

end band_section_student_count_l1419_141925


namespace oliver_first_coupon_redeem_on_friday_l1419_141935

-- Definitions of conditions in the problem
def has_coupons (n : ℕ) := n = 8
def uses_coupon_every_9_days (days : ℕ) := days = 9
def is_closed_on_monday (day : ℕ) := day % 7 = 1  -- Assuming 1 represents Monday
def does_not_redeem_on_closed_day (redemption_days : List ℕ) :=
  ∀ day ∈ redemption_days, day % 7 ≠ 1

-- Main theorem statement
theorem oliver_first_coupon_redeem_on_friday : 
  ∃ (first_redeem_day: ℕ), 
  has_coupons 8 ∧ uses_coupon_every_9_days 9 ∧
  is_closed_on_monday 1 ∧ 
  does_not_redeem_on_closed_day [first_redeem_day, first_redeem_day + 9, first_redeem_day + 18, first_redeem_day + 27, first_redeem_day + 36, first_redeem_day + 45, first_redeem_day + 54, first_redeem_day + 63] ∧ 
  first_redeem_day % 7 = 5 := sorry

end oliver_first_coupon_redeem_on_friday_l1419_141935


namespace different_sets_l1419_141976

theorem different_sets (a b c : ℤ) (h1 : 0 < a) (h2 : a < c - 1) (h3 : 1 < b) (h4 : b < c)
  (rk : ∀ (k : ℤ), 0 ≤ k ∧ k ≤ a → ∃ (r : ℤ), 0 ≤ r ∧ r < c ∧ k * b % c = r) :
  {r | ∃ k, 0 ≤ k ∧ k ≤ a ∧ r = k * b % c} ≠ {k | 0 ≤ k ∧ k ≤ a} :=
sorry

end different_sets_l1419_141976


namespace measure_of_one_exterior_angle_l1419_141960

theorem measure_of_one_exterior_angle (n : ℕ) (h : n > 2) : 
  n > 2 → ∃ (angle : ℝ), angle = 360 / n :=
by 
  sorry

end measure_of_one_exterior_angle_l1419_141960


namespace area_of_annulus_l1419_141928

variables (R r x : ℝ) (hRr : R > r) (h : R^2 - r^2 = x^2)

theorem area_of_annulus : π * R^2 - π * r^2 = π * x^2 :=
by
  sorry

end area_of_annulus_l1419_141928


namespace math_problem_l1419_141953

theorem math_problem (a b c d e : ℤ) (x : ℤ) (hx : x > 196)
  (h1 : a + b = 183) (h2 : a + c = 186) (h3 : d + e = x) (h4 : c + e = 196)
  (h5 : 183 < 186) (h6 : 186 < 187) (h7 : 187 < 190) (h8 : 190 < 191) (h9 : 191 < 192)
  (h10 : 192 < 193) (h11 : 193 < 194) (h12 : 194 < 196) (h13 : 196 < x) :
  (a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200) ∧ (∃ y, y = 10 * x + 3 ∧ y = 2003) :=
by
  sorry

end math_problem_l1419_141953


namespace cornelia_travel_countries_l1419_141906

theorem cornelia_travel_countries (europe south_america asia half_remaining : ℕ) 
  (h1 : europe = 20)
  (h2 : south_america = 10)
  (h3 : asia = 6)
  (h4 : asia = half_remaining / 2) : 
  europe + south_america + half_remaining = 42 :=
by
  sorry

end cornelia_travel_countries_l1419_141906


namespace average_paper_tape_length_l1419_141996

-- Define the lengths of the paper tapes as given in the conditions
def red_tape_length : ℝ := 20
def purple_tape_length : ℝ := 16

-- State the proof problem
theorem average_paper_tape_length : 
  (red_tape_length + purple_tape_length) / 2 = 18 := 
by
  sorry

end average_paper_tape_length_l1419_141996


namespace describe_difference_of_squares_l1419_141958

def description_of_a_squared_minus_b_squared : Prop :=
  ∃ (a b : ℝ), (a^2 - b^2) = (a^2 - b^2)

theorem describe_difference_of_squares :
  description_of_a_squared_minus_b_squared :=
by sorry

end describe_difference_of_squares_l1419_141958


namespace total_kayaks_built_by_April_l1419_141975

theorem total_kayaks_built_by_April
    (a : Nat := 9) (r : Nat := 3) (n : Nat := 4) :
    let S := a * (r ^ n - 1) / (r - 1)
    S = 360 := by
  sorry

end total_kayaks_built_by_April_l1419_141975


namespace meaning_of_a2_add_b2_ne_zero_l1419_141912

theorem meaning_of_a2_add_b2_ne_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end meaning_of_a2_add_b2_ne_zero_l1419_141912


namespace tricycle_count_l1419_141945

theorem tricycle_count
    (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ)
    (h1 : total_children - walking_children = 8)
    (h2 : 2 * (total_children - walking_children - (total_wheels - 16) / 3) + 3 * ((total_wheels - 16) / 3) = total_wheels) :
    (total_wheels - 16) / 3 = 8 :=
by
    intros
    sorry

end tricycle_count_l1419_141945


namespace total_ladybugs_and_ants_l1419_141962

def num_leaves : ℕ := 84
def ladybugs_per_leaf : ℕ := 139
def ants_per_leaf : ℕ := 97

def total_ladybugs := ladybugs_per_leaf * num_leaves
def total_ants := ants_per_leaf * num_leaves
def total_insects := total_ladybugs + total_ants

theorem total_ladybugs_and_ants : total_insects = 19824 := by
  sorry

end total_ladybugs_and_ants_l1419_141962


namespace triangle_centroid_property_l1419_141913

def distance_sq (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem triangle_centroid_property
  (A B C P : ℝ × ℝ)
  (G : ℝ × ℝ)
  (hG : G = ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )) :
  distance_sq A P + distance_sq B P + distance_sq C P = 
  distance_sq A G + distance_sq B G + distance_sq C G + 3 * distance_sq G P :=
by
  sorry

end triangle_centroid_property_l1419_141913


namespace percent_gain_is_5_333_l1419_141938

noncomputable def calculate_percent_gain (total_sheep : ℕ) 
                                         (sold_sheep : ℕ) 
                                         (price_paid_sheep : ℕ) 
                                         (sold_remaining_sheep : ℕ)
                                         (remaining_sheep : ℕ) 
                                         (total_cost : ℝ) 
                                         (initial_revenue : ℝ) 
                                         (remaining_revenue : ℝ) : ℝ :=
  (remaining_revenue + initial_revenue - total_cost) / total_cost * 100

theorem percent_gain_is_5_333
  (x : ℝ)
  (total_sheep : ℕ := 800)
  (sold_sheep : ℕ := 750)
  (price_paid_sheep : ℕ := 790)
  (remaining_sheep : ℕ := 50)
  (total_cost : ℝ := (800 : ℝ) * x)
  (initial_revenue : ℝ := (790 : ℝ) * x)
  (remaining_revenue : ℝ := (50 : ℝ) * ((790 : ℝ) * x / 750)) :
  calculate_percent_gain total_sheep sold_sheep price_paid_sheep remaining_sheep 50 total_cost initial_revenue remaining_revenue = 5.333 := by
  sorry

end percent_gain_is_5_333_l1419_141938


namespace al_original_amount_l1419_141926

theorem al_original_amount : 
  ∃ (a b c : ℝ), 
    a + b + c = 1200 ∧ 
    (a - 200 + 3 * b + 4 * c) = 1800 ∧ 
    b = 2800 - 3 * a ∧ 
    c = 1200 - a - b ∧ 
    a = 860 := by
  sorry

end al_original_amount_l1419_141926


namespace proof_l1419_141965

-- Define the expression
def expr : ℕ :=
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128)

-- Define the conjectured result
def result : ℕ := 5^128 - 4^128

-- Assert their equality
theorem proof : expr = result :=
by
    sorry

end proof_l1419_141965


namespace shopping_center_expense_l1419_141941

theorem shopping_center_expense
    (films_count : ℕ := 9)
    (films_original_price : ℝ := 7)
    (film_discount : ℝ := 2)
    (books_full_price : ℝ := 10)
    (books_count : ℕ := 5)
    (books_discount_rate : ℝ := 0.25)
    (cd_price : ℝ := 4.50)
    (cd_count : ℕ := 6)
    (tax_rate : ℝ := 0.06)
    (total_amount_spent : ℝ := 109.18) :
    let films_total := films_count * (films_original_price - film_discount)
    let remaining_books := books_count - 1
    let discounted_books_total := remaining_books * (books_full_price * (1 - books_discount_rate))
    let books_total := books_full_price + discounted_books_total
    let cds_paid_count := cd_count - (cd_count / 3)
    let cds_total := cds_paid_count * cd_price
    let total_before_tax := films_total + books_total + cds_total
    let tax := total_before_tax * tax_rate
    let total_with_tax := total_before_tax + tax
    total_with_tax = total_amount_spent :=
by
  sorry

end shopping_center_expense_l1419_141941


namespace units_digit_G1000_l1419_141924

def Gn (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G1000 : (Gn 1000) % 10 = 2 :=
by sorry

end units_digit_G1000_l1419_141924


namespace max_sin_x_value_l1419_141973

theorem max_sin_x_value (x y z : ℝ) (h1 : Real.sin x = Real.cos y) (h2 : Real.sin y = Real.cos z) (h3 : Real.sin z = Real.cos x) : Real.sin x ≤ Real.sqrt 2 / 2 :=
by
  sorry

end max_sin_x_value_l1419_141973


namespace license_plate_count_l1419_141923

theorem license_plate_count : 
  let vowels := 5
  let consonants := 21
  let digits := 10
  21 * 21 * 5 * 5 * 10 = 110250 := 
by 
  sorry

end license_plate_count_l1419_141923


namespace peter_read_more_books_l1419_141999

/-
Given conditions:
  Peter has 20 books.
  Peter has read 40% of them.
  Peter's brother has read 10% of them.
We aim to prove that Peter has read 6 more books than his brother.
-/

def total_books : ℕ := 20
def peter_read_fraction : ℚ := 0.4
def brother_read_fraction : ℚ := 0.1

def books_read_by_peter := total_books * peter_read_fraction
def books_read_by_brother := total_books * brother_read_fraction

theorem peter_read_more_books :
  books_read_by_peter - books_read_by_brother = 6 := by
  sorry

end peter_read_more_books_l1419_141999


namespace find_K_l1419_141914

theorem find_K (Z K : ℕ)
  (hZ1 : 700 < Z)
  (hZ2 : Z < 1500)
  (hK : K > 1)
  (hZ_eq : Z = K^4)
  (hZ_perfect : ∃ n : ℕ, Z = n^6) :
  K = 3 :=
by
  sorry

end find_K_l1419_141914


namespace total_bottles_l1419_141934

theorem total_bottles (n : ℕ) (h1 : ∃ one_third two_third: ℕ, one_third = n / 3 ∧ two_third = 2 * (n / 3) ∧ 3 * one_third = n)
    (h2 : 25 ≤ n)
    (h3 : ∃ damage1 damage2 damage_diff : ℕ, damage1 = 25 * 160 ∧ damage2 = (n / 3) * 160 + ((2 * (n / 3) - 25) * 130) ∧ damage1 - damage2 = 660) :
    n = 36 :=
by
  sorry

end total_bottles_l1419_141934


namespace hockey_players_l1419_141979

theorem hockey_players (n : ℕ) (h1 : n < 30) (h2 : n % 2 = 0) (h3 : n % 4 = 0) (h4 : n % 7 = 0) :
  (n / 4 = 7) :=
by
  sorry

end hockey_players_l1419_141979


namespace fewest_posts_required_l1419_141901

def dimensions_garden : ℕ × ℕ := (32, 72)
def post_spacing : ℕ := 8

theorem fewest_posts_required
  (d : ℕ × ℕ := dimensions_garden)
  (s : ℕ := post_spacing) :
  d = (32, 72) ∧ s = 8 → 
  ∃ N, N = 26 := 
by 
  sorry

end fewest_posts_required_l1419_141901


namespace prime_sum_remainder_l1419_141990

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l1419_141990


namespace angle_A_in_quadrilateral_l1419_141970

noncomputable def degree_measure_A (A B C D : ℝ) := A

theorem angle_A_in_quadrilateral 
  (A B C D : ℝ)
  (hA : A = 3 * B)
  (hC : A = 4 * C)
  (hD : A = 6 * D)
  (sum_angles : A + B + C + D = 360) :
  degree_measure_A A B C D = 206 :=
by
  sorry

end angle_A_in_quadrilateral_l1419_141970


namespace markup_percentage_l1419_141942

-- Define the purchase price and the gross profit
def purchase_price : ℝ := 54
def gross_profit : ℝ := 18

-- Define the sale price after discount
def sale_discount : ℝ := 0.8

-- Given that the sale price after the discount is purchase_price + gross_profit
theorem markup_percentage (M : ℝ) (SP : ℝ) : 
  SP = purchase_price * (1 + M / 100) → -- selling price as function of markup
  (SP * sale_discount = purchase_price + gross_profit) → -- sale price after 20% discount
  M = 66.67 := 
by
  -- sorry to skip the proof
  sorry

end markup_percentage_l1419_141942


namespace total_cost_for_photos_l1419_141921

def total_cost (n : ℕ) (f : ℝ) (c : ℝ) : ℝ :=
  f + (n - 4) * c

theorem total_cost_for_photos :
  total_cost 54 24.5 2.3 = 139.5 :=
by
  sorry

end total_cost_for_photos_l1419_141921


namespace solution_set_of_x_sq_gt_x_l1419_141950

theorem solution_set_of_x_sq_gt_x :
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} := 
sorry

end solution_set_of_x_sq_gt_x_l1419_141950


namespace find_f_zero_l1419_141933

theorem find_f_zero (f : ℝ → ℝ) (h : ∀ x, f ((x + 1) / (x - 1)) = x^2 + 3) : f 0 = 4 :=
by
  -- The proof goes here.
  sorry

end find_f_zero_l1419_141933


namespace find_n_l1419_141998

noncomputable def arithmeticSequenceTerm (a b : ℝ) (n : ℕ) : ℝ :=
  let A := Real.log a
  let B := Real.log b
  6 * B + (n - 1) * 11 * B

theorem find_n 
  (a b : ℝ) 
  (h1 : Real.log (a^2 * b^4) = 2 * Real.log a + 4 * Real.log b)
  (h2 : Real.log (a^6 * b^11) = 6 * Real.log a + 11 * Real.log b)
  (h3 : Real.log (a^12 * b^20) = 12 * Real.log a + 20 * Real.log b) 
  (h_diff : (6 * Real.log a + 11 * Real.log b) - (2 * Real.log a + 4 * Real.log b) = 
            (12 * Real.log a + 20 * Real.log b) - (6 * Real.log a + 11 * Real.log b))
  : ∃ n : ℕ, arithmeticSequenceTerm a b 15 = Real.log (b^n) ∧ n = 160 :=
by
  use 160
  sorry

end find_n_l1419_141998


namespace opposite_of_neg_2023_l1419_141903

theorem opposite_of_neg_2023 :
  ∃ y : ℝ, (-2023 + y = 0) ∧ y = 2023 :=
by
  sorry

end opposite_of_neg_2023_l1419_141903


namespace find_x2_x1_add_x3_l1419_141988

-- Definition of the polynomial
def polynomial (x : ℝ) : ℝ := (10*x^3 - 210*x^2 + 3)

-- Statement including conditions and the question we need to prove
theorem find_x2_x1_add_x3 :
  ∃ x₁ x₂ x₃ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ 
    polynomial x₁ = 0 ∧ 
    polynomial x₂ = 0 ∧ 
    polynomial x₃ = 0 ∧ 
    x₂ * (x₁ + x₃) = 21 :=
by sorry

end find_x2_x1_add_x3_l1419_141988


namespace range_of_x_l1419_141954

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the relevant conditions
axiom decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) < 0
axiom symmetry : ∀ x : ℝ, f (1 - x) = -f (1 + x)
axiom f_one : f 1 = -1

-- Define the statement to be proved
theorem range_of_x : ∀ x : ℝ, -1 ≤ f (0.5 * x - 1) ∧ f (0.5 * x - 1) ≤ 1 → 0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l1419_141954


namespace museum_wings_paintings_l1419_141963

theorem museum_wings_paintings (P A : ℕ) (h1: P + A = 8) (h2: P = 1 + 2) : P = 3 :=
by
  -- Proof here
  sorry

end museum_wings_paintings_l1419_141963


namespace number_of_rectangular_arrays_of_chairs_l1419_141984

/-- 
Given a classroom that contains 45 chairs, prove that 
the number of rectangular arrays of chairs that can be made such that 
each row contains at least 3 chairs and each column contains at least 3 chairs is 4.
-/
theorem number_of_rectangular_arrays_of_chairs : 
  ∃ (n : ℕ), n = 4 ∧ 
    ∀ (a b : ℕ), (a * b = 45) → 
      (a ≥ 3) → (b ≥ 3) → 
      (n = 4) := 
sorry

end number_of_rectangular_arrays_of_chairs_l1419_141984


namespace solve_for_m_l1419_141974

theorem solve_for_m (x y m : ℝ) 
  (h1 : 2 * x + y = 3 * m) 
  (h2 : x - 4 * y = -2 * m)
  (h3 : y + 2 * m = 1 + x) :
  m = 3 / 5 := 
by 
  sorry

end solve_for_m_l1419_141974


namespace yellow_tiled_area_is_correct_l1419_141922

noncomputable def length : ℝ := 3.6
noncomputable def width : ℝ := 2.5 * length
noncomputable def total_area : ℝ := length * width
noncomputable def yellow_tiled_area : ℝ := total_area / 2

theorem yellow_tiled_area_is_correct (length_eq : length = 3.6)
    (width_eq : width = 2.5 * length)
    (total_area_eq : total_area = length * width)
    (yellow_area_eq : yellow_tiled_area = total_area / 2) :
    yellow_tiled_area = 16.2 := 
by sorry

end yellow_tiled_area_is_correct_l1419_141922


namespace vector_addition_simplification_l1419_141916

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_simplification
  (AB BC AC DC CD : V)
  (h1 : AB + BC = AC)
  (h2 : - DC = CD) :
  AB + BC - AC - DC = CD :=
by
  -- Placeholder for the proof
  sorry

end vector_addition_simplification_l1419_141916


namespace c_sq_minus_a_sq_divisible_by_48_l1419_141978

theorem c_sq_minus_a_sq_divisible_by_48
  (a b c : ℤ) (h_ac : a < c) (h_eq : a^2 + c^2 = 2 * b^2) : 48 ∣ (c^2 - a^2) := 
  sorry

end c_sq_minus_a_sq_divisible_by_48_l1419_141978


namespace correct_relation_l1419_141959

def satisfies_relation : Prop :=
  (∀ x y, (x = 0 ∧ y = 200) ∨ (x = 1 ∧ y = 170) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 50) ∨ (x = 4 ∧ y = 0) →
  y = 200 - 10 * x - 10 * x^2) 

theorem correct_relation : satisfies_relation :=
sorry

end correct_relation_l1419_141959


namespace sale_decrease_by_20_percent_l1419_141956

theorem sale_decrease_by_20_percent (P Q : ℝ)
  (h1 : P > 0) (h2 : Q > 0)
  (price_increased : ∀ P', P' = 1.30 * P)
  (revenue_increase : ∀ R, R = P * Q → ∀ R', R' = 1.04 * R)
  (new_revenue : ∀ P' Q' R', P' = 1.30 * P → Q' = Q * (1 - x / 100) → R' = P' * Q' → R' = 1.04 * (P * Q)) :
  1 - (20 / 100) = 0.8 :=
by sorry

end sale_decrease_by_20_percent_l1419_141956


namespace ben_time_to_school_l1419_141900

/-- Amy's steps per minute -/
def amy_steps_per_minute : ℕ := 80

/-- Length of each of Amy's steps in cm -/
def amy_step_length : ℕ := 70

/-- Time taken by Amy to reach school in minutes -/
def amy_time_to_school : ℕ := 20

/-- Ben's steps per minute -/
def ben_steps_per_minute : ℕ := 120

/-- Length of each of Ben's steps in cm -/
def ben_step_length : ℕ := 50

/-- Given the above conditions, we aim to prove that Ben takes 18 2/3 minutes to reach school. -/
theorem ben_time_to_school : (112000 / 6000 : ℚ) = 18 + 2 / 3 := 
by sorry

end ben_time_to_school_l1419_141900


namespace train_speed_approx_kmph_l1419_141992

noncomputable def length_of_train : ℝ := 150
noncomputable def time_to_cross_pole : ℝ := 4.425875438161669

theorem train_speed_approx_kmph :
  (length_of_train / time_to_cross_pole) * 3.6 = 122.03 :=
by sorry

end train_speed_approx_kmph_l1419_141992


namespace range_of_m_l1419_141977

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 > 0
def B (x : ℝ) (m : ℝ) : Prop := 2 * m - 1 ≤ x ∧ x ≤ m + 3
def subset (B A : ℝ → Prop) : Prop := ∀ x, B x → A x

theorem range_of_m (m : ℝ) : (∀ x, B x m → A x) ↔ (m < -4 ∨ m > 2) :=
by 
  sorry

end range_of_m_l1419_141977


namespace length_of_train_l1419_141961

-- Definitions based on the conditions in the problem
def time_to_cross_signal_pole : ℝ := 18
def time_to_cross_platform : ℝ := 54
def length_of_platform : ℝ := 600.0000000000001

-- Prove that the length of the train is 300.00000000000005 meters
theorem length_of_train
    (L V : ℝ)
    (h1 : L = V * time_to_cross_signal_pole)
    (h2 : L + length_of_platform = V * time_to_cross_platform) :
    L = 300.00000000000005 :=
by
  sorry

end length_of_train_l1419_141961


namespace ab_value_l1419_141981

theorem ab_value (a b : ℝ) (h1 : a + b = 7) (h2 : a^3 + b^3 = 91) : a * b = 12 :=
by
  sorry

end ab_value_l1419_141981


namespace xy_relationship_l1419_141907

theorem xy_relationship (x y : ℝ) (h : y = 2 * x - 1 - Real.sqrt (y^2 - 2 * x * y + 3 * x - 2)) :
  (x ≠ 1 → y = 2 * x - 1.5) ∧ (x = 1 → y ≤ 1) :=
by
  sorry

end xy_relationship_l1419_141907


namespace isosceles_triangle_base_length_l1419_141919

theorem isosceles_triangle_base_length :
  ∀ (p_equilateral p_isosceles side_equilateral : ℕ), 
  p_equilateral = 60 → 
  side_equilateral = p_equilateral / 3 →
  p_isosceles = 55 →
  ∀ (base_isosceles : ℕ),
  side_equilateral + side_equilateral + base_isosceles = p_isosceles →
  base_isosceles = 15 :=
by
  intros p_equilateral p_isosceles side_equilateral h1 h2 h3 base_isosceles h4
  sorry

end isosceles_triangle_base_length_l1419_141919


namespace total_elephants_l1419_141929

-- Define the conditions in Lean
def G (W : ℕ) : ℕ := 3 * W
def N (G : ℕ) : ℕ := 5 * G
def W : ℕ := 70

-- Define the statement to prove
theorem total_elephants :
  G W + W + N (G W) = 1330 :=
by
  -- Proof to be filled in
  sorry

end total_elephants_l1419_141929


namespace symmetric_linear_functions_l1419_141995

theorem symmetric_linear_functions :
  (∃ (a b : ℝ), ∀ x y : ℝ, (y = a * x + 2 ∧ y = 3 * x - b) → a = 1 / 3 ∧ b = 6) :=
by
  sorry

end symmetric_linear_functions_l1419_141995


namespace false_p_and_q_l1419_141937

variable {a : ℝ} 

def p (a : ℝ) := 3 * a / 2 ≤ 1
def q (a : ℝ) := 0 < 2 * a - 1 ∧ 2 * a - 1 < 1

theorem false_p_and_q (a : ℝ) :
  ¬ (p a ∧ q a) ↔ (a ≤ (1 : ℝ) / 2 ∨ a > (2 : ℝ) / 3) :=
by
  sorry

end false_p_and_q_l1419_141937


namespace projection_plane_right_angle_l1419_141985

-- Given conditions and definitions
def is_right_angle (α β : ℝ) : Prop := α = 90 ∧ β = 90
def is_parallel_to_side (plane : ℝ → ℝ → Prop) (side : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, plane x y ↔ a * x + b * y = c ∧ ∃ d e : ℝ, ∀ x y : ℝ, side x y ↔ d * x + e * y = 90

theorem projection_plane_right_angle (plane : ℝ → ℝ → Prop) (side1 side2 : ℝ → ℝ → Prop) :
  is_right_angle (90 : ℝ) (90 : ℝ) →
  (is_parallel_to_side plane side1 ∨ is_parallel_to_side plane side2) →
  ∃ α β : ℝ, is_right_angle α β :=
by 
  sorry

end projection_plane_right_angle_l1419_141985


namespace value_of_k_l1419_141939

-- Define the conditions of the quartic equation and the product of two roots
variable (a b c d k : ℝ)
variable (hx : (Polynomial.X ^ 4 - 18 * Polynomial.X ^ 3 + k * Polynomial.X ^ 2 + 200 * Polynomial.X - 1984).rootSet ℝ = {a, b, c, d})
variable (hprod_ab : a * b = -32)

-- The statement to prove: the value of k is 86
theorem value_of_k :
  k = 86 :=
by sorry

end value_of_k_l1419_141939


namespace find_second_divisor_l1419_141982

theorem find_second_divisor (k : ℕ) (d : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k < 42)
  (h3 : k % 7 = 3)
  (h4 : k % d = 5) : d = 12 := 
sorry

end find_second_divisor_l1419_141982


namespace arithmetic_mean_of_fractions_l1419_141917

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 8 + (5 : ℚ) / 12 / 2 = 19 / 48 := by
  sorry

end arithmetic_mean_of_fractions_l1419_141917


namespace functional_equation_solution_l1419_141902

theorem functional_equation_solution (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f (f (m + n)) = f m + f n) :
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a) ∨ (∀ n : ℤ, f n = 0) := by
  sorry

end functional_equation_solution_l1419_141902


namespace total_chairs_calculation_l1419_141948

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l1419_141948


namespace simplify_expression_l1419_141989

theorem simplify_expression :
  let a := 7
  let b := 11
  let c := 19
  (49 * (1 / 11 - 1 / 19) + 121 * (1 / 19 - 1 / 7) + 361 * (1 / 7 - 1 / 11)) /
  (7 * (1 / 11 - 1 / 19) + 11 * (1 / 19 - 1 / 7) + 19 * (1 / 7 - 1 / 11)) = 37 := by
  sorry

end simplify_expression_l1419_141989


namespace find_m_l1419_141969

def vector_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) : vector_perpendicular (3, 1) (m, -3) → m = 1 :=
by
  sorry

end find_m_l1419_141969


namespace brenda_spay_cats_l1419_141993

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end brenda_spay_cats_l1419_141993


namespace total_animal_eyes_l1419_141968

def num_snakes := 18
def num_alligators := 10
def eyes_per_snake := 2
def eyes_per_alligator := 2

theorem total_animal_eyes : 
  (num_snakes * eyes_per_snake) + (num_alligators * eyes_per_alligator) = 56 :=
by 
  sorry

end total_animal_eyes_l1419_141968


namespace problem_statement_l1419_141951

theorem problem_statement (p q : Prop) :
  ¬(p ∧ q) ∧ ¬¬p → ¬q := 
by 
  sorry

end problem_statement_l1419_141951


namespace largest_systematic_sample_l1419_141944

theorem largest_systematic_sample {n_products interval start second_smallest max_sample : ℕ} 
  (h1 : n_products = 300) 
  (h2 : start = 2) 
  (h3 : second_smallest = 17) 
  (h4 : interval = second_smallest - start) 
  (h5 : n_products % interval = 0) 
  (h6 : max_sample = start + (interval * ((n_products / interval) - 1))) : 
  max_sample = 287 := 
by
  -- This is where the proof would go if required.
  sorry

end largest_systematic_sample_l1419_141944


namespace Mike_height_l1419_141946

theorem Mike_height (h_mark: 5 * 12 + 3 = 63) (h_mark_mike:  63 + 10 = 73) (h_foot: 12 = 12)
: 73 / 12 = 6 ∧ 73 % 12 = 1 := 
sorry

end Mike_height_l1419_141946


namespace arithmetic_expression_equals_fraction_l1419_141927

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end arithmetic_expression_equals_fraction_l1419_141927


namespace goose_eggs_l1419_141915

theorem goose_eggs (E : ℕ) 
  (H1 : (2/3 : ℚ) * E = h) 
  (H2 : (3/4 : ℚ) * h = m)
  (H3 : (2/5 : ℚ) * m = 180) : 
  E = 2700 := 
sorry

end goose_eggs_l1419_141915


namespace find_a8_l1419_141932

def seq (a : Nat → Int) := a 1 = -1 ∧ ∀ n, a (n + 1) = a n - 3

theorem find_a8 (a : Nat → Int) (h : seq a) : a 8 = -22 :=
by {
  sorry
}

end find_a8_l1419_141932


namespace close_time_for_pipe_b_l1419_141931

-- Define entities and rates
def rate_fill (A_rate B_rate : ℝ) (t_fill t_empty t_fill_target t_close : ℝ) : Prop :=
  A_rate = 1 / t_fill ∧
  B_rate = 1 / t_empty ∧
  t_fill_target = 30 ∧
  A_rate * (t_close + (t_fill_target - t_close)) - B_rate * t_close = 1

-- Declare the theorem statement
theorem close_time_for_pipe_b (A_rate B_rate t_fill_target t_fill t_empty t_close: ℝ) :
   rate_fill A_rate B_rate t_fill t_empty t_fill_target t_close → t_close = 26.25 :=
by have h1 : A_rate = 1 / 15 := by sorry
   have h2 : B_rate = 1 / 24 := by sorry
   have h3 : t_fill_target = 30 := by sorry
   sorry

end close_time_for_pipe_b_l1419_141931


namespace prove_billy_age_l1419_141987

-- Define B and J as real numbers representing the ages of Billy and Joe respectively
variables (B J : ℝ)

-- State the conditions
def billy_triple_of_joe : Prop := B = 3 * J
def sum_of_ages : Prop := B + J = 63

-- State the proposition to prove
def billy_age_proof : Prop := B = 47.25

-- Main theorem combining the conditions and the proof statement
theorem prove_billy_age (h1 : billy_triple_of_joe B J) (h2 : sum_of_ages B J) : billy_age_proof B :=
by
  sorry

end prove_billy_age_l1419_141987


namespace part1_part2_part3_l1419_141904

-- Definitions of conditions
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_even_between (a b : ℕ) : ℕ := sum_even b - sum_even a

-- Problem 1: Prove that for n = 8, S = 72
theorem part1 (n : ℕ) (h : n = 8) : sum_even n = 72 := by
  rw [h]
  exact rfl

-- Problem 2: Prove the general formula for the sum of the first n consecutive even numbers
theorem part2 (n : ℕ) : sum_even n = n * (n + 1) := by
  exact rfl

-- Problem 3: Prove the sum of 102 to 212 is 8792 using the formula
theorem part3 : sum_even_between 50 106 = 8792 := by
  sorry

end part1_part2_part3_l1419_141904


namespace boxes_per_hand_l1419_141949

theorem boxes_per_hand (total_people : ℕ) (total_boxes : ℕ) (boxes_per_person : ℕ) (hands_per_person : ℕ) 
  (h1: total_people = 10) (h2: total_boxes = 20) (h3: boxes_per_person = total_boxes / total_people) 
  (h4: hands_per_person = 2) : boxes_per_person / hands_per_person = 1 := 
by
  sorry

end boxes_per_hand_l1419_141949


namespace units_digit_of_p_is_6_l1419_141966

-- Given conditions
variable (p : ℕ)
variable (h1 : p % 2 = 0)                -- p is a positive even integer
variable (h2 : (p^3 % 10) - (p^2 % 10) = 0)  -- The units digit of p^3 minus the units digit of p^2 is 0
variable (h3 : (p + 2) % 10 = 8)         -- The units digit of p + 2 is 8

-- Prove the units digit of p is 6
theorem units_digit_of_p_is_6 : p % 10 = 6 :=
sorry

end units_digit_of_p_is_6_l1419_141966


namespace alpha_beta_square_eq_eight_l1419_141971

theorem alpha_beta_square_eq_eight (α β : ℝ) 
  (hα : α^2 = 2*α + 1) 
  (hβ : β^2 = 2*β + 1) 
  (h_distinct : α ≠ β) : 
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_eq_eight_l1419_141971


namespace emily_sixth_score_needed_l1419_141980

def emily_test_scores : List ℕ := [88, 92, 85, 90, 97]

def needed_sixth_score (scores : List ℕ) (target_mean : ℕ) : ℕ :=
  let current_sum := scores.sum
  let total_sum_needed := target_mean * (scores.length + 1)
  total_sum_needed - current_sum

theorem emily_sixth_score_needed :
  needed_sixth_score emily_test_scores 91 = 94 := by
  sorry

end emily_sixth_score_needed_l1419_141980


namespace arithmetic_seq_a₄_l1419_141997

-- Definitions for conditions in the given problem
def S₅ (a₁ a₅ : ℕ) : ℕ := ((a₁ + a₅) * 5) / 2
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Final proof statement to show that a₄ = 9
theorem arithmetic_seq_a₄ (a₁ a₅ : ℕ) (d : ℕ) (h₁ : S₅ a₁ a₅ = 35) (h₂ : a₅ = 11) (h₃ : d = (a₅ - a₁) / 4) :
  arithmetic_sequence a₁ d 4 = 9 :=
sorry

end arithmetic_seq_a₄_l1419_141997


namespace range_of_b_for_monotonic_function_l1419_141983

theorem range_of_b_for_monotonic_function :
  (∀ x : ℝ, (x^2 + 2 * b * x + b + 2) ≥ 0) ↔ (-1 ≤ b ∧ b ≤ 2) :=
by sorry

end range_of_b_for_monotonic_function_l1419_141983


namespace sequences_zero_at_2_l1419_141947

theorem sequences_zero_at_2
  (a b c d : ℕ → ℝ)
  (h1 : ∀ n, a (n+1) = a n + b n)
  (h2 : ∀ n, b (n+1) = b n + c n)
  (h3 : ∀ n, c (n+1) = c n + d n)
  (h4 : ∀ n, d (n+1) = d n + a n)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (h5 : a (k + m) = a m)
  (h6 : b (k + m) = b m)
  (h7 : c (k + m) = c m)
  (h8 : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 :=
by sorry

end sequences_zero_at_2_l1419_141947


namespace proof_goal_l1419_141964

noncomputable def proof_problem (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : Prop :=
  (1 / a) + (1 / b) + (1 / c) > 4

theorem proof_goal (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : 
  (1 / a) + (1 / b) + (1 / c) > 4 :=
sorry

end proof_goal_l1419_141964
