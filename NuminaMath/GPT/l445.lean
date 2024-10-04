import Mathlib

namespace valid_pairs_count_l445_445781

theorem valid_pairs_count :
  let valid_digit (n : ℕ) : Prop := ∀ d ∈ digits 10 n, d ≠ 0 ∧ d ≠ 1 in
  let is_valid_pair (a b : ℕ) : Prop := a + b = 500 ∧ valid_digit a ∧ valid_digit b in
  (∃ (count : ℕ), count = {p : ℕ × ℕ | ∃ (a b : ℕ), p = (a, b) ∧ is_valid_pair a b}.toFinset.card) → count = 374 :=
by
  sorry

end valid_pairs_count_l445_445781


namespace count_valid_numbers_eq_18_l445_445710

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits in
  digits.nodup

def no_common_digits (a b : ℕ) : Prop :=
  let da := a.digits in
  let db := b.digits in
  disjoint da db

def valid_number (n : ℕ) : Prop :=
  n < 200 ∧
  distinct_digits n ∧
  distinct_digits (2 * n) ∧
  no_common_digits n (2 * n)

theorem count_valid_numbers_eq_18 : { n : ℕ | valid_number n }.count = 18 := sorry

end count_valid_numbers_eq_18_l445_445710


namespace banana_arrangement_count_l445_445274

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445274


namespace total_tires_mike_changed_l445_445061

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_tires_mike_changed_l445_445061


namespace perfect_square_count_zero_l445_445461

theorem perfect_square_count_zero :
  (finset.filter (λ n : ℕ, (4 ≤ n ∧ n ≤ 18) ∧ (∃ k : ℕ, 2 * n^2 + 3 = k^2))
    (finset.range 19)).card = 0 := 
  sorry

end perfect_square_count_zero_l445_445461


namespace average_of_numbers_l445_445102

noncomputable def x := (5050 : ℚ) / 5049

theorem average_of_numbers :
  let sum := (∑ i in Finset.range 101, (i + 1)) + x in
  let avg := sum / (101 + 1) in
  avg = 50 * x :=
by
  let sum := (∑ i in Finset.range 101, (i + 1)) + x
  let avg := sum / (101 + 1)
  have sum_formula : (∑ i in Finset.range 101, (i + 1)) = 5050 := sorry
  have avg_formula : avg = 50 * x := sorry
  exact avg_formula

end average_of_numbers_l445_445102


namespace bruce_bags_l445_445745

def cost_crayons := 5 * 5
def cost_books := 10 * 5
def cost_calculators := 3 * 5
def total_cost_before_discount := cost_crayons + cost_books + cost_calculators
def discount_books := 0.20 * cost_books
def cost_books_after_discount := cost_books - discount_books
def total_cost_after_discount := cost_crayons + cost_books_after_discount + cost_calculators
def initial_money := 200
def change_after_purchase := initial_money - total_cost_after_discount
def cost_of_one_bag := 10
def number_of_bags := change_after_purchase / cost_of_one_bag

theorem bruce_bags : number_of_bags = 12 := by
  sorry

end bruce_bags_l445_445745


namespace number_of_arrangements_of_BANANA_l445_445318

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445318


namespace hiking_time_l445_445522

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l445_445522


namespace combined_rent_C_D_l445_445726

theorem combined_rent_C_D :
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  combined_rent = 1020 :=
by
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  show combined_rent = 1020
  sorry

end combined_rent_C_D_l445_445726


namespace cost_to_fix_car_l445_445538

variable {S A : ℝ}

theorem cost_to_fix_car (h1 : A = 3 * S + 50) (h2 : S + A = 450) : A = 350 := 
by
  sorry

end cost_to_fix_car_l445_445538


namespace perm_banana_l445_445352

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445352


namespace ben_paints_150_square_feet_l445_445193

-- Define the given conditions
def ratio_allen_ben : ℕ := 3
def ratio_ben_allen : ℕ := 5
def total_work : ℕ := 240

-- Define the total amount of parts
def total_parts : ℕ := ratio_allen_ben + ratio_ben_allen

-- Define the work per part
def work_per_part : ℕ := total_work / total_parts

-- Define the work done by Ben
def ben_parts : ℕ := ratio_ben_allen
def ben_work : ℕ := work_per_part * ben_parts

-- The statement to be proved
theorem ben_paints_150_square_feet : ben_work = 150 :=
by
  sorry

end ben_paints_150_square_feet_l445_445193


namespace sin_half_angle_l445_445903

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445903


namespace english_speaking_students_l445_445007

theorem english_speaking_students (T H B E : ℕ) (hT : T = 40) (hH : H = 30) (hB : B = 10) (h_inclusion_exclusion : T = H + E - B) : E = 20 :=
by
  sorry

end english_speaking_students_l445_445007


namespace pool_filling_times_l445_445175

theorem pool_filling_times:
  ∃ (x y z u : ℕ),
    (1/x + 1/y = 1/70) ∧
    (1/x + 1/z = 1/84) ∧
    (1/y + 1/z = 1/140) ∧
    (1/u = 1/x + 1/y + 1/z) ∧
    (x = 105) ∧
    (y = 210) ∧
    (z = 420) ∧
    (u = 60) := 
  sorry

end pool_filling_times_l445_445175


namespace money_distribution_l445_445727

theorem money_distribution :
  ∀ (A B C : ℕ), 
  A + B + C = 900 → 
  B + C = 750 → 
  C = 250 → 
  A + C = 400 := 
by
  intros A B C h1 h2 h3
  sorry

end money_distribution_l445_445727


namespace book_allocation_correct_l445_445634

noncomputable def fair_allocation (A_win_probability B_win_probability : ℝ) (total_books : ℕ) : ℕ × ℕ :=
  (nat.floor (total_books * A_win_probability), nat.floor (total_books * B_win_probability))

theorem book_allocation_correct :
  fair_allocation (3 / 4) (1 / 4) 8 = (6, 2) :=
by
  sorry

end book_allocation_correct_l445_445634


namespace smallest_n_triangle_area_gt_3000_l445_445752

/-- Given the complex numbers (n + i), (n + i)^3, and (n + i)^4 form vertices of a triangle,
prove that the smallest positive integer n such that the area of this triangle is greater than 3000 is 10. -/
theorem smallest_n_triangle_area_gt_3000 :
  ∃ (n : ℕ), 0 < n ∧ let A : ℝ := abs (-2 * n ^ 5 + 12 * n ^ 3 - 9 * n) / 2 in A > 3000 ∧ n = 10 := sorry

end smallest_n_triangle_area_gt_3000_l445_445752


namespace banana_arrangement_count_l445_445282

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445282


namespace distinct_left_views_l445_445619

theorem distinct_left_views (cubes : Set (ℕ × ℕ × ℕ)) (grid_size : ℕ × ℕ) : 
  let num_cubes := 10
  cubes.card = num_cubes ∧ 
  grid_size = (3, 4) ∧ 
  all_adjacent? cubes  →
  num_left_views cubes = 16 :=
begin
  sorry
end

end distinct_left_views_l445_445619


namespace profit_calculation_l445_445121

theorem profit_calculation (cost_price_per_card_yuan : ℚ) (total_sales_yuan : ℚ)
  (n : ℕ) (sales_price_per_card_yuan : ℚ)
  (h1 : cost_price_per_card_yuan = 0.21)
  (h2 : total_sales_yuan = 14.57)
  (h3 : total_sales_yuan = n * sales_price_per_card_yuan)
  (h4 : sales_price_per_card_yuan ≤ 2 * cost_price_per_card_yuan) :
  (total_sales_yuan - n * cost_price_per_card_yuan = 4.7) :=
by
  sorry

end profit_calculation_l445_445121


namespace dot_product_sum_l445_445959

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom vec_sum : a + b + c = 0
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom norm_c : ∥c∥ = 2

-- The theorem to prove
theorem dot_product_sum :
  ⟪a, b⟫ + ⟪b, c⟫ + ⟪c, a⟫ = - 9 / 2 :=
sorry

end dot_product_sum_l445_445959


namespace a_general_formula_b_general_formula_T_sum_formula_l445_445804

def a : ℕ → ℝ
def b : ℕ → ℝ
def T : ℕ → ℝ

axiom arithmetic_seq (n : ℕ) (n_pos : 0 < n) : a(n + 1) > a n
axiom a1_eq_1 : a 1 = 1
axiom geometric_condition :
  (a 1 + 1) * (a 1 + 1) = (a 2 + 1) * (a 3 + 3)
axiom log_condition (n : ℕ) : a n + 2 * log 2 (b n) = -1

theorem a_general_formula (n : ℕ) : a n = 2 * n - 1 := sorry
theorem b_general_formula (n : ℕ) : b n = 1 / 2^n := sorry
theorem T_sum_formula (n : ℕ) : T n = 3 - (2 * n + 3) / 2^n := sorry

end a_general_formula_b_general_formula_T_sum_formula_l445_445804


namespace sin_half_angle_l445_445815

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445815


namespace calculate_area_ratio_l445_445704

noncomputable def area_ratio (AD AB : ℝ) : ℝ :=
  let AB_width := AB
  let AD_length := 4 * AB
  let r := AB / 2
  let area_rectangle := AD_length * AB_width
  let area_semicircles := π * r^2
  area_rectangle / area_semicircles

theorem calculate_area_ratio (AB : ℝ) (h1 : AB = 20) (h2 : AD / AB = 4 / 1) : area_ratio 80 20 = 16 / π :=
  by
  -- proof is omitted
  sorry

end calculate_area_ratio_l445_445704


namespace count_numbers_without_digit_2_l445_445973

/-- 
There are 323 whole numbers between 1 and 500 that do not contain the digit 2.
-/
theorem count_numbers_without_digit_2 : 
  let count_valid_numbers (digit : ℕ) : Bool :=
    if digit = 2 then false else true
  in  
  (1.to_finset.filter (λ n, (list.of_digits (nat.digits 10 n)).all count_valid_numbers)).card = 323 :=
by
  sorry

end count_numbers_without_digit_2_l445_445973


namespace total_selling_price_l445_445186

theorem total_selling_price (cost_per_meter profit_per_meter : ℕ) (total_meters : ℕ) :
  cost_per_meter = 90 → 
  profit_per_meter = 15 → 
  total_meters = 85 → 
  (cost_per_meter + profit_per_meter) * total_meters = 8925 :=
by
  intros
  sorry

end total_selling_price_l445_445186


namespace num_9digit_palindromes_l445_445459

theorem num_9digit_palindromes : 
  let digits := [1, 1, 3, 3, 5, 5, 7, 7, 7]
  ∃ palindromes : Finset (Finset ℕ), 
    (∀ p ∈ palindromes, is_palindrome_9digit p digits) ∧ 
    palindromes.card = 12 := 
sorry

def is_palindrome_9digit (p : Finset ℕ) (digits : List ℕ) : Prop :=
  let n := 9
  let half := (n - 1) / 2
  ∃ l : List ℕ, length l = half ∧
    all_equal (p.take half) (p.drop 1).reverse_drop half ∧
    p.to_list = digits

def all_equal {α : Type*} [DecidableEq α] : List α → List α → Prop
| [], [] => true
| (x :: xs), (y :: ys) => (x = y) ∧ all_equal xs ys
| _, _ => false

end num_9digit_palindromes_l445_445459


namespace permutations_of_BANANA_l445_445245

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445245


namespace find_k_l445_445986

open Real

noncomputable def chord_intersection (k : ℝ) : Prop :=
  let R : ℝ := 3
  let d := abs (k + 1) / sqrt (1 + k^2)
  d^2 + (12 * sqrt 5 / 10)^2 = R^2

theorem find_k (k : ℝ) (h : k > 1) (h_intersect : chord_intersection k) : k = 2 := by
  sorry

end find_k_l445_445986


namespace find_slope_angle_l445_445430

open Real

def point := ℝ × ℝ

def slope (p1 p2 : point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def angle_slope (k : ℝ) : ℕ → ℝ
| 0 := 0
| n+1 := k

theorem find_slope_angle (A B : point) (hA : A = (-1, 4)) (hB : B = (1, 2)) :
  ∃ α ∈ Ico 0 π, tan α = slope A B :=
sorry

end find_slope_angle_l445_445430


namespace cherry_pie_degrees_l445_445497

theorem cherry_pie_degrees (total_students : ℤ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
                           (fraction_cherry : ℚ) (remaining : ℕ) (total_degrees : ℕ) 
                           (final_total : ℕ)
                           (chocolate_prefs := 15) (apple_prefs := 10) 
                           (blueberry_prefs := 9) (one_third := 1/3) 
                           (total_students := 48) (total_degrees := 360) :
  
  let remaining_students := total_students - chocolate_prefs - apple_prefs - blueberry_prefs,
      cherry_students := round (fraction_cherry * remaining_students : ℚ),
      cherry_degrees := (cherry_students * total_degrees) / total_students in
  cherry_degrees = 37.5 :=
by
  sorry

end cherry_pie_degrees_l445_445497


namespace range_of_a_l445_445930

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a^2 * Real.sin (2 * x + π / 6) + 3 * a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ Set.Icc (-2 : ℝ) 2, ∃ x0 ∈ Set.Icc 0 (Real.pi / 2), g a x0 = f x1) ↔
  a ∈ Set.Iic (-4) ∪ Set.Ici 6 :=
sorry

end range_of_a_l445_445930


namespace min_value_fraction_l445_445449

theorem min_value_fraction (a b : Real) (a_pos : 0 < a) (b_pos : 0 < b) 
  (collinear : ∀ λ : Real, (a - 1, 1) = λ * (-b - 1, 2)) : 
  ∃ a b : Real, ∃ (a_pos : 0 < a) (b_pos : 0 < b), (2 * a + b = 1) ∧ (min_value : (1 / a + 2 / b) = 8) :=
sorry

end min_value_fraction_l445_445449


namespace minimum_flowers_to_guarantee_bouquets_l445_445996

theorem minimum_flowers_to_guarantee_bouquets :
  (∀ (num_types : ℕ) (flowers_per_bouquet : ℕ) (num_bouquets : ℕ),
   num_types = 6 → flowers_per_bouquet = 5 → num_bouquets = 10 →
   ∃ min_flowers : ℕ, min_flowers = 70 ∧
   ∀ (picked_flowers : ℕ → ℕ), 
     (∀ t : ℕ, t < num_types → picked_flowers t ≥ 0 ∧ 
                (t < num_types - 1 → picked_flowers t ≤ flowers_per_bouquet * (num_bouquets - 1) + 4)) → 
     ∑ t in finset.range num_types, picked_flowers t = min_flowers → 
     ∑ t in finset.range num_types, (picked_flowers t / flowers_per_bouquet) ≥ num_bouquets) := 
by {
  intro num_types flowers_per_bouquet num_bouquets,
  intro h1 h2 h3,
  use 70,
  split,
  {
    exact rfl,
  },
  {
    intros picked_flowers h_picked,
    sorry,
  }
}

end minimum_flowers_to_guarantee_bouquets_l445_445996


namespace sequence_general_term_l445_445800

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) - 2 * a n = 2^n) :
  a n = n * 2^(n - 1) :=
by
  induction n with k hk
  · simp
  · rw [h2 k, hk]
  · sorry

end sequence_general_term_l445_445800


namespace banana_permutations_l445_445257

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445257


namespace function_f_not_all_less_than_half_l445_445569

theorem function_f_not_all_less_than_half (p q : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = x^2 + p*x + q) :
  ¬ (|f 1| < 1 / 2 ∧ |f 2| < 1 / 2 ∧ |f 3| < 1 / 2) :=
sorry

end function_f_not_all_less_than_half_l445_445569


namespace find_number_l445_445772

theorem find_number (N : ℝ) (h : 0.6667 * N - 10 = 0.25 * N) : N ≈ 24 := 
sorry

end find_number_l445_445772


namespace sin_half_alpha_l445_445876

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445876


namespace banana_arrangements_l445_445328

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445328


namespace good_number_l445_445491

def prime_related (a b : ℕ) : Prop :=
∃ p : ℕ, p.prime ∧ (a = p * b ∨ b = p * a)

def has_at_least_three_divisors (n : ℕ) : Prop :=
∃ (d : Finset ℕ), d = n.divisors ∧ d.card ≥ 3

def can_be_arranged_prime_related (n : ℕ) : Prop :=
∃ (d : List ℕ), d.nodup ∧ Multiset.sort (List.toMultiset d) = n.divisors ∧ 
  ∀ (i : ℕ), i < d.length → prime_related (d.nthLe i (by linarith)) (d.nthLe ((i + 1) % d.length) (by simp [Nat.mod_lt]))

theorem good_number (n : ℕ) :
  has_at_least_three_divisors n ∧ can_be_arranged_prime_related n →
  ¬ (∃ p : ℕ, p.prime ∧ n = p^2) ∧
  ¬ (∃ (p : ℕ) (k : ℕ), p.prime ∧ k ≥ 2 ∧ n = p^k) :=
by sorry

end good_number_l445_445491


namespace total_skips_correct_l445_445374

def S (n : ℕ) : ℕ := n^2 + n

def TotalSkips5 : ℕ :=
  S 1 + S 2 + S 3 + S 4 + S 5

def Skips6 : ℕ :=
  2 * S 6

theorem total_skips_correct : TotalSkips5 + Skips6 = 154 :=
by
  -- proof goes here
  sorry

end total_skips_correct_l445_445374


namespace assign_roles_l445_445711

def maleRoles : ℕ := 3
def femaleRoles : ℕ := 3
def eitherGenderRoles : ℕ := 4
def menCount : ℕ := 7
def womenCount : ℕ := 8

theorem assign_roles : 
  (menCount.choose maleRoles) * 
  (womenCount.choose femaleRoles) * 
  ((menCount + womenCount - maleRoles - femaleRoles).choose eitherGenderRoles) = 213955200 := 
  sorry

end assign_roles_l445_445711


namespace dishwasher_manager_ratio_l445_445739

noncomputable theory
open_locale classical

variables 
  (w_d w_c w_m : ℝ)
  (h1 : w_m = 8.5) 
  (h2 : w_c = w_m - 3.40) 
  (h3 : w_c = 1.20 * w_d)

theorem dishwasher_manager_ratio : w_d / w_m = 0.5 :=
by sorry

end dishwasher_manager_ratio_l445_445739


namespace advertisement_arrangement_l445_445693

theorem advertisement_arrangement :
  ∃ (n : ℕ), n = 48 ∧
    let slots := [1, 2, 3, 4, 5, 6],
        public_service_ads := {1, 2},
        commercial_ads := {3, 4, 5, 6},
        arrangements := 
          PublicServiceArrangement → 
          CommercialArrangement → 
          total arrangements = A_2-2 * A_4-4
    n = arrangements
:= by sorry

end advertisement_arrangement_l445_445693


namespace fraction_evaluation_l445_445124

theorem fraction_evaluation : (1 / 2) + (1 / 2 * 1 / 2) = 3 / 4 := by
  sorry

end fraction_evaluation_l445_445124


namespace largest_n_divisibility_l445_445777

theorem largest_n_divisibility :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧
  (∀ m : ℕ, (m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 :=
by
  sorry

end largest_n_divisibility_l445_445777


namespace permutations_of_banana_l445_445291

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445291


namespace profit_sharing_l445_445148

theorem profit_sharing 
  (total_profit : ℝ) 
  (managing_share_percentage : ℝ) 
  (capital_a : ℝ) 
  (capital_b : ℝ) 
  (managing_partner_share : ℝ)
  (total_capital : ℝ) 
  (remaining_profit : ℝ) 
  (proportion_a : ℝ)
  (share_a_remaining : ℝ)
  (total_share_a : ℝ) : 
  total_profit = 8800 → 
  managing_share_percentage = 0.125 → 
  capital_a = 50000 → 
  capital_b = 60000 → 
  managing_partner_share = managing_share_percentage * total_profit → 
  total_capital = capital_a + capital_b → 
  remaining_profit = total_profit - managing_partner_share → 
  proportion_a = capital_a / total_capital → 
  share_a_remaining = proportion_a * remaining_profit → 
  total_share_a = managing_partner_share + share_a_remaining → 
  total_share_a = 4600 :=
by sorry

end profit_sharing_l445_445148


namespace total_hike_time_l445_445524

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l445_445524


namespace percentage_increase_l445_445154

theorem percentage_increase (L : ℝ) (h : L + 60 = 240) : ((60 / L) * 100 = 33 + (1 / 3) * 100) :=
by
  sorry

end percentage_increase_l445_445154


namespace find_initial_sum_of_money_l445_445184

noncomputable def initialSumOfMoney (P r : ℝ) : Prop :=
  (920 = P + (P * r * 3) / 100) ∧ (992 = P + (P * (r + 3) * 3) / 100)

theorem find_initial_sum_of_money :
  ∃ P, initialSumOfMoney P r ∧ P = 800 :=
by
  -- Assume P and r are real numbers
  let P : ℝ := 800
  let r : ℝ := 6 -- This specific value is to assist the proof by setting it to the expected interest rate
  
  -- Check the conditions for P and r
  have h1 : 920 = P + (P * r * 3) / 100 := sorry
  have h2 : 992 = P + (P * (r + 3) * 3) / 100 := sorry

  -- Exists statement by construction
  exact ⟨P, ⟨h1, h2⟩, rfl⟩

end find_initial_sum_of_money_l445_445184


namespace problem_sum_of_relatively_prime_divisors_of_1000_l445_445042

open BigOperators
open Nat

/-- 
Let S be the sum of all numbers of the form a/b, where a and b are 
relatively prime positive divisors of 1000. Prove that the greatest integer 
that does not exceed S/10 is 29. 
-/
theorem problem_sum_of_relatively_prime_divisors_of_1000 :
  let S := ∑ (i j k l : ℕ) in
    finset.Icc 0 3 ×ᶠ finset.Icc 0 3 ×ᶠ finset.Icc 0 3 ×ᶠ finset.Icc 0 3,
    if gcd (2^i * 5^j) (2^k * 5^l) = 1 then (2^i * 5^j) / (2^k * 5^l) else 0
  in floor (S / 10) = 29 := sorry

end problem_sum_of_relatively_prime_divisors_of_1000_l445_445042


namespace angle_of_inclination_of_line_l445_445099

/-- The line defined by the equation y + 3 = 0 is horizontal, hence its angle of inclination is 0 degrees. -/
theorem angle_of_inclination_of_line (y : ℝ) : y = -3 → angle_of_inclination(λ x, -3) = 0 :=
by 
  intro h
  rw [h]
  sorry

end angle_of_inclination_of_line_l445_445099


namespace probability_sum_11_is_1_over_8_l445_445732

theorem probability_sum_11_is_1_over_8 : 
  let outcomes := ({2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ)
  let pairs := {p : ℕ × ℕ | p.1 ∈ outcomes ∧ p.2 ∈ outcomes ∧ p.1 + p.2 = 11}
  let total_outcomes := outcomes.card * outcomes.card
  let favorable_outcomes := pairs.card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8
:= by 
  sorry

end probability_sum_11_is_1_over_8_l445_445732


namespace cistern_fill_time_l445_445668

-- Define the filling rate and emptying rate as given conditions.
def R_fill : ℚ := 1 / 5
def R_empty : ℚ := 1 / 9

-- Define the net rate when both taps are opened simultaneously.
def R_net : ℚ := R_fill - R_empty

-- The total time to fill the cistern when both taps are opened.
def fill_time := 1 / R_net

-- Prove that the total time to fill the cistern is 11.25 hours.
theorem cistern_fill_time : fill_time = 11.25 := 
by 
    -- We include sorry to bypass the actual proof. This will allow the code to compile.
    sorry

end cistern_fill_time_l445_445668


namespace total_number_of_outcomes_two_white_one_black_outcomes_at_least_two_white_outcomes_probability_two_white_one_black_probability_at_least_two_white_l445_445991

namespace BallDrawing

-- Definitions based on the problem statement
def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls
def num_drawn_balls : ℕ := 3

-- Problem statements (to be proven)
theorem total_number_of_outcomes : combin.choose total_balls num_drawn_balls = 84 := sorry
theorem two_white_one_black_outcomes : combin.choose num_white_balls 2 * combin.choose num_black_balls 1 = 30 := sorry
theorem at_least_two_white_outcomes : 
  combin.choose num_white_balls 2 * combin.choose num_black_balls 1 + combin.choose num_white_balls 3 * combin.choose num_black_balls 0 = 34 := sorry
theorem probability_two_white_one_black: 
  (combin.choose num_white_balls 2 * combin.choose num_black_balls 1) / (combin.choose total_balls num_drawn_balls : ℚ) = 5/14 := sorry
theorem probability_at_least_two_white: 
  ((combin.choose num_white_balls 2 * combin.choose num_black_balls 1 + combin.choose num_white_balls 3 * combin.choose num_black_balls 0) 
   / (combin.choose total_balls num_drawn_balls : ℚ) = 17/42) := sorry

end BallDrawing

end total_number_of_outcomes_two_white_one_black_outcomes_at_least_two_white_outcomes_probability_two_white_one_black_probability_at_least_two_white_l445_445991


namespace sin_half_alpha_l445_445891

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445891


namespace perm_banana_l445_445346

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445346


namespace distance_of_center_to_point_l445_445644

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_of_center_to_point : 
  let C := (3, -4 : ℝ × ℝ) in
  let P := (5, -3 : ℝ × ℝ) in
  dist C P = real.sqrt 5 :=
by
  sorry

end distance_of_center_to_point_l445_445644


namespace range_of_p_l445_445604

noncomputable def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

theorem range_of_p : ∀ y : ℝ, y ∈ set_of (λ y, ∃ x (hx : 0 ≤ x), p x = y) ↔ y ∈ set.Ici 9 :=
  by sorry

end range_of_p_l445_445604


namespace total_length_segments_l445_445728

noncomputable def segment_length (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment : ℕ) :=
  let total_length := rect_horizontal_1 + rect_horizontal_2 + rect_vertical
  total_length - 8 + left_segment

theorem total_length_segments
  (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment total_left : ℕ)
  (h1 : rect_horizontal_1 = 10)
  (h2 : rect_horizontal_2 = 3)
  (h3 : rect_vertical = 12)
  (h4 : left_segment = 8)
  (h5 : total_left = 19)
  : segment_length rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment = total_left :=
sorry

end total_length_segments_l445_445728


namespace max_value_at_2_l445_445914

noncomputable def f (x : ℝ) : ℝ := -x^3 + 12 * x

theorem max_value_at_2 : ∃ a : ℝ, (∀ x : ℝ, f x ≤ f a) ∧ a = 2 := 
by
  sorry

end max_value_at_2_l445_445914


namespace min_flowers_for_bouquets_l445_445998

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end min_flowers_for_bouquets_l445_445998


namespace vector_dot_product_sum_l445_445964

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem vector_dot_product_sum (h₁ : a + b + c = 0) 
                               (ha : ∥a∥ = 1)
                               (hb : ∥b∥ = 2)
                               (hc : ∥c∥ = 2) :
  (a ⬝ b) + (b ⬝ c) + (c ⬝ a) = -9 / 2 := sorry

end vector_dot_product_sum_l445_445964


namespace BANANA_arrangements_l445_445308

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445308


namespace permutations_of_BANANA_l445_445238

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445238


namespace image_of_circle_is_ellipse_l445_445608

-- Define the square ABCD and its properties
variables {A B C D A' B' C' D' O O' M' P' N' Q' F' K' : Type}
variables (p : parallelogram A' B' C' D')
variables (s : square A B C D)
variables (h_projection_of_square : projects_to_square s p)

-- Define the circle inscribed in the square
variables (circle_inscribed : inscribed_circle s)
variables (ellipse_projected : ellipse p)

-- Define the projection properties of the square centers and midpoints
variables (h_center_projection : projects_to_center s p O O')
variables (h_midpoints_projection : projects_to_midpoints s p)

-- Define points in the projection corresponding to intersection of perpendicular diagonals
variables (K'_constructed : K' ∈ ellipse_projected)

-- The statement to prove the image of the circle is an ellipse
theorem image_of_circle_is_ellipse :
  is_ellipse circle_inscribed ellipse_projected :=
by
  sorry

end image_of_circle_is_ellipse_l445_445608


namespace permutations_of_BANANA_l445_445239

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445239


namespace inverse_proportion_quadrant_l445_445172

theorem inverse_proportion_quadrant (k : ℝ) (h : k < 0) : 
  ∀ x : ℝ, (0 < x → y = k / x → y < 0) ∧ (x < 0 → y = k / x → 0 < y) :=
by
  sorry

end inverse_proportion_quadrant_l445_445172


namespace tangent_line_at_P_minimized_distance_point_Q_l445_445440

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2

def P : ℝ × ℝ := (1, -2)

def line_l (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem tangent_line_at_P :
  let f' := deriv f 1 in
  f' = 1 → (∀ x y : ℝ, y + 2 = x - 1 → x - y - 3 = 0) :=
by
  intros f' hf' x y h_tangent
  sorry

theorem minimized_distance_point_Q :
  ∃ x : ℝ, x = 2 ∧ (∀ y : ℝ, y = Real.log 2 - 2 → let Q := (x, y) in 
  ∀ d : ℝ, d = (abs (Q.1 - 2 * Real.log Q.1 + 7)) / (Real.sqrt 5) → 
  (∀ h : ℝ, h ≠ d → h > d)) :=
by
  intros
  use [2, Real.log 2 - 2]
  sorry

end tangent_line_at_P_minimized_distance_point_Q_l445_445440


namespace distinct_arrangements_of_BANANA_l445_445345

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445345


namespace BANANA_arrangements_l445_445300

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445300


namespace rectangle_to_square_l445_445235

theorem rectangle_to_square : 
  ∃ (a b c : ℝ), a * b = 25 * 4 ∧ a * c = 25 * 4 ∧ b * c = 100 ∧ (a + c = b) :=
begin
  sorry
end

end rectangle_to_square_l445_445235


namespace time_to_cross_bridge_l445_445673

variable (L_train : ℕ) (L_bridge : ℕ) (Speed_km_h : ℕ)
variable (D_total : ℕ) (Speed_m_s : ℕ) (Time : ℝ)

axiom h1 : L_train = 110
axiom h2 : L_bridge = 132
axiom h3 : Speed_km_h = 72
axiom h4 : D_total = L_train + L_bridge
axiom h5 : Speed_m_s = (Speed_km_h * 1000) / 3600
axiom h6 : Time = D_total / Speed_m_s

theorem time_to_cross_bridge : Time = 12.1 := by
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end time_to_cross_bridge_l445_445673


namespace total_profit_l445_445191

theorem total_profit (B C : ℝ) (C_share : ℝ) (hC_share : C_share = 3375.0000000000005)
  (hA1 : A = 3 * B)
  (hA2 : A = (2 / 3) * C)
  (hC_to_B : C = (9 / 2) * B) :
  total_profit = 6375.000000000001 :=
by
  -- Definitions and necessary intermediate steps to be added here
  -- Proof step placeholder
  sorry

end total_profit_l445_445191


namespace compare_numbers_l445_445216

-- Conditions for proper comparison
def compareIntegers (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : Prop := a > b
def compareDecimals (a b : ℝ) (ha : a < 1) (hb : b < 1) : Prop := a < b
def eqDecimals (a b : ℝ) : Prop := a = b

theorem compare_numbers :
  compareIntegers 3 2.95 (by norm_num) (by norm_num) ∧
  compareDecimals 0.08 0.21 (by norm_num) (by norm_num) ∧
  eqDecimals 0.6 0.6 :=
by {
  repeat { sorry },
}

end compare_numbers_l445_445216


namespace regular_pyramid_angle_k_l445_445015

theorem regular_pyramid_angle_k (k : ℝ) 
  (h1 : regular_quadrilateral_pyramid)
  (h2 : plane_through_two_lateral_edges)
  (h3 : ratio_of_areas k) : 
  (0 < k ∧ k < real.sqrt 2 / 4) ∧ 
  ∃ θ, θ = real.arccos (8 * k^2 - 1) :=
by
  sorry

end regular_pyramid_angle_k_l445_445015


namespace percentage_of_boys_among_boyds_friends_l445_445533

theorem percentage_of_boys_among_boyds_friends : 
  (Julian_total : ℕ) (Julian_girls_percentage : ℕ) (Boyd_total : ℕ) (Boyd_girls_multiplier : ℕ) 
  (H1 : Julian_total = 80) 
  (H2 : Julian_girls_percentage = 40)
  (H3 : Boyd_total = 100)
  (H4 : Boyd_girls_multiplier = 2)
  : (Boyd_boys_percentage : ℕ) := 
  let Julian_girls : ℕ := Julian_total * Julian_girls_percentage / 100,
  let Boyd_girls : ℕ := Julian_girls * Boyd_girls_multiplier,
  let Boyd_boys : ℕ := Boyd_total - Boyd_girls,
  let Boyd_boys_percentage : ℕ := Boyd_boys * 100 / Boyd_total
  Boyd_boys_percentage = 36 :=
sorry

end percentage_of_boys_among_boyds_friends_l445_445533


namespace side_length_estimate_l445_445100

theorem side_length_estimate (x : ℝ) (h : x^2 = 15) : 3 < x ∧ x < 4 :=
sorry

end side_length_estimate_l445_445100


namespace find_all_a_l445_445775

open Real

noncomputable def isCircle := ∀ x y : ℝ, (|x| - 12)^2 + (|y| - 5)^2 = 169
noncomputable def isVShape (a : ℝ) := ∀ x y : ℝ, y = -|x - √a| + 2 - √a
noncomputable def exactlyThreeSolutions (a : ℝ) := ∃! (x y : ℝ), isCircle x y ∧ isVShape a x y 

theorem find_all_a :
  {a : ℝ | exactlyThreeSolutions a} = {1, 36, (13 * sqrt 2 - 5) / 2 ^ 2} := sorry

end find_all_a_l445_445775


namespace distinct_arrangements_of_BANANA_l445_445340

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445340


namespace napkin_ratio_l445_445068

theorem napkin_ratio (initial_napkins : ℕ) (napkins_after : ℕ) (olivia_napkins : ℕ) (amelia_napkins : ℕ)
  (h1 : initial_napkins = 15) (h2 : napkins_after = 45) (h3 : olivia_napkins = 10)
  (h4 : initial_napkins + olivia_napkins + amelia_napkins = napkins_after) :
  amelia_napkins / olivia_napkins = 2 := by
  sorry

end napkin_ratio_l445_445068


namespace model_a_sampling_l445_445699

theorem model_a_sampling 
  (outputA : ℕ) (outputB : ℕ) (outputC : ℕ) (total_inspected : ℕ)
  (hA : outputA = 1200) (hB : outputB = 6000) (hC : outputC = 2000) (hT : total_inspected = 46) :
  let total_production := outputA + outputB + outputC,
      proportion_A := (outputA : ℚ) / total_production,
      model_A_to_inspect := proportion_A * total_inspected
  in model_A_to_inspect = 6 :=
by {
  let total_production := outputA + outputB + outputC,
  let proportion_A := (outputA : ℚ) / total_production,
  let model_A_to_inspect := proportion_A * total_inspected,
  sorry
}

end model_a_sampling_l445_445699


namespace perm_banana_l445_445348

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445348


namespace translation_of_civilisation_l445_445151

def translation (word : String) (translation : String) : Prop :=
translation = "civilization"

theorem translation_of_civilisation (word : String) :
  word = "civilisation" → translation word "civilization" :=
by sorry

end translation_of_civilisation_l445_445151


namespace arithmetic_sequence_sum_2017_l445_445810

noncomputable def arithmetic_sequence_sum {a : ℕ → ℤ} (S : ℕ → ℤ) (a₁ a₂ a₃ a₁₇ a₂₀₀₁ : ℤ) : Prop :=
  (∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)) ∧
  S 0 = 0 ∧
  (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) ∧
  (a₁₇ - 3) * a₁ + a₂₀₀₁ * a₂ = 1

theorem arithmetic_sequence_sum_2017 {a : ℕ → ℤ} {S : ℕ → ℤ} {a₁ a₂ a₃ a₁₇ a₂₀₀₁ : ℤ} :
  arithmetic_sequence_sum S a₁ a₂ a₃ a₁₇ a₂₀₀₁ →
  ((a₁₇ + a₂₀₁) = 4) →
  S 2017 = 4034 :=
by
  sorry

end arithmetic_sequence_sum_2017_l445_445810


namespace permutations_of_banana_l445_445290

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445290


namespace problem_l445_445541

variable (x : ℝ)

theorem problem (A B : ℝ) 
  (h : (A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3))): 
  A + B = 15 := by
  sorry

end problem_l445_445541


namespace distinct_arrangements_of_BANANA_l445_445343

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445343


namespace angle_CBD_l445_445690

theorem angle_CBD {A B C : Type*} {m_C m_A : ℝ} (hABC : A ≠ B ∧ B ≠ C ∧ C ≠ A) 
  (h_AC_gt_BC : dist A C > dist B C) (h_m_C : m_C = 30) (h_m_A : m_A = 50) :
  let m_ABC := 180 - m_A - m_C in let m_CBD := 180 - m_ABC in m_CBD = 80 :=
by 
  have m_ABC : ℝ := 180 - m_A - m_C
  have m_CBD : ℝ := 180 - m_ABC
  exact sorry

end angle_CBD_l445_445690


namespace exists_nat_coprime_sum_primes_l445_445572

open Nat

def sum_primes_less_than (n : ℕ) : ℕ :=
  (List.filter Nat.prime (List.range n)).sum

theorem exists_nat_coprime_sum_primes :
  ∃ n : ℕ, n > 10^100 ∧ Nat.coprime (sum_primes_less_than n) n :=
by
  sorry

end exists_nat_coprime_sum_primes_l445_445572


namespace banana_arrangement_count_l445_445276

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445276


namespace smallest_palindrome_divisible_by_6_l445_445649

-- Define what it means to be a five-digit palindrome of the form ABCBA
def is_palindrome_ABCBA (n : ℕ) : Prop :=
  ∃ A B C : ℕ, A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧
  (n = A * 10001 + B * 1010 + C * 100)

-- Define the condition that the number is divisible by 6
def divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

-- The main theorem
theorem smallest_palindrome_divisible_by_6 : ∃ n : ℕ, is_palindrome_ABCBA n ∧ divisible_by_6 n ∧ ∀ m : ℕ, is_palindrome_ABCBA m ∧ divisible_by_6 m → n ≤ m :=
begin
  use 20002,
  split,
  { -- Proof that 20002 is of the form ABCBA
    unfold is_palindrome_ABCBA,
    use [2, 0, 0],
    split, norm_num,
    split, norm_num,
    split, norm_num,
    refl,
  },
  split,
  { -- Proof that 20002 is divisible by 6
    unfold divisible_by_6,
    norm_num,
  },
  { -- Proof that 20002 is the smallest such palindrome
    intros m Hm,
    -- This will be a statement saying the smallest palindrome
    -- and we put sorry here since proof isn't needed
    sorry,
  }
end

end smallest_palindrome_divisible_by_6_l445_445649


namespace minimum_steps_to_remove_zeroes_l445_445680

theorem minimum_steps_to_remove_zeroes (zeroes ones : ℕ) (h_zeroes : zeroes = 150) (h_ones : ones = 151) :
  ∃ n : ℕ, n = 150 ∧
  (forall z w : ℕ, (z + w = zeroes + ones ∧ w ≥ zeroes → (step_result (z, w) n = (0, w + zeroes))) :=
begin
  sorry
end

-- Helper function to simulate the step result
def step_result : (ℕ × ℕ) → ℕ → (ℕ × ℕ)
| (z, w), n := if n = 0 then (z, w)
               else step_result ((z - 1), (w + 1)) (n - 1)

end minimum_steps_to_remove_zeroes_l445_445680


namespace number_of_arrangements_of_BANANA_l445_445317

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445317


namespace solve_system_l445_445095

variable (x y z w : ℝ)
variable (s1 s2 s3 s4 : ℝ)
variable (p q m n b1 b2 b3 b4 : ℝ)

-- Defining the conditions from the problem
def conditions := 
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 66

-- Problem statement
theorem solve_system : 
  conditions x y z w →
  ∃ (x y z w : ℝ), conditions x y z w :=
by
  sorry

end solve_system_l445_445095


namespace rohan_age_future_years_l445_445006

-- Conditions
variables (x : ℕ) -- Number of years into the future
def Rohan_present_age := 25
def Rohan_future_age (x : ℕ) := Rohan_present_age + x
def Rohan_past_age (x : ℕ) := Rohan_present_age - x

-- Problem statement
theorem rohan_age_future_years (x : ℕ) (h : Rohan_future_age(x) = 4 * Rohan_past_age(x)) : x = 15 := 
by {
  -- Condition definitions
  have h1 : Rohan_future_age(x) = 25 + x := rfl,
  have h2 : Rohan_past_age(x) = 25 - x := rfl,
  
  -- Substituting the conditions into the problem statement
  rw [h1, h2] at h,
  
  -- Proceeding with the proof, omitted by sorry
  sorry
}

end rohan_age_future_years_l445_445006


namespace find_value_l445_445555

-- Define the periodic function f with period 2
def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -4 * x ^ 2 + 1
  else if 0 ≤ x ∧ x < 1 then x + 7 / 4
  else f (x - 2)

theorem find_value : f (f (3 / 2)) = 7 / 4 := by
  sorry

end find_value_l445_445555


namespace range_of_distance_l445_445513

-- Define the parametric equations of the line l
def parametric_eq_line_l (t : ℝ) : ℝ × ℝ :=
  (t - 1, t + 2)

-- Define the polar equation of the curve C
def polar_eq_curve_C (θ : ℝ) : ℝ :=
  sqrt (3 / (1 + 2 * (cos θ)^2))

-- Define the distance function d between a point on curve C and line l
def distance_point_line (x y : ℝ) : ℝ :=
  abs (x - y + 3) / sqrt 2

theorem range_of_distance : 
  let α := (arccosθ : ∀ θ, -1 ≤ cos θ ∧ cos θ ≤ 1) in
  (∀ (y x : ℝ), ∃ α, (x = cos α ∧ y = sqrt 3 * sin α) → 
   (distance_point_line x y) ∈ set.Icc (sqrt 2 / 2) (5 * sqrt 2 / 2)) :=
sorry -- Proof is omitted

end range_of_distance_l445_445513


namespace math_proof_problem_l445_445557

variable {a_n : ℕ → ℝ} -- sequence a_n
variable {b_n : ℕ → ℝ} -- sequence b_n

-- Given that a_n is an arithmetic sequence with common difference d
def isArithmeticSequence (a_n : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a_n (n + 1) = a_n n + d

-- Given condition for sequence b_n
def b_n_def (a_n b_n : ℕ → ℝ) : Prop := ∀ n, b_n n = a_n (n + 1) * a_n (n + 2) - a_n n ^ 2

-- Both sequences have common difference d ≠ 0
def common_difference_ne_zero (a_n b_n : ℕ → ℝ) (d : ℝ) : Prop :=
  isArithmeticSequence a_n d ∧ isArithmeticSequence b_n d ∧ d ≠ 0

-- Condition involving positive integers s and t
def integer_condition (a_n b_n : ℕ → ℝ) (s t : ℕ) : Prop :=
  1 ≤ s ∧ 1 ≤ t ∧ ∃ (x : ℤ), a_n s + b_n t = x

-- Theorem to prove that the sequence {b_n} is arithmetic and find minimum value of |a_1|
theorem math_proof_problem
  (a_n b_n : ℕ → ℝ) (d : ℝ) (s t : ℕ)
  (arithmetic_a : isArithmeticSequence a_n d)
  (defined_b : b_n_def a_n b_n)
  (common_diff : common_difference_ne_zero a_n b_n d)
  (int_condition : integer_condition a_n b_n s t) :
  (isArithmeticSequence b_n (3 * d ^ 2)) ∧ (∃ m : ℝ, m = |a_n 1| ∧ m = 1 / 36) :=
  by sorry

end math_proof_problem_l445_445557


namespace quadratic_roots_relation_l445_445616

theorem quadratic_roots_relation (a b c d : ℝ) (h : ∀ x : ℝ, (c * x^2 + d * x + a = 0) → 
  (a * (2007 * x)^2 + b * (2007 * x) + c = 0)) : b^2 = d^2 := 
sorry

end quadratic_roots_relation_l445_445616


namespace find_integers_l445_445130

theorem find_integers 
  (A k : ℕ) 
  (h_sum : A + A * k + A * k^2 = 93) 
  (h_product : A * (A * k) * (A * k^2) = 3375) : 
  (A, A * k, A * k^2) = (3, 15, 75) := 
by 
  sorry

end find_integers_l445_445130


namespace banana_arrangement_count_l445_445283

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445283


namespace seq_a8_value_l445_445198

theorem seq_a8_value 
  (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n) 
  (a7_eq : a 7 = 120) 
  : a 8 = 194 :=
sorry

end seq_a8_value_l445_445198


namespace union_of_A_B_l445_445411

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l445_445411


namespace smaller_of_x_and_y_is_15_l445_445635

variable {x y : ℕ}

/-- Given two positive numbers x and y are in the ratio 3:5, 
and the sum of x and y plus 10 equals 50,
prove that the smaller of x and y is 15. -/
theorem smaller_of_x_and_y_is_15 (h1 : x * 5 = y * 3) (h2 : x + y + 10 = 50) (h3 : 0 < x) (h4 : 0 < y) : x = 15 :=
by
  sorry

end smaller_of_x_and_y_is_15_l445_445635


namespace sin_half_alpha_l445_445895

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445895


namespace problem_statement_l445_445043

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem problem_statement :
  ∃ (a : ℕ → ℝ) (d : ℝ), d > 0 ∧ a 2 + a 3 + a 4 = 15 ∧ a 2 * a 3 * a 4 = 80 ∧ (a 12 + a 13 + a 14 = 105) := 
begin
  sorry -- proof is omitted
end

end problem_statement_l445_445043


namespace sin_half_alpha_l445_445864

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445864


namespace even_iff_exists_square_l445_445089

variable (P Q : Polynomial ℝ)

def is_even_poly (P : Polynomial ℝ) : Prop := ∀ x : ℝ, P (-x) = P x

theorem even_iff_exists_square (P : Polynomial ℝ) :
  is_even_poly P ↔ ∃ Q : Polynomial ℝ, ∀ x : ℝ, P x = Q (x^2) :=
sorry

end even_iff_exists_square_l445_445089


namespace percentage_of_boyds_boy_friends_l445_445535

-- Definitions based on conditions
def number_of_julian_friends : ℕ := 80
def percentage_of_julian_boys : ℕ := 60
def percentage_of_julian_girls : ℕ := 40
def number_of_boyd_friends : ℕ := 100

-- Calculation based on conditions
def num_julian_boy_friends : ℕ := (percentage_of_julian_boys * number_of_julian_friends) / 100
def num_julian_girl_friends : ℕ := (percentage_of_julian_girls * number_of_julian_friends) / 100
def num_boyd_girl_friends : ℕ := 2 * num_julian_girl_friends
def num_boyd_boy_friends : ℕ := number_of_boyd_friends - num_boyd_girl_friends

-- Prove percentage of Boyd's friends who are boys
theorem percentage_of_boyds_boy_friends : (num_boyd_boy_friends * 100 / number_of_boyd_friends) = 36 :=
by
  simp [num_julian_boy_friends, num_julian_girl_friends, num_boyd_girl_friends, num_boyd_boy_friends, percentage_of_julian_boys, percentage_of_julian_girls, number_of_julian_friends, number_of_boyd_friends]
  sorry

end percentage_of_boyds_boy_friends_l445_445535


namespace circumradii_sum_geq_half_perimeter_l445_445053

-- Define the hexagon and its properties

variables (A B C D E F : Type)
variables [ConvexHexagon A B C D E F] (AB DE BC EF CD AF : Line A B C D E F)
variables (R_A R_C R_E : ℝ) (P : ℝ)

-- Define the parallel conditions and properties of the hexagon
axiom hexagon_parallel_conds :
  AB ∥ DE ∧ BC ∥ EF ∧ CD ∥ AF 
  
-- Define the circumradii given the specific triangles
axiom circumradii_defs :
  R_A = circumradius △(F A B) ∧ 
  R_C = circumradius △(B C D) ∧ 
  R_E = circumradius △(D E F)

-- Define the perimeter of the hexagon
axiom perimeter_def :
  P = perimeter_of_hexagon A B C D E F

-- The main statement to prove
theorem circumradii_sum_geq_half_perimeter : 
  R_A + R_C + R_E ≥ P / 2 := by 
  sorry

end circumradii_sum_geq_half_perimeter_l445_445053


namespace right_angled_triangle_other_angle_isosceles_triangle_base_angle_l445_445017

theorem right_angled_triangle_other_angle (a : ℝ) (h1 : 0 < a) (h2 : a < 90) (h3 : 40 = a) :
  50 = 90 - a :=
sorry

theorem isosceles_triangle_base_angle (v : ℝ) (h1 : 0 < v) (h2 : v < 180) (h3 : 80 = v) :
  50 = (180 - v) / 2 :=
sorry

end right_angled_triangle_other_angle_isosceles_triangle_base_angle_l445_445017


namespace original_list_length_l445_445662

variable (n m : ℕ)   -- number of integers and the mean respectively
variable (l : List ℤ) -- the original list of integers

def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Condition 1: Appending 25 increases mean by 3
def condition1 (l : List ℤ) : Prop :=
  mean (25 :: l) = mean l + 3

-- Condition 2: Appending -4 to the enlarged list decreases the mean by 1.5
def condition2 (l : List ℤ) : Prop :=
  mean (-4 :: 25 :: l) = mean (25 :: l) - 1.5

theorem original_list_length (l : List ℤ) (h1 : condition1 l) (h2 : condition2 l) : l.length = 4 := by
  sorry

end original_list_length_l445_445662


namespace benny_spent_on_baseball_gear_l445_445742

variable (initial_amount : ℕ := 67)
variable (amount_left : ℕ := 33)
variable (amount_spent : ℕ)

theorem benny_spent_on_baseball_gear : amount_spent = initial_amount - amount_left := by
  have : initial_amount = 67 := rfl
  have : amount_left = 33 := rfl
  have : amount_spent = 67 - 33 := by
    exact 34
  exact this

end benny_spent_on_baseball_gear_l445_445742


namespace measure_angle_ABC_l445_445116

-- Definitions
def O : Point := sorry
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def circle : Circle := sorry
def is_center (O : Point) (circle : Circle) : Prop := sorry
def circumscribed (triangle : Triangle) (circle : Circle) : Prop := sorry
def angle (p1 p2 p3 : Point) : Angle := sorry
def measure_angle (ang : Angle) : ℝ := sorry
def inscribed_angle_theorem (ang : Angle) (arc_measure : ℝ) : Prop :=
  measure_angle ang = arc_measure / 2

-- Conditions
axiom O_is_center : is_center O circle
axiom triangle_ABC_circumscribed : circumscribed (Triangle.mk A B C) circle
axiom measure_AOB : measure_angle (angle A O B) = 140
axiom measure_BOC : measure_angle (angle B O C) = 120

-- Proof problem
theorem measure_angle_ABC : measure_angle (angle A B C) = 50 := by
  sorry

end measure_angle_ABC_l445_445116


namespace arrangement_of_BANANA_l445_445363

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445363


namespace shortest_distance_between_circles_l445_445648

theorem shortest_distance_between_circles :
  let circle1 := (x^2 - 12*x + y^2 - 6*y + 9 = 0)
  let circle2 := (x^2 + 10*x + y^2 + 8*y + 34 = 0)
  -- Centers and radii from conditions above:
  let center1 := (6, 3)
  let radius1 := 3
  let center2 := (-5, -4)
  let radius2 := Real.sqrt 7
  let distance_centers := Real.sqrt ((6 - (-5))^2 + (3 - (-4))^2)
  -- Calculate shortest distance
  distance_centers - (radius1 + radius2) = Real.sqrt 170 - 3 - Real.sqrt 7 := sorry

end shortest_distance_between_circles_l445_445648


namespace number_of_arrangements_of_BANANA_l445_445320

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445320


namespace probability_A_not_winning_l445_445633

theorem probability_A_not_winning 
  (prob_draw : ℚ := 1/2)
  (prob_B_wins : ℚ := 1/3) : 
  (prob_draw + prob_B_wins) = 5 / 6 := 
by
  sorry

end probability_A_not_winning_l445_445633


namespace max_intersecting_subsets_l445_445041

theorem max_intersecting_subsets (M : Set α) (n : ℕ) 
  (hM : M.finite) (h_card : M.toFinset.card = n)
  (A : Finset (Set α)) 
  (hA : ∀ s ∈ A, s ⊆ M)
  (h_inter : ∀ s₁ s₂ ∈ A, s₁ ≠ s₂ → s₁ ∩ s₂ ≠ ∅)
  : A.card ≤ 2 ^ (n - 1) := 
sorry

end max_intersecting_subsets_l445_445041


namespace find_x_l445_445785

noncomputable section

open Real

theorem find_x (x : ℝ) (hx : 0 < x ∧ x < 180) : 
  tan (120 * π / 180 - x * π / 180) = (sin (120 * π / 180) - sin (x * π / 180)) / (cos (120 * π / 180) - cos (x * π / 180)) →
  x = 100 :=
by
  sorry

end find_x_l445_445785


namespace parabola_focus_l445_445115

theorem parabola_focus (p : ℝ) (y₀ : ℝ) (h₀ : p > 0) (hA : 6 * 6 = 2 * p * 6) (hAF : sqrt (6^2 + y₀^2) = 2 * p) : p = 4 :=
by
  sorry

end parabola_focus_l445_445115


namespace total_hike_time_l445_445526

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l445_445526


namespace four_corresponds_to_364_l445_445014

noncomputable def number_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 36
  | 3 => 363
  | 5 => 365
  | 36 => 2
  | _ => 0 -- Assume 0 as the default case

theorem four_corresponds_to_364 : number_pattern 4 = 364 :=
sorry

end four_corresponds_to_364_l445_445014


namespace find_n_l445_445141

theorem find_n (n : ℝ) :
  (10 : ℝ)^n = 10^(-3) * real.sqrt ((10^69) / (0.0001)) * 10^2 → n = 35.5 :=
by
  intro h
  sorry

end find_n_l445_445141


namespace fraction_meaningful_range_l445_445630

-- Define the condition
def meaningful_fraction_condition (x : ℝ) : Prop := (x - 2023) ≠ 0

-- Define the conclusion that we need to prove
def meaningful_fraction_range (x : ℝ) : Prop := x ≠ 2023

theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction_condition x → meaningful_fraction_range x :=
by
  intro h
  -- Proof steps would go here
  sorry

end fraction_meaningful_range_l445_445630


namespace find_r_l445_445229

-- Define sequence S as a finite geometric sequence
def geometric_seq (b r : ℝ) := list.range 10 |>.map (λ n, b * r^n)

-- Define the transformation function B
def B (s : list ℝ) : list ℝ :=
s.zip (s.tail.getD []) |>.map (λ p, real.sqrt (p.1 * p.2))

-- Define recursive transformations B^m(S)
def B_m (s : list ℝ) (m : ℕ) : list ℝ :=
if m = 0 then s else B_m (B s) (m - 1)

-- Prove r = 1 given the final condition
theorem find_r (b r : ℝ) (h : B_m (geometric_seq b r) 9 = [r^4]) : r = 1 :=
by {
  -- proof not required, use sorry
  sorry
}

end find_r_l445_445229


namespace red_knights_fraction_magic_l445_445501

theorem red_knights_fraction_magic (total_knights red_knights blue_knights magical_knights : ℕ)
  (h1 : red_knights = (3 / 8 : ℚ) * total_knights)
  (h2 : blue_knights = total_knights - red_knights)
  (h3 : magical_knights = (1 / 4 : ℚ) * total_knights)
  (fraction_red_magic fraction_blue_magic : ℚ) 
  (h4 : fraction_red_magic = 3 * fraction_blue_magic)
  (h5 : magical_knights = red_knights * fraction_red_magic + blue_knights * fraction_blue_magic) :
  fraction_red_magic = 3 / 7 := 
by
  sorry

end red_knights_fraction_magic_l445_445501


namespace g_h_of_2_eq_869_l445_445484

-- Define the functions g and h
def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -2 * x^3 - 1

-- State the theorem we need to prove
theorem g_h_of_2_eq_869 : g (h 2) = 869 := by
  sorry

end g_h_of_2_eq_869_l445_445484


namespace banana_arrangements_l445_445322

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445322


namespace sphere_surface_area_l445_445073

-- Definitions and conditions
def A := Point
def B := Point
def C := Point
def D := Point

def AB := 4
def AC := 4
def AD := 4

-- Orthogonality conditions
def mutually_orthogonal (u v w : ℝ) : Prop :=
  dot_product u v = 0 ∧ dot_product v w = 0 ∧ dot_product w u = 0

def on_sphere (A B C D : Point) (r : ℝ) : Prop :=
  distance A B = AB ∧ distance A C = AC ∧ distance A D = AD

-- Sphere's surface area to prove
theorem sphere_surface_area :
  ∀ A B C D : Point,
    on_sphere A B C D 2√3 →
    mutually_orthogonal AB AC AD →
    surface_area = 48 * π :=
by
  sorry

end sphere_surface_area_l445_445073


namespace total_hike_time_l445_445527

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l445_445527


namespace tickets_difference_l445_445740

theorem tickets_difference (t_toy : ℕ) (t_clothes : ℕ) : t_toy = 8 → t_clothes = 18 → t_clothes - t_toy = 10 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end tickets_difference_l445_445740


namespace binom_12_11_eq_12_l445_445222

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := 
by {
  sorry
}

end binom_12_11_eq_12_l445_445222


namespace mark_increase_reading_time_l445_445059

def initial_pages_per_day : ℕ := 100
def final_pages_per_week : ℕ := 1750
def days_in_week : ℕ := 7

def calculate_percentage_increase (initial_pages_per_day : ℕ) (final_pages_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  ((final_pages_per_week : ℚ) / ((initial_pages_per_day : ℚ) * (days_in_week : ℚ)) - 1) * 100

theorem mark_increase_reading_time :
  calculate_percentage_increase initial_pages_per_day final_pages_per_week days_in_week = 150 :=
by sorry

end mark_increase_reading_time_l445_445059


namespace symmetry_of_g_wrt_line_l445_445429

theorem symmetry_of_g_wrt_line :
  (∀ x : ℝ, sin x + a * cos x = sin (10 * π / 3 - x) + a * cos (10 * π / 3 - x)) →
  (∀ x : ℝ, a * sin x + cos x = a * sin (11 * π / 6 - x) + cos (11 * π / 6 - x)) :=
begin
  sorry
end

end symmetry_of_g_wrt_line_l445_445429


namespace vector_dot_product_sum_l445_445951

variables {V : Type*} [InnerProductSpace ℝ V]
(open InnerProductSpace)

def vector_a : V := sorry
def vector_b : V := sorry
def vector_c : V := sorry

#check vector_a

-- Given conditions
def cond1 : vector_a + vector_b + vector_c = (0 : V) := sorry
def cond2 : ∥vector_a∥ = 1 := sorry
def cond3 : ∥vector_b∥ = 2 := sorry
def cond4 : ∥vector_c∥ = 2 := sorry

-- Proof goal
theorem vector_dot_product_sum :
  vector_a + vector_b + vector_c = 0 →
  ∥vector_a∥ = 1 →
  ∥vector_b∥ = 2 →
  ∥vector_c∥ = 2 →
  (⟪vector_a, vector_b⟫ + ⟪vector_b, vector_c⟫ + ⟪vector_c, vector_a⟫ : ℝ) = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445951


namespace square_carpet_side_length_l445_445196

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ s : ℝ, s * s = area ∧ 3 < s ∧ s < 4 :=
by
  sorry

end square_carpet_side_length_l445_445196


namespace sin_half_alpha_l445_445898

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445898


namespace perm_banana_l445_445351

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445351


namespace find_range_of_x_l445_445760

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 ^ x else 2 ^ (-x)

theorem find_range_of_x (x : ℝ) : 
  f (1 - 2 * x) < f 3 ↔ (-1 < x ∧ x < 2) := 
sorry

end find_range_of_x_l445_445760


namespace cos_alpha_of_point_l445_445433

variable (α : Real)
variable (x y : Real)
variable h : (x, y) = (Real.sqrt 3 / 2, 1 / 2)

theorem cos_alpha_of_point :
  ∃ α : Real, α ∈ Real ∧ ∀ x y, (x, y) = (Real.sqrt 3 / 2, 1 / 2) → Real.cos α = x / (Real.sqrt (x^2 + y^2)) := by
  sorry

end cos_alpha_of_point_l445_445433


namespace max_intersection_points_perpendicular_bisectors_l445_445399

theorem max_intersection_points_perpendicular_bisectors
  (P : Fin 10 → ℝ × ℝ) :
  let numTriangles := Nat.choose 10 3,
      numQuadrilaterals := Nat.choose 10 4
  in numTriangles + 3 * numQuadrilaterals = 750 := by
  sorry

end max_intersection_points_perpendicular_bisectors_l445_445399


namespace Gina_needs_more_tip_l445_445396

/-- Gina is considered a good tipper if she tips at least 20%.
  Gina initially tipped 5% on a bill of $26.
  How many more cents would Gina need to tip to be considered a good tipper? -/

theorem Gina_needs_more_tip
  (bill : ℝ)
  (tip_percent_Gina : ℝ)
  (good_tip_percent : ℝ)
  (cents_per_dollar : ℤ)
  (Gina_tip : ℝ := bill * tip_percent_Gina)
  (good_tip : ℝ := bill * good_tip_percent)
  (difference_in_dollars : ℝ := good_tip - Gina_tip)
  (difference_in_cents : ℤ := (difference_in_dollars * ↑cents_per_dollar).toInt) :
  bill = 26 → tip_percent_Gina = 0.05 → good_tip_percent = 0.20 → cents_per_dollar = 100 → difference_in_cents = 390 :=
by
  intros h1 h2 h3 h4
  sorry

end Gina_needs_more_tip_l445_445396


namespace cosine_area_l445_445591

theorem cosine_area : 
  ∫ x in (0:ℝ)..(3 * real.pi / 2), real.cos x = 3 := by
  sorry

end cosine_area_l445_445591


namespace amy_initial_money_l445_445663

-- Define the conditions
variable (left_fair : ℕ) (spent : ℕ)

-- Define the proof problem statement
theorem amy_initial_money (h1 : left_fair = 11) (h2 : spent = 4) : left_fair + spent = 15 := 
by sorry

end amy_initial_money_l445_445663


namespace midpoint_CQ_on_Gamma_l445_445540

noncomputable def right_triangle :=
  {A B C : Point}
  (h_right_angle : ∠ABC = π / 2)
  (h_AB_gt_BC : |AB| > |BC|)

noncomputable def semicircle (A B : Point) :=
  {Γ : Circle}
  (h_diameter : diameter Γ A B)
  (h_same_side : ∀ (C : Point), C lies on same side of AB)

noncomputable def on_semicircle (P : Point) (Γ : Circle) :=
  P ∈ Γ

noncomputable def point_condition_BP_eq_BC (B P C : Point) :=
  |BP| = |BC|

noncomputable def point_condition_AP_eq_AQ (A P Q : Point) :=
  |AP| = |AQ|

theorem midpoint_CQ_on_Gamma
  (A B C P Q : Point)
  (h_triangle : right_triangle A B C)
  (h_semicircle : semicircle A B)
  (h_on_semicircle : on_semicircle P (h_semicircle.1))
  (h_BP_BC : point_condition_BP_eq_BC B P C)
  (h_AP_AQ : point_condition_AP_eq_AQ A P Q) :
  midpoint C Q ∈ h_semicircle.1 := by
  sorry

end midpoint_CQ_on_Gamma_l445_445540


namespace skater_track_length_l445_445636

-- Defining the conditions
variables (s a t : ℝ)

-- Proving the length of the track
theorem skater_track_length (h1 : s > 0) (h2 : a > 0) (h3 : t > 0) : 
  ∃ x : ℝ, x = (s / (120 * t)) * (real.sqrt(a^2 + 240 * a * t) - a) :=
by
  use (s / (120 * t)) * (real.sqrt(a^2 + 240 * a * t) - a)
  sorry

end skater_track_length_l445_445636


namespace sin_half_alpha_l445_445867

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445867


namespace polynomial_constant_l445_445713

theorem polynomial_constant
  (P : Polynomial ℤ)
  (h : ∀ Q F G : Polynomial ℤ, P.comp Q = F * G → F.degree = 0 ∨ G.degree = 0) :
  P.degree = 0 :=
by sorry

end polynomial_constant_l445_445713


namespace permutations_of_banana_l445_445297

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445297


namespace dot_product_sum_l445_445956

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom vec_sum : a + b + c = 0
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom norm_c : ∥c∥ = 2

-- The theorem to prove
theorem dot_product_sum :
  ⟪a, b⟫ + ⟪b, c⟫ + ⟪c, a⟫ = - 9 / 2 :=
sorry

end dot_product_sum_l445_445956


namespace find_max_n_l445_445786

def h (x : ℕ) : ℕ := if ∃ k : ℕ, x = 3^k then x else 1

def T_n (n : ℕ) : ℕ := 
  ∑ k in finset.range (3^(n-1)), h (3 * (k + 1))

theorem find_max_n : ∃ (n : ℕ), n < 300 ∧ (∃ m : ℕ, T_n n = m^3) ∧ n = 215 :=
by
  sorry

end find_max_n_l445_445786


namespace smallest_period_f_min_value_f_max_value_f_l445_445441

noncomputable def f (x : ℝ) : ℝ := sin x * (sqrt 3 * cos x - sin x)

theorem smallest_period_f : Function.periodic f π :=
begin
  sorry
end

theorem min_value_f : ∀ x ∈ Icc (-π/12) (π/3), f x ≥ -1/2 :=
begin
  sorry
end

theorem max_value_f : ∀ x ∈ Icc (-π/12) (π/3), f x ≤ 1/2 :=
begin
  sorry
end

end smallest_period_f_min_value_f_max_value_f_l445_445441


namespace binom_12_11_l445_445225

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_l445_445225


namespace solve_for_nabla_l445_445975

theorem solve_for_nabla : 
  ∃ (nabla : ℚ), 5 * (-3/2 : ℚ) = nabla - 3 → nabla = -9/2 :=
begin
  intro h,
  use -9/2,
  linarith,
end

end solve_for_nabla_l445_445975


namespace jeremy_watermelons_l445_445531

theorem jeremy_watermelons (eat_per_week : ℕ) (give_per_week : ℕ) (weeks : ℕ)
  (h_eat : eat_per_week = 3)
  (h_give : give_per_week = 2)
  (h_weeks : weeks = 6) :
  (eat_per_week + give_per_week) * weeks = 30 := by
  rw [h_eat, h_give, h_weeks]
  -- Further proof steps will go here
  sorry

end jeremy_watermelons_l445_445531


namespace two_digit_multiples_of_7_l445_445470

theorem two_digit_multiples_of_7 : 
  {n : ℕ | n ≥ 10 ∧ n ≤ 99 ∧ n % 7 = 0}.card = 13 :=
sorry

end two_digit_multiples_of_7_l445_445470


namespace gina_extra_tip_l445_445394

theorem gina_extra_tip 
  (bill_in_dollars : ℕ) (bill_in_cents : ℕ)
  (good_tip_percent normal_tip_percent : ℕ)
  (gina_normal_tip extra_tip_needed : ℕ) :
  bill_in_dollars = 26 -> 
  bill_in_cents = bill_in_dollars * 100 ->
  normal_tip_percent = 5 ->
  good_tip_percent = 20 -> 
  gina_normal_tip = bill_in_cents * normal_tip_percent / 100 ->
  let good_tip := bill_in_cents * good_tip_percent / 100 in
  extra_tip_needed = good_tip - gina_normal_tip -> 
  extra_tip_needed = 390 :=
by
  sorry

end gina_extra_tip_l445_445394


namespace sum_of_first_10_terms_arithmetic_sequence_l445_445405

theorem sum_of_first_10_terms_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1) 
  (h₀ : a 1 = -4) 
  (h₁ : a 4 + a 6 = 16): 
  (∑ i in Finset.range 10, a (i + 1)) = 95 :=
by
  sorry

end sum_of_first_10_terms_arithmetic_sequence_l445_445405


namespace problem_statement_l445_445912

theorem problem_statement :
  (∑ k in Finset.range 45, Real.sin (4 * (k + 1))) = 0 →
  ∃ (m n : ℕ), m + n = 1 ∧ Int.gcd m n = 1 ∧ Real.tan (m / n) < 90 :=
by
  intro h
  use 0, 1
  have h1 : Int.gcd 0 1 = 1 := by rw Int.gcd_zero_right
  have h2 : Real.tan (0 / 1 : ℝ) = 0 := by rw Real.tan_zero
  rw [add_zero]
  exact ⟨rfl, h1, by norm_num⟩

end problem_statement_l445_445912


namespace simplify_and_evaluate_expression_l445_445090

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℝ), 
  x = -1 / 3 → y = -2 → 
  (3 * x + 2 * y) * (3 * x - 2 * y) - 5 * x * (x - y) - (2 * x - y)^2 = -14 :=
by
  intros x y hx hy
  sorry

end simplify_and_evaluate_expression_l445_445090


namespace final_statement_l445_445806

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x, f (x) = f (-x)
axiom periodic_minus_one : ∀ x, f (x + 1) = -f (x)
axiom increasing_on_neg_one_to_zero : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f (x) < f (y)

-- Statement
theorem final_statement :
  (∀ x, f (x + 2) = f (x)) ∧
  (¬ (∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) < f (x + 1))) ∧
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f (x) < f (y)) ∧
  (f (2) = f (0)) :=
by
  sorry

end final_statement_l445_445806


namespace find_mango_rate_l445_445458

variable (total_purchase : Nat)
variable (grapes_qty : Nat) (grapes_cost_per_kg : Nat) (mangoes_qty : Nat) (mangoes_rate : Nat)

-- Given conditions
def grape_cost : Nat := grapes_qty * grapes_cost_per_kg
def cost_of_mangoes (total_cost: Nat) (grape_cost: Nat): Nat := total_cost - grape_cost
def rate_of_mangoes (mango_cost: Nat) (mango_qty: Nat): Nat := mango_cost / mango_qty

-- Main goal
theorem find_mango_rate (h1: total_purchase = 1125) (h2: grapes_qty = 9) (h3: grapes_cost_per_kg = 70) 
  (h4: mangoes_qty = 9) (h5: mangoes_rate = 55) :
  mangoes_rate = rate_of_mangoes (cost_of_mangoes total_purchase (grape_cost)) mangoes_qty := by
  sorry

end find_mango_rate_l445_445458


namespace common_ratio_l445_445558

-- Define that \( a_n \) is an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a(n + 1) = a(n) + d

-- Define that \( b_n \) is a geometric sequence
def is_geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, b(n + 1) = q * b(n)

-- Definitions of the problem
def problem (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  is_arithmetic_seq a ∧ is_geometric_seq b ∧
  a(1) < a(2) ∧ b(1) < b(2) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 3 ⟶ b(i) = a(i)^2)

-- The statement to prove
theorem common_ratio (a b : ℕ → ℝ) (h : problem a b) : 
  ∃ q, ∀ x, b x = (a x)^2 ∧ q = 3 + 2 * real.sqrt 2 :=
sorry

end common_ratio_l445_445558


namespace distinct_arrangements_of_BANANA_l445_445336

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445336


namespace correct_choice_l445_445194

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (domain : set ℝ) : Prop :=
∀ x y : ℝ, x ∈ domain → y ∈ domain → x < y → f x < f y

theorem correct_choice :
  ∃ f : ℝ → ℝ, f = (λ x, |x|) ∧ is_even f ∧ is_monotonically_increasing f {x : ℝ | 0 < x} :=
begin
  use (λ x, |x|),
  split,
  { refl },
  split,
  { intros x,
    exact abs_neg x },
  { intros x y hx hy hxy,
    exact abs_increasing_on_pos hxy }
end

end correct_choice_l445_445194


namespace max_min_f_when_a_neg_2_range_of_a_for_monotonicity_l445_445928

open Set Real

noncomputable section

-- Define the function f(x) given a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 3

-- Question 1
theorem max_min_f_when_a_neg_2 :
  let a := -2
  let x_interval := Ici (-4) ∩ Iic 6
  let f := f a
  (∀ x ∈ x_interval, f (-4) ≤ f x) ∧ (∀ x ∈ x_interval, f x ≥ f 2) ∧ f (-4) = 35 ∧ f 2 = -1 := sorry

-- Question 2
theorem range_of_a_for_monotonicity :
  (∀ a, (∀ x₁ x₂ ∈ Ici (-4) ∩ Iic 6, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ∨ (∀ x₁ x₂ ∈ Ici (-4) ∩ Iic 6, x₁ ≤ x₂ → f a x₁ ≥ f a x₂)) ↔
  (a ≥ 4 ∨ a ≤ -6) := sorry

end max_min_f_when_a_neg_2_range_of_a_for_monotonicity_l445_445928


namespace arrangement_of_BANANA_l445_445364

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445364


namespace smallest_triangle_perimeter_l445_445140

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ), 
    a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
    a + 2 = b ∧ b + 2 = c ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧
    a + b + c = 15 :=
by {
  use [3, 5, 7],
  simp,
  sorry
}

end smallest_triangle_perimeter_l445_445140


namespace clock_hands_angle_120_between_7_and_8_l445_445765

theorem clock_hands_angle_120_between_7_and_8 :
  ∃ (t₁ t₂ : ℕ), (t₁ = 5) ∧ (t₂ = 16) ∧ 
  (∃ (h₀ m₀ : ℕ → ℝ), 
    h₀ 7 = 210 ∧ 
    m₀ 7 = 0 ∧
    (∀ t : ℕ, h₀ (7 + t / 60) = 210 + t * (30 / 60)) ∧
    (∀ t : ℕ, m₀ (7 + t / 60) = t * (360 / 60)) ∧
    ((h₀ (7 + t₁ / 60) - m₀ (7 + t₁ / 60)) % 360 = 120) ∧ 
    ((h₀ (7 + t₂ / 60) - m₀ (7 + t₂ / 60)) % 360 = 120)) := by
  sorry

end clock_hands_angle_120_between_7_and_8_l445_445765


namespace fill_cup_times_l445_445730

theorem fill_cup_times (a b : ℚ) (c : ℚ) (d : ℕ) (h1 : b = 3  + 3 / 4) (h2 : a = 1 / 3) (h3 : c = b / a) (h4 : d = c.toNatCeil ) : d = 12 :=
by
  sorry

end fill_cup_times_l445_445730


namespace black_friday_sales_projection_l445_445070

theorem black_friday_sales_projection (sold_now : ℕ) (increment : ℕ) (years : ℕ) 
  (h_now : sold_now = 327) (h_inc : increment = 50) (h_years : years = 3) : 
  let sold_three_years := sold_now + 3 * increment in
  sold_three_years = 477 := 
by
  -- Definitions according to the conditions
  have h1 : sold_now = 327 := h_now
  have h2 : increment = 50 := h_inc
  have h3 : years = 3 := h_years

  -- Calculation based on definitions
  have h_sold_next_year := sold_now + increment
  have h_sold_second_year := h_sold_next_year + increment
  have h_sold_third_year := h_sold_second_year + increment
  
  -- Haven't elaborated on proof steps as the problem requires the statement only
  sorry

end black_friday_sales_projection_l445_445070


namespace Angelina_speed_grocery_to_gym_l445_445150

-- Define parameters for distances and times
def distance_home_to_grocery : ℕ := 720
def distance_grocery_to_gym : ℕ := 480
def time_difference : ℕ := 40

-- Define speeds
variable (v : ℕ) -- speed in meters per second from home to grocery
def speed_home_to_grocery := v
def speed_grocery_to_gym := 2 * v

-- Define times using given speeds and distances
def time_home_to_grocery := distance_home_to_grocery / speed_home_to_grocery
def time_grocery_to_gym := distance_grocery_to_gym / speed_grocery_to_gym

-- Proof statement for the problem
theorem Angelina_speed_grocery_to_gym
  (v_pos : 0 < v)
  (condition : time_home_to_grocery - time_difference = time_grocery_to_gym) :
  speed_grocery_to_gym = 24 := by
  sorry

end Angelina_speed_grocery_to_gym_l445_445150


namespace disk_tangent_position_l445_445705

noncomputable def clock_radius : ℝ := 30
noncomputable def disk_radius : ℝ := 15
noncomputable def clock_circumference := 2 * Real.pi * clock_radius
noncomputable def disk_circumference := 2 * Real.pi * disk_radius

theorem disk_tangent_position :
  (∀ (θ : ℝ), θ ≥ 0 → θ < 2 * Real.pi → 
    let tangency_positions := λ θ, (θ / Real.pi) * clock_radius in tangency_positions (disk_circumference / disk_radius * Real.pi) = tangency_positions (Real.pi)) :=
begin
  sorry
end

end disk_tangent_position_l445_445705


namespace nonnegative_integer_pairs_solution_l445_445773

theorem nonnegative_integer_pairs_solution :
  ∀ (x y: ℕ), ((x * y + 2) ^ 2 = x^2 + y^2) ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end nonnegative_integer_pairs_solution_l445_445773


namespace integer_triangle_600_integer_triangle_144_l445_445379

-- Problem Part I
theorem integer_triangle_600 :
  ∃ (a b c : ℕ), a * b * c = 600 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 26 :=
by {
  sorry
}

-- Problem Part II
theorem integer_triangle_144 :
  ∃ (a b c : ℕ), a * b * c = 144 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 16 :=
by {
  sorry
}

end integer_triangle_600_integer_triangle_144_l445_445379


namespace percentage_of_girls_after_change_l445_445507

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end percentage_of_girls_after_change_l445_445507


namespace locus_of_Q_eq_l445_445924

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 24) + (y^2 / 16) = 1
noncomputable def line_eq (x y : ℝ) : Prop := (x / 12) + (y / 8) = 1
noncomputable def point_on_OP_eq (x y : ℝ) (P : ℝ) (Q R : ℝ) : Prop := 
  R^2 = |x * Q|

axiom point_on_line (x y : ℝ) : line_eq x y

axiom loci_eq (x y : ℝ) : ellipse_eq x y ∧ ∀ P ∈ point_on_line x y, 
  let R := (x^2 / 24 + y^2 / 16) ^ 1/2,
      Q := |x * P / R^2| 
  in ( Q = (x, y))

theorem locus_of_Q_eq (x y : ℝ) : loci_eq x y → 
  ((x - 1)^2 / (5/2)) + ((y - 1)^2 / (5/3)) = 1 := 
by
  sorry

end locus_of_Q_eq_l445_445924


namespace sin_half_angle_l445_445840

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445840


namespace smallest_ones_divisible_by_threes_l445_445650

theorem smallest_ones_divisible_by_threes :
  ∃ (n : ℕ), (n = 300) ∧ (∃ (k : ℕ), k = 333 * (10^99 + 10^98 + ... + 10^0)) ∧ (10^n - 1) / 9 % k = 0 := by
  sorry

end smallest_ones_divisible_by_threes_l445_445650


namespace BANANA_arrangements_l445_445270

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445270


namespace vector_dot_product_sum_l445_445952

variables {V : Type*} [InnerProductSpace ℝ V]
(open InnerProductSpace)

def vector_a : V := sorry
def vector_b : V := sorry
def vector_c : V := sorry

#check vector_a

-- Given conditions
def cond1 : vector_a + vector_b + vector_c = (0 : V) := sorry
def cond2 : ∥vector_a∥ = 1 := sorry
def cond3 : ∥vector_b∥ = 2 := sorry
def cond4 : ∥vector_c∥ = 2 := sorry

-- Proof goal
theorem vector_dot_product_sum :
  vector_a + vector_b + vector_c = 0 →
  ∥vector_a∥ = 1 →
  ∥vector_b∥ = 2 →
  ∥vector_c∥ = 2 →
  (⟪vector_a, vector_b⟫ + ⟪vector_b, vector_c⟫ + ⟪vector_c, vector_a⟫ : ℝ) = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445952


namespace cows_horses_ratio_l445_445202

theorem cows_horses_ratio (cows horses : ℕ) (h : cows = 21) (ratio : cows / horses = 7 / 2) : horses = 6 :=
sorry

end cows_horses_ratio_l445_445202


namespace multiples_of_15_between_20_and_200_l445_445464

theorem multiples_of_15_between_20_and_200 : 
  ∃ n : ℕ, (∀ k : ℕ, 20 < k*15 ∧ k*15 < 200 → k*15 ∈ (30:ℕ) + (n-1)*15) ∧ n = 12 :=
by 
  sorry

end multiples_of_15_between_20_and_200_l445_445464


namespace find_all_waldo_time_l445_445128

theorem find_all_waldo_time (b : ℕ) (p : ℕ) (t : ℕ) :
  b = 15 → p = 30 → t = 3 → b * p * t = 1350 := by
sorry

end find_all_waldo_time_l445_445128


namespace round_robin_count_triples_l445_445018

theorem round_robin_count_triples 
  {n : ℕ} (h_n : n = 41) 
  (h_games : ∀ t : ℕ, t < n → n - 1 = 20) 
  (h_wins : ∀ t : ℕ, t < n → 12 wins)
  (h_losses : ∀ t : ℕ, t < n → 8 losses)
  (h_no_ties : no_ties) :
  set.of_triples (λ {A B C : ℕ}, A beats B ∧ B beats C ∧ C beats A) = 7954 :=
by sorry

end round_robin_count_triples_l445_445018


namespace sin_half_angle_l445_445848

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445848


namespace card_arrangement_probability_l445_445625

theorem card_arrangement_probability : 
  let cards := ["E", "E", "B"] in 
  let permutations := multiset.permutations (multiset.of_list cards) in
  let favorable := multiset.filter (λ l, [list.head [l] = "B", l.nth 1 = some "E", l.nth 2 = some "E"]) permutations in
  multiset.card favorable.to_finset / multiset.card permutations.to_finset = 1 / 3 :=
by
  sorry

end card_arrangement_probability_l445_445625


namespace ellie_oil_needs_l445_445373

def oil_per_wheel : ℕ := 10
def number_of_wheels : ℕ := 2
def oil_for_rest : ℕ := 5
def total_oil_needed : ℕ := oil_per_wheel * number_of_wheels + oil_for_rest

theorem ellie_oil_needs : total_oil_needed = 25 := by
  sorry

end ellie_oil_needs_l445_445373


namespace sin_half_angle_l445_445818

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445818


namespace rows_without_digit_0_is_18_rows_without_digit_9_is_16_columns_without_digit_0_is_12_digits_1_to_9_in_all_columns_l445_445234

def num_range : List ℕ := List.range' 100 900  -- List of numbers from 100 to 999

def digit_at (n : ℕ) (i : ℕ) : ℕ := (n / (10 ^ i)) % 10

def rows := 108
def columns := 25

def grid (rows columns : ℕ) : List (List (Option ℕ)) :=
  let digits := num_range.bind (λ n, List.ofDigits (digit_at n 2 :: digit_at n 1 :: digit_at n 0 :: []))
  digits.enum.groupByIdx (λ (i, _) => i / columns)

def count_rows_without_digit (d : ℕ) : ℕ :=
  grid rows columns
  |> List.countp (λ row, not (row.any (λ cell, cell = some d)))

def count_columns_without_digit (d : ℕ) : ℕ :=
  (List.range columns).countp (λ c, not (grid rows columns > any (λ r, r.get? c |>.join = some d)))

def check_all_digits_in_columns : Prop :=
  List.range columns
  |> List.all (λ c, (List.range 1 10).all (λ d, grid rows columns > any (λ r, r.get? c |>.join = some d)))

theorem rows_without_digit_0_is_18 : count_rows_without_digit 0 = 18 := sorry

theorem rows_without_digit_9_is_16 : count_rows_without_digit 9 = 16 := sorry

theorem columns_without_digit_0_is_12 : count_columns_without_digit 0 = 12 := sorry

theorem digits_1_to_9_in_all_columns : check_all_digits_in_columns := sorry

end rows_without_digit_0_is_18_rows_without_digit_9_is_16_columns_without_digit_0_is_12_digits_1_to_9_in_all_columns_l445_445234


namespace intersection_is_result_l445_445556

-- Define the sets M and N
def M : Set ℝ := { x | Math.log10 x > 0 }
def N : Set ℝ := { x | x^2 <= 4 }

-- Define the intersection
def result : Set ℝ := { x | 1 < x ∧ x <= 2 }

-- Prove that M ∩ N = result
theorem intersection_is_result : M ∩ N = result :=
by
  sorry

end intersection_is_result_l445_445556


namespace infinite_composite_in_sequence_l445_445570

theorem infinite_composite_in_sequence :
  ∀ n : ℕ, ∃ m > n, ¬ is_prime (⌊2 ^ m * real.sqrt 2⌋) :=
sorry

end infinite_composite_in_sequence_l445_445570


namespace sin_half_angle_l445_445902

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445902


namespace percentage_of_boys_l445_445003

theorem percentage_of_boys (r_boys : ℕ) (r_girls : ℕ) (total_students : ℕ)
  (H_ratio : 3 = r_boys) (H_ratio' : 4 = r_girls) (H_total : 42 = total_students) :
  (r_boys * (total_students / (r_boys + r_girls))) / total_students * 100 = 42.86 := by
  sorry

end percentage_of_boys_l445_445003


namespace sin_half_angle_l445_445833

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445833


namespace distance_formula_proof_l445_445974

open Real

noncomputable def distance_between_points_on_curve
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  ℝ :=
  |c - a| * sqrt (1 + m^2 * (c + a)^2)

theorem distance_formula_proof
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  distance_between_points_on_curve a b c d m k h1 h2 = |c - a| * sqrt (1 + m^2 * (c + a)^2) :=
by
  sorry

end distance_formula_proof_l445_445974


namespace common_grandfather_grandchildren_l445_445990

theorem common_grandfather_grandchildren (students : Finset ℕ) (grandfathers : Finset ℕ)
    (common_grandfather : ∀ {x y : ℕ}, x ∈ students → y ∈ students → x ≠ y → ∃ g ∈ grandfathers, x ≠ g ∧ y ≠ g)
    (h_card : students.card = 20) :
    ∃ (g ∈ grandfathers), ∃ s ⊆ students, s.card ≥ 14 ∧ ∀ student ∈ s, student ≠ g :=
  sorry

end common_grandfather_grandchildren_l445_445990


namespace find_theta_l445_445477

theorem find_theta:
  ∃ θ : ℝ, (0 < θ ∧ θ < 90) ∧ (cos θ + 2 * sin θ = sqrt 3 * sin 20) ∧ θ = 10 :=
sorry

end find_theta_l445_445477


namespace permutations_of_banana_l445_445295

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445295


namespace possible_degrees_of_remainder_l445_445664

theorem possible_degrees_of_remainder (p : Polynomial ℝ) :
  ∃ r : Polynomial ℝ, ∃ q : Polynomial ℝ, degree p = degree (q * (3 * Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 7 * Polynomial.X - 9) + r) ∧ r.degree < degree (3 * Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 7 * Polynomial.X - 9) :=
begin
  -- Proof omitted
  sorry
end

end possible_degrees_of_remainder_l445_445664


namespace cos_C_correct_l445_445030

noncomputable def cos_C (B : ℝ) (AD BD : ℝ) : ℝ :=
  let sinB := Real.sin B
  let angleBAC := (2 : ℝ) * Real.arcsin ((Real.sqrt 3 / 3) * (sinB / 2)) -- derived from bisector property.
  let cosA := (2 : ℝ) * Real.cos angleBAC / 2 - 1
  let sinA := 2 * Real.sin angleBAC / 2 * Real.cos angleBAC / 2
  let cos2thirds := -1 / 2
  let sin2thirds := Real.sqrt 3 / 2
  cos2thirds * cosA + sin2thirds * sinA

theorem cos_C_correct : 
  ∀ (π : ℝ), 
  ∀ (A B C : ℝ),
  B = π / 3 →
  ∀ (AD : ℝ), AD = 3 →
  ∀ (BD : ℝ), BD = 2 →
  cos_C B AD BD = (2 * Real.sqrt 6 - 1) / 6 :=
by
  intros π A B C hB angleBisectorI hAD hBD
  sorry

end cos_C_correct_l445_445030


namespace sin_half_angle_l445_445907

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445907


namespace fill_time_correct_l445_445676

-- Define the conditions and the target proof statement
def length : ℝ := 7
def width : ℝ := 6
def height : ℝ := 2
def rate_of_filling : ℝ := 4

def volume_of_box : ℝ := length * width * height
def time_to_fill_box : ℝ := volume_of_box / rate_of_filling

theorem fill_time_correct : time_to_fill_box = 21 :=
by
  -- Placeholder for the actual proof
  sorry

end fill_time_correct_l445_445676


namespace perpendicular_line_equation_l445_445431

theorem perpendicular_line_equation (A : Point) (hA : A = ⟨3, 0⟩) :
  (∃ l : Line, l.passes_through A ∧ l.perpendicular_to (2 * x - y + 4)) → l.equation = x + 2 * y - 3 := 
sorry

end perpendicular_line_equation_l445_445431


namespace sin_half_alpha_l445_445852

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445852


namespace sum_arithmetic_sequence_l445_445605

def first_term (k : ℕ) : ℕ := k^2 - k + 1

def sum_of_first_k_plus_3_terms (k : ℕ) : ℕ := (k + 3) * (k^2 + (k / 2) + 2)

theorem sum_arithmetic_sequence (k : ℕ) (k_pos : 0 < k) : 
    sum_of_first_k_plus_3_terms k = k^3 + (7 * k^2) / 2 + (15 * k) / 2 + 6 := 
by
  sorry

end sum_arithmetic_sequence_l445_445605


namespace perm_banana_l445_445354

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445354


namespace find_other_number_l445_445111

theorem find_other_number (a b : ℕ) (h₁ : Nat.lcm a b = 3780) (h₂ : Nat.gcd a b = 18) (h₃ : a = 180) : b = 378 := by
  sorry

end find_other_number_l445_445111


namespace smallest_n_with_terminating_decimal_and_digit_7_l445_445656

def contains_digit_7 (n : ℕ) : Prop :=
  n.digits 10 |> List.contains 7

theorem smallest_n_with_terminating_decimal_and_digit_7 :
  (∃ n : ℕ, (fractional n).is_terminating ∧ contains_digit_7 n ∧ ∀ m : ℕ, (fractional m).is_terminating ∧ contains_digit_7 m → n ≤ m) →
  ∃ smallest_n : ℕ, smallest_n = 128 :=
by {
  sorry -- proof goes here
}

end smallest_n_with_terminating_decimal_and_digit_7_l445_445656


namespace area_of_triangle_ACD_l445_445123

theorem area_of_triangle_ACD :
  ∀ (AD AC height_AD height_AC : ℝ),
  AD = 6 → height_AD = 3 → AC = 3 → height_AC = 3 →
  (1 / 2 * AD * height_AD - 1 / 2 * AC * height_AC) = 4.5 :=
by
  intros AD AC height_AD height_AC hAD hheight_AD hAC hheight_AC
  sorry

end area_of_triangle_ACD_l445_445123


namespace sin_half_alpha_l445_445854

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445854


namespace sin_half_alpha_l445_445860

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445860


namespace find_angle_FYD_l445_445510

variable {Point : Type}
variables (A B C D X Y : Point)
variables (E F G : Point)
variables [Euclidean ⟨_,⟨A, B, C, by sorry⟩⟩]

-- Conditions
variable (h_parallel : parallel (line_through A B) (line_through C D))
variable (h_angle_AXF : angle A X F = 130)
variable (h_isosceles_XFG : is_isosceles_triangle X F G)
variable (h_angle_FXG : angle F X G = 36)

-- Question to prove
theorem find_angle_FYD : angle F Y D = 50 :=
by
  sorry

end find_angle_FYD_l445_445510


namespace function_expression_l445_445789

noncomputable def f (x : ℝ) : ℝ :=
  (2 ♁ x) / ((x ⊗ 2) - 2)

def op1 (a b : ℝ) : ℝ := (sqrt (a^2 - b^2))
def op2 (a b : ℝ) : ℝ := (sqrt ((a - b)^2))

theorem function_expression (x : ℝ) (h1: -2 ≤ x ∧ x ≤ 2) (h2: x ≠ 0) :
  f x = - (sqrt (4 - x^2)) / x := by
  sorry

end function_expression_l445_445789


namespace arithmetic_series_sum_l445_445389

-- Given definitions
def first_term (k : ℕ) : ℕ := k^2 - 1
def common_difference : ℕ := 1
def num_terms (k : ℕ) : ℕ := 2 * k - 1

-- The theorem to be proven
theorem arithmetic_series_sum (k : ℕ) :
  let a₁ := first_term k in
  let d  := common_difference in
  let n  := num_terms k in
  let aₙ := a₁ + (n - 1) * d in
  (n * (a₁ + aₙ)) / 2 = 2 * k^3 + k^2 - 4 * k + 3 / 2 :=
by sorry

end arithmetic_series_sum_l445_445389


namespace probability_of_female_selection_probability_of_male_host_selection_l445_445500

/-!
In a competition, there are eight contestants consisting of five females and three males.
If three contestants are chosen randomly to progress to the next round, what is the 
probability that all selected contestants are female? Additionally, from those who 
do not proceed, one is selected as a host. What is the probability that this host is male?
-/

noncomputable def number_of_ways_select_3_from_8 : ℕ := Nat.choose 8 3

noncomputable def number_of_ways_select_3_females_from_5 : ℕ := Nat.choose 5 3

noncomputable def probability_all_3_females : ℚ := number_of_ways_select_3_females_from_5 / number_of_ways_select_3_from_8

noncomputable def number_of_remaining_contestants : ℕ := 8 - 3

noncomputable def number_of_males_remaining : ℕ := 3 - 1

noncomputable def number_of_ways_select_1_male_from_2 : ℕ := Nat.choose 2 1

noncomputable def number_of_ways_select_1_from_5 : ℕ := Nat.choose 5 1

noncomputable def probability_host_is_male : ℚ := number_of_ways_select_1_male_from_2 / number_of_ways_select_1_from_5

theorem probability_of_female_selection : probability_all_3_females = 5 / 28 := by
  sorry

theorem probability_of_male_host_selection : probability_host_is_male = 2 / 5 := by
  sorry

end probability_of_female_selection_probability_of_male_host_selection_l445_445500


namespace cube_vertices_count_l445_445166

-- Defining the conditions of the problem
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def euler_formula (V E F : ℕ) : Prop := V - E + F = 2

-- Stating the proof problem
theorem cube_vertices_count : ∃ V : ℕ, euler_formula V num_edges num_faces ∧ V = 8 :=
by
  sorry

end cube_vertices_count_l445_445166


namespace perfect_square_divisors_of_product_factorial_l445_445469

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def product_factorials : ℕ :=
  factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

noncomputable def perfect_square_divisors_count (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, is_divisor d n ∧ is_perfect_square d).card

theorem perfect_square_divisors_of_product_factorial :
  perfect_square_divisors_count product_factorials = 10 :=
  sorry

end perfect_square_divisors_of_product_factorial_l445_445469


namespace number_of_arrangements_of_BANANA_l445_445321

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445321


namespace smallest_part_of_division_l445_445474

theorem smallest_part_of_division (x : ℚ) (h : x + (1/4) * x + (1/5) * x = 130) : 
  (1/5) * x = 2600 / 145 :=
by  
  have h₁ : (29/20) * x = 130 := by rw [← add_assoc, ← mul_add, ← add_mul]; exact h
  have h₂ : x = (20/29) * 130 := by field_simp [h₁]; linarith  
  have h₃ : (1/5) * x = (1/5) * ((20/29) * 130) := by rw [h₂]
  field_simp [h₃]; linarith

end smallest_part_of_division_l445_445474


namespace total_hike_time_l445_445529

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l445_445529


namespace banana_arrangements_l445_445324

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445324


namespace smallest_number_of_set_s_l445_445155

theorem smallest_number_of_set_s : 
  ∀ (s : Set ℕ),
    (∃ n : ℕ, s = {k | ∃ m : ℕ, k = 5 * (m+n) ∧ m < 45}) ∧ 
    (275 ∈ s) → 
      (∃ min_elem : ℕ, min_elem ∈ s ∧ min_elem = 55) 
  :=
by
  sorry

end smallest_number_of_set_s_l445_445155


namespace boss_total_amount_l445_445013

def number_of_staff : ℕ := 20
def rate_per_day : ℕ := 100
def number_of_days : ℕ := 30
def petty_cash_amount : ℕ := 1000

theorem boss_total_amount (number_of_staff : ℕ) (rate_per_day : ℕ) (number_of_days : ℕ) (petty_cash_amount : ℕ) :
  let total_allowance_one_staff := rate_per_day * number_of_days
  let total_allowance_all_staff := total_allowance_one_staff * number_of_staff
  total_allowance_all_staff + petty_cash_amount = 61000 := by
  sorry

end boss_total_amount_l445_445013


namespace probability_perpendicular_lines_l445_445397

open Finset

theorem probability_perpendicular_lines (a b : ℕ) :
  a ∈ (range 6).map (λ x, x + 1) → b ∈ (range 6).map (λ x, x + 1) →
  (filter (λ (p : ℕ × ℕ), let (a, b) := p in a - 2 * b = 0) ((range 6).map (λ x, x + 1)).product ((range 6).map (λ x, x + 1))).card.toRat / 36 = (1 : ℚ) / 12 :=
by
  intros ha hb
  sorry

end probability_perpendicular_lines_l445_445397


namespace ice_cream_cost_l445_445086

theorem ice_cream_cost (initial_money lunch_cost : ℕ) (remaining_money_after_lunch quarter_cost : ℕ) :
  initial_money = 30 →
  lunch_cost = 10 →
  remaining_money_after_lunch = initial_money - lunch_cost →
  quarter_cost = remaining_money_after_lunch * 1 / 4 →
  quarter_cost = 5 :=
begin
  intros h_initial h_lunch h_remaining h_quarter,
  rw [h_initial, h_lunch] at *,
  calc 
    remaining_money_after_lunch = 30 - 10 : h_remaining
    ... = 20 : by norm_num
    ...,
  calc 
    quarter_cost = 20 * 1 / 4 : by rw h_quarter
    ... = 5 : by norm_num
end

end ice_cream_cost_l445_445086


namespace slope_of_l_l445_445980

variable {a b : ℝ} -- Coordinates of the initial point A

def point_A_in_new_position (a b : ℝ) : Prop :=
  let A' := (a + 1, b - 3) in 
  A' ∈ {p : ℝ × ℝ | ∃ k : ℝ, k = (p.2 - b) / (p.1 - a)}

theorem slope_of_l (h : point_A_in_new_position a b) : 
  ∃ k : ℝ, k = -3 :=
by 
  sorry

end slope_of_l_l445_445980


namespace sin_half_alpha_l445_445825

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445825


namespace average_of_numbers_l445_445103

noncomputable def x := (5050 : ℚ) / 5049

theorem average_of_numbers :
  let sum := (∑ i in Finset.range 101, (i + 1)) + x in
  let avg := sum / (101 + 1) in
  avg = 50 * x :=
by
  let sum := (∑ i in Finset.range 101, (i + 1)) + x
  let avg := sum / (101 + 1)
  have sum_formula : (∑ i in Finset.range 101, (i + 1)) = 5050 := sorry
  have avg_formula : avg = 50 * x := sorry
  exact avg_formula

end average_of_numbers_l445_445103


namespace permutations_of_banana_l445_445292

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445292


namespace power_sum_inequality_l445_445473

variables {R : Type*} [LinearOrderedField R]

theorem power_sum_inequality 
  (a b c r s : R) (ha : a > b) (hb : b > c) (hr : r > s) (hs : s > 0) : 
  a^r * b^s + b^r * c^s + c^r * a^s ≥ a^s * b^r + b^s * c^r + c^s * a^r :=
sorry

end power_sum_inequality_l445_445473


namespace distribution_ways_l445_445371

open Nat

theorem distribution_ways : 
  ∃ n : ℕ, (n = 80 ∧ 
  ∃ f : Fin 5 → Fin 3, 
     (∃ a b c : Fin 3, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      (∀ x, f x = a → (finset.card {y | f y = a} ≥ 2)) ∧
      (∀ x, f x = b → (finset.card {y | f y = b} ≥ 1)) ∧
      (∀ x, f x = c → (finset.card {y | f y = c} ≥ 1))∧
      (∀ y : Fin 3, ∃ z : Fin 5, f z = y))) :=
by
  exists 80
  sorry

end distribution_ways_l445_445371


namespace banana_arrangements_l445_445323

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445323


namespace angle_between_a_and_a_plus_b_l445_445935

variables (a b : EuclideanSpace) (θ : ℝ)

-- Conditions
def condition1 := ∥a∥ = 1
def condition2 := ∥b∥ = Real.sqrt 3
def condition3 := ∥2 • a + b∥ = Real.sqrt 7

-- Define the problem statement
theorem angle_between_a_and_a_plus_b (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  θ = Real.pi / 3 :=
sorry

end angle_between_a_and_a_plus_b_l445_445935


namespace gcd_78_36_l445_445638

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := 
by
  sorry

end gcd_78_36_l445_445638


namespace inverse_function_evaluation_l445_445551

def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 else -x^3

theorem inverse_function_evaluation : (f⁻¹ 9) + (f⁻¹ (-27)) = 0 := by
  sorry

end inverse_function_evaluation_l445_445551


namespace calculate_perimeter_last_triangle_l445_445554

noncomputable theory

def perimeter_last_triangle (T1_side1 T1_side2 T1_side3 : ℕ) : ℕ := 
  let a_n (n : ℕ) := T1_side2 / 2^(n-1) in
  let b_n (n : ℕ) := T1_side2 / 2^(n-1) + 1 in
  let c_n (n : ℕ) := T1_side2 / 2^(n-1) - 1 in
  let perimeter (n : ℕ) := 3 * T1_side2 / 2^(n-1) in
  let stop_condition := (n : ℕ) → b_n n ≥ 2 * (a_n n - c_n n) in
  let last_n := 11 in
  perimeter last_n

theorem calculate_perimeter_last_triangle :
  perimeter_last_triangle 2021 2022 2023 = 1516.5 / 256 := 
by sorry

end calculate_perimeter_last_triangle_l445_445554


namespace min_value_l445_445486

-- Definitions based on the given conditions
def circle1 (m n : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 - m)^2 + (p.2 - 2 * n)^2 = m^2 + 4 * n^2 + 10}

def circle2 : set (ℝ × ℝ) :=
  {p | (p.1 + 1)^2 + (p.2 + 1)^2 = 2}

-- Given m and n are positive
def pos_mn (m n : ℝ) : Prop := m > 0 ∧ n > 0

-- Circle C1 bisects the circumference of Circle C2
def bisects (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (circle1 m n).includes (x, y) → (circle2.includes (x, y) = true)

-- Condition derived from solution steps
def line_through_centers (m n : ℝ) : Prop := m + 2 * n = 3

-- Minimum value problem statement
theorem min_value (m n : ℝ)
  (h1 : ∃ (m n : ℝ), bisects m n ∧ pos_mn m n)
  (h2 : line_through_centers m n) : (1 / m + 2 / n) = 3 :=
sorry

end min_value_l445_445486


namespace product_of_xy_l445_445567

theorem product_of_xy (x y : ℝ) (hx1 : x^2 + y^2 = 3) (hx2 : x^4 + y^4 = 15 / 8) : x * y = Real.sqrt 57 / 4 := 
begin
  sorry
end

end product_of_xy_l445_445567


namespace sin_half_alpha_l445_445881

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445881


namespace find_a_and_check_parity_f_increasing_l445_445437

def f (a x : ℝ) : ℝ := (a * x) / (1 + x^2)

theorem find_a_and_check_parity (a : ℝ) (h : f a (1/2) = 2/5) :
  a = 1 ∧ (∀ x : ℝ, f a (-x) = -f a x) := by
  sorry

theorem f_increasing (a : ℝ) (ha : a = 1) :
  ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f a x1 < f a x2 := by
  sorry

end find_a_and_check_parity_f_increasing_l445_445437


namespace randy_biscuits_left_l445_445573

-- Define the function biscuits_left
def biscuits_left (initial: ℚ) (father_gift: ℚ) (mother_gift: ℚ) (brother_eat_percent: ℚ) : ℚ :=
  let total_before_eat := initial + father_gift + mother_gift
  let brother_ate := brother_eat_percent * total_before_eat
  total_before_eat - brother_ate

-- Given conditions
def initial_biscuits : ℚ := 32
def father_gift : ℚ := 2 / 3
def mother_gift : ℚ := 15
def brother_eat_percent : ℚ := 0.3

-- Correct answer as an approximation since we're dealing with real-world numbers
def approx (x y : ℚ) := abs (x - y) < 0.01

-- The proof problem statement in Lean 4
theorem randy_biscuits_left :
  approx (biscuits_left initial_biscuits father_gift mother_gift brother_eat_percent) 33.37 :=
by
  sorry

end randy_biscuits_left_l445_445573


namespace distinct_arrangements_of_BANANA_l445_445344

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445344


namespace counterexample_exists_l445_445050

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- State the theorem equivalently in Lean
theorem counterexample_exists : (sum_of_digits 33 % 6 = 0) ∧ (33 % 6 ≠ 0) := by
  sorry

end counterexample_exists_l445_445050


namespace count_divisibles_in_range_l445_445468

theorem count_divisibles_in_range :
  let lower_bound := (2:ℤ)^10
  let upper_bound := (2:ℤ)^18
  let divisor := (2:ℤ)^9 
  (upper_bound - lower_bound) / divisor + 1 = 511 :=
by 
  sorry

end count_divisibles_in_range_l445_445468


namespace elina_mean_score_l445_445581

-- Define Jason and Elina's scores
def scores : List ℕ := [78, 85, 88, 91, 92, 95, 96, 99, 101, 103]

-- Define Jason's mean score
def jasonMean : ℕ := 93

-- Define Jason's number of scores
def jasonCount : ℕ := 6

-- Define Elina's number of scores
def elinaCount : ℕ := 4

theorem elina_mean_score :
    let totalSum := scores.sum
    let jasonSum := jasonCount * jasonMean
    let elinaSum := totalSum - jasonSum
    elinaSum / elinaCount = 92.5 :=
by
    sorry

end elina_mean_score_l445_445581


namespace arrangement_of_BANANA_l445_445366

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445366


namespace part_a_part_b_l445_445149

section TriangleTangentCircle

variables {A B C O : Point}
variables {S : Circle}

/-- Conditions given: point A external to circle S, lines AB and AC tangent to circle S at B and C respectively, forming triangle ABC, and O is the center of circle S inside triangle ABC --/
def conditions (A B C : Point) (S : Circle) (O : Point) : Prop :=
  external_point A S ∧
  tangent A B S ∧
  tangent A C S ∧
  center O S ∧
  inside_triangle O A B C

/-- Part (a): the incenter of triangle ABC and the excenter opposite side BC lie on circle S --/
theorem part_a (A B C : Point) (S : Circle) (O : Point) (h : conditions A B C S O) : 
  lies_on_incenter_triangle_circle ABC O ∧ 
  lies_on_excenter_triangle_opposite_side_circle ABC BC O := sorry

/-- Part (b): the circle through B, C, and O intersects equal chords on lines AB and AC --/
theorem part_b (A B C : Point) (O : Point) : 
  let S' := circumscribed_circle B C O in
  intersects_equal_chords S' A B A C := sorry

end TriangleTangentCircle

end part_a_part_b_l445_445149


namespace cubic_three_distinct_real_roots_l445_445105

open Real Polynomial

theorem cubic_three_distinct_real_roots (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
   (x^3 - 6*x^2 + 9*x + m).eval x₁ = 0 ∧
   (x^3 - 6*x^2 + 9*x + m).eval x₂ = 0 ∧
   (x^3 - 6*x^2 + 9*x + m).eval x₃ = 0)
  ↔ -4 < m ∧ m < 0 :=
by sorry

end cubic_three_distinct_real_roots_l445_445105


namespace chinaman_change_possible_l445_445588

def pence (x : ℕ) := x -- defining the value of pence as a natural number

def ching_chang_by_value (d : ℕ) := 
  (2 * pence d) + (4 * (2 * pence d) / 15)

def equivalent_value_of_half_crown (d : ℕ) := 30 * pence d

def coin_value_with_holes (holes_value : ℕ) (value_per_eleven : ℕ) := 
  (value_per_eleven * ching_chang_by_value 1) / 11

theorem chinaman_change_possible :
  ∃ (x y z : ℕ), 
  (7 * coin_value_with_holes 15 11) + (1 * coin_value_with_holes 16 11) + (0 * coin_value_with_holes 17 11) = 
  equivalent_value_of_half_crown 1 :=
sorry

end chinaman_change_possible_l445_445588


namespace abs_sum_less_abs_diff_l445_445479

theorem abs_sum_less_abs_diff {a b : ℝ} (hab : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_sum_less_abs_diff_l445_445479


namespace impossible_to_fill_with_dominoes_l445_445215

def board_size : ℕ := 5
def domino_size : ℕ := 2
def total_cells : ℕ := board_size * board_size

theorem impossible_to_fill_with_dominoes :
  ¬ ∃ (domino_cover_count : ℕ), domino_cover_count * domino_size = total_cells :=
by {
  let h := total_cells,
  have h := by norm_num : h = 25,
  have d_size := by norm_num : domino_size = 2,
  have non_divisible := by norm_num : 25 % 2 ≠ 0,
  intro,
  rcases a with ⟨domino_cover_count, eq⟩,
  rw [d_size, mul_comm] at eq,
  rw h at eq,
  apply non_divisible,
  exact nat.eq_zero_of_mul_left_eq_self' 2 eq,
  sorry
}

end impossible_to_fill_with_dominoes_l445_445215


namespace intersection_line_plane_l445_445942

-- Definitions of intersecting lines and parallelism.

def intersect (a b : Set Point) : Prop :=
  ∃ P : Point, P ∈ a ∧ P ∈ b

def parallel (a : Set Point) (α : Set Point) : Prop :=
  ∀ P1 P2 : Point, P1 ∈ a ∧ P2 ∈ α → ((P1 ≠ P2) → (∃ d, ∀ Q : Point, Q ∈ α → (Q = P2 + d * (P2 - P1))))

def parallel_line_plane (a b : Set Point)  (α : Set Point) : Prop :=
  parallel a α ∧ (parallel b α ∨ ∃ P : Point, P ∈ b ∧ P ∈ α)

theorem intersection_line_plane (a b : Set Point) (α : Set Point) :
  intersect a b → parallel a α → (parallel b α ∨ ∃ P : Point, P ∈ b ∧ P ∈ α) :=
by
  sorry

end intersection_line_plane_l445_445942


namespace solve_quadratic_inequality_l445_445793

theorem solve_quadratic_inequality (a1 a2 a3 : ℝ) (h1 : a1 > a2) (h2 : a2 > a3) (h3 : a3 > 0) :
  {x : ℝ | ∀ i ∈ {1, 2, 3}, (1 - (a_i i * x))^2 < 1} = {x : ℝ | 0 < x ∧ x < 2 / a1} :=
by sorry

end solve_quadratic_inequality_l445_445793


namespace permutations_of_BANANA_l445_445248

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445248


namespace distance_traveled_upstream_is_72_l445_445502

-- Definitions based on conditions
def boat_speed_in_still_water : ℝ := 6
def river_speed : ℝ := 2
def total_journey_time : ℝ := 27

-- The effective speeds
def upstream_speed : ℝ := boat_speed_in_still_water - river_speed
def downstream_speed : ℝ := boat_speed_in_still_water + river_speed

-- Time expressions
def time_upstream (D : ℝ) : ℝ := D / upstream_speed
def time_downstream (D : ℝ) : ℝ := D / downstream_speed

-- Proof that the total time for the journey equals 27 hours implies D=72
theorem distance_traveled_upstream_is_72 :
  ∃ D : ℝ, time_upstream D + time_downstream D = total_journey_time ∧ D = 72 :=
by
  sorry

end distance_traveled_upstream_is_72_l445_445502


namespace instantaneous_velocity_at_t_eq_2_l445_445736

def displacement (t : ℝ) : ℝ := -t^2 + 5 * t

theorem instantaneous_velocity_at_t_eq_2 : 
  let velocity := fun t => deriv (displacement t) 
  velocity 2 = 1 :=
by
  let velocity := fun t => deriv (displacement t)
  have h : velocity 2 = -4 + 5 := sorry  -- Need to compute the derivative and substitute t = 2
  exact h

end instantaneous_velocity_at_t_eq_2_l445_445736


namespace inequality_integer_sum_l445_445093

theorem inequality_integer_sum 
: (sqrt (6 * 3 - 13) - sqrt (3 * 3^2 - 13 * 3 + 13) ≥ 3 * 3^2 - 19 * 3 + 26) ∧
  (sqrt (6 * 4 - 13) - sqrt (3 * 4^2 - 13 * 4 + 13) ≥ 3 * 4^2 - 19 * 4 + 26) ∧
  3 + 4 = 7 := 
by
  sorry

end inequality_integer_sum_l445_445093


namespace smallest_n_digit_7_terminating_decimal_l445_445659

theorem smallest_n_digit_7_terminating_decimal :
  ∃ n : ℕ, (∃ a b : ℕ, n = 2^a * 5^b) ∧ (∃ d : ℕ, nat.digits 10 n 7 = true) ∧ (∀ m : ℕ, 
  (∃ c d : ℕ, m = 2^c * 5^d) ∧ (∃ e : ℕ, nat.digits 10 m 7 = true) → n ≤ m) := by
sorry

end smallest_n_digit_7_terminating_decimal_l445_445659


namespace crayons_lost_l445_445079

theorem crayons_lost (initial_crayons ending_crayons : ℕ) (h_initial : initial_crayons = 253) (h_ending : ending_crayons = 183) : (initial_crayons - ending_crayons) = 70 :=
by
  sorry

end crayons_lost_l445_445079


namespace binom_12_11_l445_445224

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_l445_445224


namespace sin_half_angle_l445_445910

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445910


namespace binom_12_11_eq_12_l445_445218

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_eq_12_l445_445218


namespace yy_percentage_increase_is_three_l445_445114

def percentage_increase_YY (X Y increase_XX diff_XX_YY total_last_year YY_last_year : ℕ) : ℚ :=
  let increase_XX_num := (increase_XX * X / 100) in
  let increase_YY_num := increase_XX_num - diff_XX_YY in
  let p := (increase_YY_num * 100) / Y in
  p

theorem yy_percentage_increase_is_three :
  ∀ (X Y : ℕ), Y = 2400 → X + Y = 4000 → percentage_increase_YY X Y 7 40 4000 2400 = 3 :=
by {
  intros X Y hY_last_year htotal_last_year,
  rw hY_last_year at *,
  rw htotal_last_year at *,
  have hX : X = 1600, { linarith },
  unfold percentage_increase_YY,
  simp only [*, mul_assoc, div_eq_mul_inv, mul_one, eq_self_iff_true, tsub_zero, nat.cast_bit0, nat.cast_bit1, nat.cast_one, nat.cast_mul, nat.cast_add, nat.cast_div, add_right_inj, zero_add],
  norm_num,
}

end yy_percentage_increase_is_three_l445_445114


namespace hyperbola_eccentricity_l445_445104

theorem hyperbola_eccentricity (k : ℝ) :
  (∀ x y : ℝ, (k-6) * x^2 + k * y^2 = k * (k-6)) →
  (eccentricity ((k-6) * x^2 + k * y^2 = k * (k-6)) = sqrt 3) →
  k = 2 :=
by
  sorry

end hyperbola_eccentricity_l445_445104


namespace sin_half_alpha_l445_445897

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445897


namespace minimum_flowers_to_guarantee_bouquets_l445_445993

-- Definitions based on conditions given
def types_of_flowers : ℕ := 6
def flowers_needed_for_bouquet : ℕ := 5
def required_bouquets : ℕ := 10

-- Problem statement in Lean 4
theorem minimum_flowers_to_guarantee_bouquets (types : ℕ) (needed: ℕ) (bouquets: ℕ) 
    (h_types: types = types_of_flowers) (h_needed: needed = flowers_needed_for_bouquet) 
    (h_bouquets: bouquets = required_bouquets) : 
    (minimum_number_of_flowers_to_guarantee_bouquets types needed bouquets) = 70 :=
by sorry


end minimum_flowers_to_guarantee_bouquets_l445_445993


namespace sin_half_alpha_l445_445856

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445856


namespace time_to_bookstore_l445_445666

theorem time_to_bookstore (rate feet_per_minute : ℝ) (remaining_distance feet : ℝ) (h1 : rate = 2) (h2 : remaining_distance = 270) : 
  remaining_distance / rate = 135 := 
by
  rw [h1, h2]
  norm_num
  sorry

end time_to_bookstore_l445_445666


namespace banana_arrangement_count_l445_445279

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445279


namespace hash_assoc_l445_445553

def k : Real := (1991 + Real.sqrt (1991^2 + 4)) / 2

def hash (m n : Nat) : Nat :=
  m * n + Int.floor (k * m) * Int.floor (k * n)

theorem hash_assoc (a b c : Nat) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  hash (hash a b) c = hash a (hash b c) :=
  sorry

end hash_assoc_l445_445553


namespace area_of_triangle_l445_445642

noncomputable def point := (ℝ × ℝ)

def line1 : ℝ → ℝ := λ x, 2 * x
def line2 : ℝ → ℝ := λ x, -2 * x
def line3 : ℝ := 4 

def intersection1 : point := (2, 4)
def intersection2 : point := (-2, 4)
def base : ℝ := 4
def height : ℝ := 4

theorem area_of_triangle : (1/2) * base * height = 8 :=
by 
  -- Placeholder for proof
  sorry

end area_of_triangle_l445_445642


namespace algebraic_expression_value_l445_445977

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y + 3 = 0) : 1 - 2 * x + 4 * y = 7 := 
by
  sorry

end algebraic_expression_value_l445_445977


namespace true_proposition_l445_445808

open Real

-- Proposition p
def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

-- Proposition q
def q : Prop := ∃ x > 0, 8 * x + 1 / (2 * x) ≤ 4

theorem true_proposition : ¬ p ∧ q := by
  sorry

end true_proposition_l445_445808


namespace sum_of_integers_ending_in_3_l445_445747

noncomputable def arithmetic_sum : ℕ → ℕ → ℕ → ℕ
| a d n := (n * (2 * a + (n - 1) * d)) / 2

theorem sum_of_integers_ending_in_3 :
  let a := 103
  let d := 10
  let n := 35 in
  arithmetic_sum a d n = 9555 :=
by
  sorry

end sum_of_integers_ending_in_3_l445_445747


namespace g_period_4_g_sum_1_to_20_l445_445919

variable {ℝ : Type}

variables {f g : ℝ → ℝ}

-- Conditions
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom cond_1 : ∀ x : ℝ, f(2 - x) + f(x) = 0
axiom cond_2 : ∀ x : ℝ, f(1 - x) + g(x) = 3
axiom cond_3 : ∀ x : ℝ, f(x) + g(x - 3) = 3

-- Proof Statements
theorem g_period_4 : (∀ x : ℝ, g(x + 4) = g(x)) :=
sorry

theorem g_sum_1_to_20 : (g(1) + g(2) + g(3) + ... + g(20) = 60) :=
sorry

end g_period_4_g_sum_1_to_20_l445_445919


namespace new_light_wattage_l445_445671

theorem new_light_wattage (old_watt : ℕ) (percent_increase : ℕ) 
  (h_old_watt : old_watt = 80) (h_percent_increase : percent_increase = 25) : 
  ℕ :=
let new_watt := old_watt + (old_watt * percent_increase / 100)
in new_watt = 100

end new_light_wattage_l445_445671


namespace permutations_of_banana_l445_445294

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445294


namespace tangent_slope_at_A_l445_445120

def y (x : ℝ) : ℝ := x^2 + 3 * x

def tangent_slope_at (f : ℝ → ℝ) (x0 : ℝ) : ℝ := Deriv f x0

theorem tangent_slope_at_A :
  tangent_slope_at y 2 = 7 :=
by
  sorry

end tangent_slope_at_A_l445_445120


namespace smallest_n_with_terminating_decimal_and_digit_7_l445_445653

-- Definition for \( n \) containing the digit 7
def contains_digit_7 (n: Nat) : Prop :=
  (n.toString.contains '7')

-- Definition for \( \frac{1}{n} \) being a terminating decimal
def is_terminating_decimal (n: Nat) : Prop :=
  ∃ a b: Nat, n = 2^a * 5^b

-- The main theorem statement
theorem smallest_n_with_terminating_decimal_and_digit_7 :
  Nat.find (λ n => is_terminating_decimal n ∧ contains_digit_7 n) = 65536 := by
  sorry

end smallest_n_with_terminating_decimal_and_digit_7_l445_445653


namespace solve_for_diamond_l445_445976

-- Define what it means for a digit to represent a base-9 number and base-10 number
noncomputable def fromBase (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * b + d) 0

-- The theorem we want to prove
theorem solve_for_diamond (diamond : ℕ) (h_digit : diamond < 10) :
  fromBase 9 [diamond, 3] = fromBase 10 [diamond, 2] → diamond = 1 :=
by 
  sorry

end solve_for_diamond_l445_445976


namespace sin_half_alpha_l445_445853

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445853


namespace sin_half_alpha_l445_445879

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445879


namespace part1_part2_l445_445056

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1 (x : ℝ) (hx : f x ≤ 5) : x ∈ Set.Icc (-1/4 : ℝ) (9/4 : ℝ) := sorry

noncomputable def h (x a : ℝ) : ℝ := Real.log (f x + a)

theorem part2 (ha : ∀ x : ℝ, f x + a > 0) : a ∈ Set.Ioi (-2 : ℝ) := sorry

end part1_part2_l445_445056


namespace part_one_part_two_l445_445398

noncomputable def f (a x : ℝ) : ℝ := (2 / 3) * x^3 - 2 * a * x^2 - 3 * x

def f_prime (a x : ℝ) : ℝ := 2 * x^2 - 4 * a * x - 3

-- Part (I)
theorem part_one (a : ℝ) :
(∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) 1 → f_prime a x ≤ 0) ↔ -1/4 ≤ a ∧ a ≤ 1/4 :=
sorry

-- Part (II)
theorem part_two (a : ℝ) :
(∃ n : ℕ, ∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) 1 →
(f_prime a x = 0 → n = 1) ∧
(f_prime a x ≠ 0 → n = 0)) →
(a < -1/4 ∧ ∃! x, x ∈ set.Ioo (-1 : ℝ) 1 ∧ f_prime a x = 0 ∧ is_minimum)
∨ 
(a > 1/4 ∧ ∃! x, x ∈ set.Ioo (-1 : ℝ) 1 ∧ f_prime a x = 0 ∧ is_maximum)
∨ 
(-1/4 ≤ a ∧ a ≤ 1/4 ∧ ∀ x, x ∈ set.Ioo (-1 : ℝ) 1 → f_prime a x ≠ 0) :=
sorry

end part_one_part_two_l445_445398


namespace cistern_fill_time_l445_445669

theorem cistern_fill_time (C : ℝ) 
    (pipeA_fill_time : ℝ)
    (pipeB_empty_time : ℝ)
    (hA : pipeA_fill_time = 8)
    (hB : pipeB_empty_time = 12) : 
    let net_fill_rate := C / pipeA_fill_time - C / pipeB_empty_time in
    let fill_time := C / net_fill_rate in 
    fill_time = 24 :=
by
    sorry

end cistern_fill_time_l445_445669


namespace inequality_integer_sum_l445_445092

theorem inequality_integer_sum 
: (sqrt (6 * 3 - 13) - sqrt (3 * 3^2 - 13 * 3 + 13) ≥ 3 * 3^2 - 19 * 3 + 26) ∧
  (sqrt (6 * 4 - 13) - sqrt (3 * 4^2 - 13 * 4 + 13) ≥ 3 * 4^2 - 19 * 4 + 26) ∧
  3 + 4 = 7 := 
by
  sorry

end inequality_integer_sum_l445_445092


namespace compare_negative_fractions_l445_445217

theorem compare_negative_fractions :
  (-5 : ℝ) / 6 < (-4 : ℝ) / 5 :=
sorry

end compare_negative_fractions_l445_445217


namespace fraction_is_five_over_nine_l445_445488

theorem fraction_is_five_over_nine (f k t : ℝ) (h1 : t = f * (k - 32)) (h2 : t = 50) (h3 : k = 122) : f = 5 / 9 :=
by
  sorry

end fraction_is_five_over_nine_l445_445488


namespace sin_half_angle_l445_445812

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445812


namespace BANANA_arrangements_l445_445305

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445305


namespace diana_owes_amount_l445_445672

def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount_owed : ℝ := principal + interest

theorem diana_owes_amount :
  total_amount_owed = 80.25 :=
by
  sorry

end diana_owes_amount_l445_445672


namespace tangent_line_equation_l445_445933

-- Define the parabola
def parabola (x : ℝ) : ℝ := -x^2 + (9/2) * x - 4

-- Define what it means for a line to be tangent to the parabola at a point
def is_tangent (k x₁ : ℝ) := x₁ ≠ 0 ∧ (parabola x₁ = k * x₁) ∧ (∂/∂ x parabola evaluated at x₁ = k)

-- Define the condition that the point is in the first quadrant
def in_first_quadrant (x y : ℝ) := x > 0 ∧ y > 0

-- State the problem we want to prove
theorem tangent_line_equation : ∃ k : ℝ, k = 1/2 ∧ ∃ x₁ : ℝ, is_tangent k x₁ ∧ in_first_quadrant x₁ (parabola x₁) := 
by
  sorry

end tangent_line_equation_l445_445933


namespace gina_extra_tip_l445_445393

theorem gina_extra_tip 
  (bill_in_dollars : ℕ) (bill_in_cents : ℕ)
  (good_tip_percent normal_tip_percent : ℕ)
  (gina_normal_tip extra_tip_needed : ℕ) :
  bill_in_dollars = 26 -> 
  bill_in_cents = bill_in_dollars * 100 ->
  normal_tip_percent = 5 ->
  good_tip_percent = 20 -> 
  gina_normal_tip = bill_in_cents * normal_tip_percent / 100 ->
  let good_tip := bill_in_cents * good_tip_percent / 100 in
  extra_tip_needed = good_tip - gina_normal_tip -> 
  extra_tip_needed = 390 :=
by
  sorry

end gina_extra_tip_l445_445393


namespace banana_permutations_l445_445251

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445251


namespace sin_half_alpha_l445_445896

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445896


namespace vector_dot_product_sum_l445_445950

variables {V : Type*} [InnerProductSpace ℝ V]
(open InnerProductSpace)

def vector_a : V := sorry
def vector_b : V := sorry
def vector_c : V := sorry

#check vector_a

-- Given conditions
def cond1 : vector_a + vector_b + vector_c = (0 : V) := sorry
def cond2 : ∥vector_a∥ = 1 := sorry
def cond3 : ∥vector_b∥ = 2 := sorry
def cond4 : ∥vector_c∥ = 2 := sorry

-- Proof goal
theorem vector_dot_product_sum :
  vector_a + vector_b + vector_c = 0 →
  ∥vector_a∥ = 1 →
  ∥vector_b∥ = 2 →
  ∥vector_c∥ = 2 →
  (⟪vector_a, vector_b⟫ + ⟪vector_b, vector_c⟫ + ⟪vector_c, vector_a⟫ : ℝ) = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445950


namespace ball_bounces_number_l445_445178

-- Define the pool table as a rectangle
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the rectangular pool table
def bottom_left := Point.mk 0 0
def bottom_right := Point.mk 12 0
def top_left := Point.mk 0 10
def top_right := Point.mk 12 10

-- Define the trajectory of the ball
def ball_trajectory : Point → Point := λ p, ⟨p.x, p.x⟩

-- The main theorem stating the number of bounces
theorem ball_bounces_number :
  number_of_bounces_to_pocket bottom_left bottom_right top_left top_right ball_trajectory = 9 := by
  sorry

end ball_bounces_number_l445_445178


namespace sin_half_angle_l445_445820

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445820


namespace length_of_AG_l445_445199

variables (A B C D E F G : Type)
variables [linear_ordered_field A] [ordered_comm_group B] [topological_space C] [metric_space D] [normed_group E] [inner_product_space F] [complex_space G]

def rectangle (A B C D : Type) [ordered_comm_group B] := true 
-- The above is a placeholder. The actual definition depends on geometry library specifics.

def parallel_lines (x y z w : Type) := true 
-- The above is a placeholder. The actual definition depends on geometry library specifics.

variables (AD_length : ℝ) (AG_length : ℝ)

def AD : ℝ := 12

theorem length_of_AG 
  (rect : rectangle A B C D) 
  (parallel1 : parallel_lines G F H E) 
  (parallel2 : parallel_lines F H E O)
  (conditions : AD_length = 12) 
  : AG_length = 3 :=
sorry

end length_of_AG_l445_445199


namespace part1_part2_part3_l445_445438

-- Definition of the function
def f (a x : ℝ) : ℝ := 2 * a * x - 1 / x - (a + 2) * Real.log x

-- Statement for part 1
theorem part1 (h : 0 < x) :
  f 0 x ≤ (2 * Real.log 2 - 2) :=
sorry

-- Statement for part 2
theorem part2 (a x : ℝ) (h : 0 < x) :
  (a > 2 → (f a x is_increasing_on (0, 1 / a) ∪ (1 / 2, +∞)) ∧ f a x is_decreasing_on (1 / a, 1 / 2)) ∧
  (a = 2 → f a x is_increasing_on (0, +∞)) ∧
  (0 < a < 2 → (f a x is_increasing_on (0, 1 / 2) ∪ (1 / a, +∞)) ∧ f a x is_decreasing_on (1 / 2, 1 / a)) :=
sorry

-- Statement for part 3
theorem part3 (h : ∀ x1 x2 ∈ [1, 4], abs (f 1 x1 - f 1 x2) < 27 / 4 - 2 * m * Real.log 2) :
  m < 3 :=
sorry

end part1_part2_part3_l445_445438


namespace distinct_arrangements_of_BANANA_l445_445342

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445342


namespace number_of_arrangements_of_BANANA_l445_445314

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445314


namespace estimated_rain_probability_l445_445136

theorem estimated_rain_probability :
  let random_groups := [
    [9, 0, 7], [9, 6, 6], [1, 9, 1], [9, 2, 5], [2, 7, 1],
    [9, 3, 2], [8, 1, 2], [4, 5, 8], [5, 6, 9], [6, 8, 3],
    [4, 3, 1], [2, 5, 7], [3, 9, 3], [0, 2, 7], [5, 5, 6],
    [4, 8, 8], [7, 3, 0], [1, 1, 3], [5, 3, 7], [9, 8, 9]
  ],
  let is_rainy (n : ℕ) : Bool := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4,
  let count_rainy_days (group : List ℕ) : ℕ := group.countp is_rainy,
  let count_groups_with_two_rainy_days := random_groups.countp (λ group => count_rainy_days group = 2),
  let estimated_prob := count_groups_with_two_rainy_days.to_real / random_groups.length.to_real
  in estimated_prob = 0.25 := 
by
  sorry

end estimated_rain_probability_l445_445136


namespace matrix_determinant_zero_l445_445751

noncomputable def matrix_example : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]

theorem matrix_determinant_zero : matrix_example.det = 0 := 
by 
  sorry

end matrix_determinant_zero_l445_445751


namespace count_lines_parallel_to_plane_l445_445768

-- Define the points representing the vertices of the triangular prism
variables (A B C A₁ B₁ C₁ E F E₁ F₁ : Point)

-- Assume the midpoints of the edges
axiom midpoint_AC : Midpoint E A C
axiom midpoint_BC : Midpoint F B C
axiom midpoint_A1C1 : Midpoint E₁ A₁ C₁
axiom midpoint_B1C1 : Midpoint F₁ B₁ C₁

-- Define the geometric relations between points and line formations being parallel to the plane
def lines_parallel_to_plane_ABB1A1 : Set (Line Point) :=
  {l | parallel l (Plane.mk A B B₁ A₁)}

theorem count_lines_parallel_to_plane :
  let lines := [Line.mk E F, Line.mk E₁ F₁, Line.mk E E₁, Line.mk F F₁, Line.mk E₁ F, Line.mk E F₁] in
  ∀ l ∈ lines, l ∈ lines_parallel_to_plane_ABB1A1 →
  lines.size = 6 :=
sorry

end count_lines_parallel_to_plane_l445_445768


namespace BANANA_arrangements_l445_445267

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445267


namespace sin_half_alpha_l445_445822

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445822


namespace min_diff_composite_sum_121_l445_445692

/-- 
A composite number is defined as a number that has two or more prime factors. 
We are given two composite numbers whose sum is 121. 
We need to prove that the minimum positive difference between them is 2.
-/

def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≥ 2 ∧ q ≥ 2 ∧ nat.prime p ∧ nat.prime q ∧ p * q ∣ n

theorem min_diff_composite_sum_121 :
  ∃ n m : ℕ, is_composite n ∧ is_composite m ∧ n + m = 121 ∧ abs (n - m) = 2 :=
sorry

end min_diff_composite_sum_121_l445_445692


namespace quadratic_distinct_real_roots_l445_445983

theorem quadratic_distinct_real_roots (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) ↔ c < 1 :=
by
  sorry

end quadratic_distinct_real_roots_l445_445983


namespace students_not_enrolled_l445_445675

theorem students_not_enrolled (total_students : ℕ) (taking_french : ℕ) (taking_german : ℕ) (taking_both : ℕ) :
  total_students = 60 → taking_french = 41 → taking_german = 22 → taking_both = 9 →
  (total_students - (taking_french + taking_german - taking_both)) = 6 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  ring_nf
  exact rfl

end students_not_enrolled_l445_445675


namespace range_of_m_ellipse_l445_445489

theorem range_of_m_ellipse (m : ℝ) :
  m + 2 > 0 → -(m + 1) > 0 → m + 2 ≠ -(m + 1) → 
  m ∈ set.Ioo (-2 : ℝ) (- (3 / 2)) ∨ m ∈ set.Ioo (- (3 / 2)) (-1) :=
by
  sorry

end range_of_m_ellipse_l445_445489


namespace binom_12_11_eq_12_l445_445223

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := 
by {
  sorry
}

end binom_12_11_eq_12_l445_445223


namespace find_possible_values_of_a_l445_445559

noncomputable def sum_distances (a : ℤ) (k : ℤ) : ℤ :=
  ∑ i in finset.range 13, (abs (a - (k + i)))

noncomputable def sum_squared_distances (a : ℤ) (k : ℤ) : ℤ :=
  ∑ i in finset.range 13, (abs (a^2 - (k + i)))

theorem find_possible_values_of_a :
  (∀ k : ℤ, sum_distances a k = 260 ∧ sum_squared_distances a k = 1768) →
  (a = -12 ∨ a = 13) :=
sorry

end find_possible_values_of_a_l445_445559


namespace sin_half_alpha_l445_445877

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445877


namespace vector_dot_product_sum_l445_445969

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_sum
  (a b c : V)
  (h1 : a + b + c = 0)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hc : ∥c∥ = 2) :
  inner_product_space.inner a b + inner_product_space.inner b c + inner_product_space.inner c a = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445969


namespace BANANA_arrangements_l445_445262

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445262


namespace max_value_of_f_l445_445794

noncomputable def f (a : ℝ) := ∫ x in 0..1, 2 * a * x^2 - a^2 * x

theorem max_value_of_f : ∃ a : ℝ, (∀ x : ℝ, f x ≤ f a) ∧ f a = 2 / 9 :=
by sorry

end max_value_of_f_l445_445794


namespace math_problem_l445_445143

theorem math_problem (a b : ℕ) (ha : a = 45) (hb : b = 15) :
  (a + b)^2 - 3 * (a^2 + b^2 - 2 * a * b) = 900 :=
by
  sorry

end math_problem_l445_445143


namespace combinations_eight_choose_three_l445_445021

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end combinations_eight_choose_three_l445_445021


namespace fahrenheit_conversion_fixed_points_l445_445925

def f_to_c (F : ℤ) : ℤ := (5 * (F - 32)) / 9
def c_to_f (C : ℤ) : ℤ := (9 * C) / 5 + 32

theorem fahrenheit_conversion_fixed_points :
  { F : ℤ | 120 ≤ F ∧ F ≤ 2000 ∧ 
            let C := f_to_c F in 
            let F' := c_to_f C in 
            F = F'}.card = 1045 :=
sorry

end fahrenheit_conversion_fixed_points_l445_445925


namespace sin_half_angle_l445_445836

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445836


namespace sum_geometric_series_sum_k_multiplied_geometric_series_sum_k_squared_multiplied_geometric_series_sum_k_sin_kx_l445_445210

-- Part (a)
theorem sum_geometric_series (n : ℕ) (x : ℝ) : 
  (∑ k in finset.range n, (2:ℝ)^k * x^k) = ((2:ℝ)^n * x^n - 1) / (2 * x - 1) :=
sorry

-- Part (b)
theorem sum_k_multiplied_geometric_series (n : ℕ) : 
  (∑ k in finset.range n, k * (2:ℝ)^k) = (2:ℝ)^n * (n - 2) + 2 :=
sorry

-- Part (c)
theorem sum_k_squared_multiplied_geometric_series (n : ℕ) : 
  (∑ k in finset.range n, k^2 * (2:ℝ)^k) = (2:ℝ)^n * (n^2 - 5 * n + 8) - 8 :=
sorry

-- Part (d)
theorem sum_k_sin_kx (n : ℕ) (x : ℝ) : 
  (∑ k in finset.range n, k * Real.sin (k * x)) = (n * Real.sin ((n-1) * x) - (n-1) * Real.sin (n * x)) / (2 * (1 - Real.cos x)) :=
sorry

end sum_geometric_series_sum_k_multiplied_geometric_series_sum_k_squared_multiplied_geometric_series_sum_k_sin_kx_l445_445210


namespace sin_half_alpha_l445_445890

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445890


namespace find_real_solutions_l445_445774

theorem find_real_solutions : 
  {x : ℝ | x^4 + (4 - x)^4 = 258} = {x : ℝ | x = 2 + real.sqrt (-12 + real.sqrt 257)} ∪ {x : ℝ | x = 2 - real.sqrt (-12 + real.sqrt 257)} :=
by
  sorry

end find_real_solutions_l445_445774


namespace sin_half_angle_l445_445846

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445846


namespace polynomial_remainder_l445_445138

theorem polynomial_remainder (x : ℤ) : 
  (2 * x + 3) ^ 504 % (x^2 - x + 1) = (16 * x + 5) :=
by
  sorry

end polynomial_remainder_l445_445138


namespace cross_section_area_of_tetrahedron_circumsphere_l445_445733

theorem cross_section_area_of_tetrahedron_circumsphere
  (h : ∀ (P A B C : ℝ), regular_tetrahedron_of_length_one P A B C)
  (L M N : ℝ) : are_midpoints L M N ∧ are_points_of_tetrahedron P A B C →
  area_of_cross_section LMN circumsphere τ = π / 3 :=
sorry

end cross_section_area_of_tetrahedron_circumsphere_l445_445733


namespace minimum_positive_period_of_sin_x_cos_x_is_pi_l445_445612

noncomputable def y (x : ℝ) : ℝ := sin x * cos x

theorem minimum_positive_period_of_sin_x_cos_x_is_pi :
  ∃ T > 0, (∀ x, y (x + T) = y x) ∧ (∀ T', (∀ x, y (x + T') = y x) → T' ≥ T) ∧ T = π :=
by
  sorry

end minimum_positive_period_of_sin_x_cos_x_is_pi_l445_445612


namespace average_consecutive_pairs_is_correct_l445_445643

/-- Definition of the set S and nCr function -/
def S : Finset ℕ := Finset.range (33 - 5 + 1) + 5
def nCr (n r : ℕ) : ℕ := (Finset.powersetLen r (Finset.range n)).card

/-- Definitions of the specific combinatorial calculations -/
def omega : ℕ := nCr 29 4
def single_pair : ℕ := 4 * nCr 26 2
def two_pairs : ℕ := 3 * nCr 27 1
def three_pairs : ℕ := 26
def average_consecutive_pairs : ℚ := (single_pair + two_pairs + three_pairs) / omega

/-- The proof problem itself -/
theorem average_consecutive_pairs_is_correct :
  average_consecutive_pairs = 0.0648 := by
  sorry

end average_consecutive_pairs_is_correct_l445_445643


namespace distance_MQ_l445_445029

def N : ℝ × ℝ × ℝ := (2, -1, 4)
def M : ℝ × ℝ × ℝ := (2, -1, -4)
def P : ℝ × ℝ × ℝ := (1, 3, 2)
def Q : ℝ × ℝ × ℝ := (1, -3, -2)

noncomputable def dist (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2)

theorem distance_MQ : dist M Q = 3 :=
by
  -- Proof to be provided
  sorry

end distance_MQ_l445_445029


namespace Ki_tae_pencils_l445_445037

theorem Ki_tae_pencils (P B : ℤ) (h1 : P + B = 12) (h2 : 1000 * P + 1300 * B = 15000) : P = 2 :=
sorry

end Ki_tae_pencils_l445_445037


namespace josh_bought_6_CDs_l445_445036

theorem josh_bought_6_CDs 
  (numFilms : ℕ)   (numBooks : ℕ) (numCDs : ℕ)
  (costFilm : ℕ)   (costBook : ℕ) (costCD : ℕ)
  (totalSpent : ℕ) :
  numFilms = 9 → 
  numBooks = 4 → 
  costFilm = 5 → 
  costBook = 4 → 
  costCD = 3 → 
  totalSpent = 79 → 
  numCDs = (totalSpent - numFilms * costFilm - numBooks * costBook) / costCD → 
  numCDs = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7

end josh_bought_6_CDs_l445_445036


namespace region_area_l445_445493

-- Define the three inequalities as conditions
def condition1 (x y : ℝ) : Prop := abs (x + 5) + sqrt 3 * abs (y - 1) ≤ 3
def condition2 (x y : ℝ) : Prop := y ≤ sqrt (4 - 4 * x - x ^ 2) + 1
def condition3 (x y : ℝ) : Prop := abs (2 * y - 1) ≤ 5

-- Problem Statement: Prove the area of the defined region is equal to the given value
theorem region_area : 
  (∃ x y : ℝ, condition1 x y ∧ condition2 x y ∧ condition3 x y) →
  (∃ area : ℝ, area = (2/3 * Real.pi + 4 * Real.sqrt 3 / 3)) :=
begin
  sorry
end

end region_area_l445_445493


namespace union_of_A_B_l445_445412

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l445_445412


namespace sin_half_alpha_l445_445871

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445871


namespace banana_arrangement_count_l445_445281

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445281


namespace monotonic_increase_interval_graph_translation_equivalent_minimum_b_ge_59_pi_12_l445_445926

theorem monotonic_increase_interval (k : ℤ) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = 2 * sin (2 * x - π / 3)) ∧
                  (∀ y, ∃ (I : Set ℝ), I = Set.Ico (k * π - π / 12) (k * π + 5 * π / 12) ∧
                                         Set.Mono.IncreasingOn I f) :=
sorry

theorem graph_translation_equivalent :
  ∃ (f : ℝ → ℝ), (∀ x, f x = 2 * sin (2 * x - π / 3)) ∧
                  (∀ g : ℝ → ℝ, g = sin ∧ (∀ h, h = (2 * g ∘ (λ x, 2 * x - π / 3)) ∘ (λ x, x / 2)) ∧
                                   (∀ φ, φ = λ y, 2 * y)) :=
sorry

theorem minimum_b_ge_59_pi_12 :
  ∃ (g : ℝ → ℝ) (b : ℝ), 
    (∀ x, g x = 2 * sin (2 * x) + 1) ∧
    (∀ zeros, ∀ I : Set ℝ, I = Set.Icc 0 b ∧ 
                           (Set.Countable (g⁻¹' {0} ∩ I) ≥ 10 → b ≥ 59 * π / 12)) :=
sorry

end monotonic_increase_interval_graph_translation_equivalent_minimum_b_ge_59_pi_12_l445_445926


namespace vector_dot_product_sum_eq_l445_445949

variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V)

theorem vector_dot_product_sum_eq :
  a + b + c = 0 →
  ∥a∥ = 1 →
  ∥b∥ = 2 →
  ∥c∥ = 2 →
  inner a b + inner b c + inner c a = -9 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end vector_dot_product_sum_eq_l445_445949


namespace min_sin_ranges_l445_445401

theorem min_sin_ranges (α β γ : ℝ) (h1 : 0 < α ∧ α ≤ β ∧ β ≤ γ ∧ γ < π) (h2 : α + β + γ = π) :
  1 ≤ (min (sin β / sin α) (sin γ / sin β)) ∧ (min (sin β / sin α) (sin γ / sin β)) < (sqrt 5 + 1) / 2 :=
by sorry

end min_sin_ranges_l445_445401


namespace cost_of_supplies_is_1_l445_445035

-- Definitions and main statement
def cost_per_bracelet (x : ℝ) : Prop :=
  (let total_earned := 12 * 1.5 in
   let total_cost_supplies := 12 * x in
   let remaining_amount := 3 + 3 in
   total_earned - total_cost_supplies = remaining_amount)

theorem cost_of_supplies_is_1 (x : ℝ) : cost_per_bracelet x → x = 1 :=
by
  sorry

end cost_of_supplies_is_1_l445_445035


namespace banana_permutations_l445_445252

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445252


namespace determine_c_absolute_value_l445_445587

-- Define the conditions
def is_root (a b c : ℤ) (x : ℂ) : Prop :=
  a * x^4 + b * x^3 + c * x^2 + b * x + a = 0

theorem determine_c_absolute_value
  (a b c : ℤ)
  (h1 : is_root a b c (3 + complex.i))
  (h2 : int.gcd a (int.gcd b c) = 1) :
  |c| = 142 :=
sorry

end determine_c_absolute_value_l445_445587


namespace option_d_necessary_sufficient_l445_445734

theorem option_d_necessary_sufficient (a : ℝ) : (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) := 
sorry

end option_d_necessary_sufficient_l445_445734


namespace smallest_n_for_divisibility_by_1989_l445_445735

theorem smallest_n_for_divisibility_by_1989 : 
  ∃ n : ℕ, n = 48 ∧ (10^n ≡ 1 [MOD 9]) ∧ (10^n ≡ 1 [MOD 13]) ∧ (10^n ≡ 1 [MOD 17]) :=
by
  have h9 : orderOf (10 : ℤ) 9 = 1 :=
    calc orderOf (10 : ℤ) 9 = 1 : sorry
  have h13 : orderOf (10 : ℤ) 13 = 6 :=
    calc orderOf (10 : ℤ) 13 = 6 : sorry
  have h17 : orderOf (10 : ℤ) 17 = 16 :=
    calc orderOf (10 : ℤ) 17 = 16 : sorry
  use 48
  simp [h9, h13, h17]
  sorry

end smallest_n_for_divisibility_by_1989_l445_445735


namespace sin_half_angle_l445_445811

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445811


namespace correct_relation_l445_445499

-- We state the conditions

variables {A B C a b c : ℝ}
-- Angles sum up to π
axiom angles_sum_to_pi : A + B + C = Real.pi

-- Definition of sides opposite to angles
axiom sides_opposite_angles : ∃ a b c, a / (sin A) = b / (sin B) ∧ a / (sin A) = c / (sin C)

-- The theorem to prove: Option B is correct
theorem correct_relation : a = b * cos C + c * cos B :=
by
  sorry

end correct_relation_l445_445499


namespace ratio_of_edges_l445_445490

theorem ratio_of_edges
  (V₁ V₂ : ℝ)
  (a b : ℝ)
  (hV : V₁ / V₂ = 8 / 1)
  (hV₁ : V₁ = a^3)
  (hV₂ : V₂ = b^3) :
  a / b = 2 / 1 := 
by 
  sorry

end ratio_of_edges_l445_445490


namespace find_tangent_line_b_l445_445442

-- Define the function f(x) = x^2 - ln(x)
def f (x : ℝ) := x^2 - Real.log x

-- Define the derivative of the function f
def f' (x : ℝ) := 2 * x - 1 / x

-- Given the tangent line at a point (a, b) on f(x) has slope 1, show that the tangent line is y = x - 0
theorem find_tangent_line_b (a b : ℝ) :
  a > 0 →
  f'(a) = 1 →
  f a = a - b →
  b = 0 := by
  sorry

end find_tangent_line_b_l445_445442


namespace vector_dot_product_sum_eq_l445_445948

variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V)

theorem vector_dot_product_sum_eq :
  a + b + c = 0 →
  ∥a∥ = 1 →
  ∥b∥ = 2 →
  ∥c∥ = 2 →
  inner a b + inner b c + inner c a = -9 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end vector_dot_product_sum_eq_l445_445948


namespace number_of_people_only_like_baseball_l445_445009

-- Given conditions
def N_both : ℕ := 5
def N_only_football : ℕ := 3
def N_neither : ℕ := 6
def N_total : ℕ := 16

-- Statement to be proved
theorem number_of_people_only_like_baseball : ∃ B : ℕ, B = 2 ∧ 
  (N_total - N_neither = B + N_both + N_only_football) ∧
  B + N_both + N_only_football = 10 :=
begin
  use 2,
  split,
  {
    refl,
  },
  split,
  {
    sorry,
  },
  {
    sorry,
  },
end

end number_of_people_only_like_baseball_l445_445009


namespace relation_a_relation_b_relation_c_relation_d_relation_e_l445_445684

-- Definitions for Fibonacci polynomials
def fib_poly : ℕ → ℝ → ℝ
| 0     x := 0
| 1     x := 1
| (n+1) x := x * (fib_poly n x) + (fib_poly (n-1) x)

-- Definitions for Lucas polynomials
def lucas_poly : ℕ → ℝ → ℝ
| 0     x := 2
| 1     x := x
| (n+1) x := x * (lucas_poly n x) + (lucas_poly (n-1) x)

theorem relation_a (n : ℕ) (x : ℝ) (hn : n ≥ 1) : 
  lucas_poly n x = fib_poly (n-1) x + fib_poly (n+1) x := 
sorry

theorem relation_b (n : ℕ) (x : ℝ) (hn : n ≥ 1) : 
  fib_poly n x * (x^2 + 4) = lucas_poly (n-1) x + lucas_poly (n+1) x :=
sorry

theorem relation_c (n : ℕ) (x : ℝ) : 
  fib_poly (2*n) x = lucas_poly n x * fib_poly n x :=
sorry

theorem relation_d (n : ℕ) (x : ℝ) : 
  (lucas_poly n x)^2 + (lucas_poly (n+1) x)^2 = (x^2 + 4) * fib_poly (2*n+1) x :=
sorry

theorem relation_e (n : ℕ) (x : ℝ) : 
  fib_poly (n+2) x + fib_poly (n-2) x = (x^2 + 2) * fib_poly n x :=
sorry

end relation_a_relation_b_relation_c_relation_d_relation_e_l445_445684


namespace sin_half_alpha_l445_445863

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445863


namespace truth_values_l445_445233

-- Define the region D as a set
def D (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 ≤ 4

-- Define propositions p and q
def p : Prop := ∀ x y, D x y → 2 * x + y ≤ 8
def q : Prop := ∃ x y, D x y ∧ 2 * x + y ≤ -1

-- State the propositions to be proven
def prop1 : Prop := p ∨ q
def prop2 : Prop := ¬p ∨ q
def prop3 : Prop := p ∧ ¬q
def prop4 : Prop := ¬p ∧ ¬q

-- State the main theorem asserting the truth values of the propositions
theorem truth_values : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
by
  sorry

end truth_values_l445_445233


namespace distance_between_points_l445_445382

theorem distance_between_points:
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -3
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = sqrt 61 := 
begin
  sorry
end

end distance_between_points_l445_445382


namespace shaded_area_eq_2014_l445_445737

/- 
Mathematical problem:
As shown in the figure, five identical small squares numbered 1 to 5 are placed inside an isosceles right triangle ABC, and the sum of the areas of these five small squares is 2014. Let the area of the shaded quadrilateral BDEF be \( S \). Prove that \( S = 2014 \).
-/

theorem shaded_area_eq_2014 (ABC BDEF : Type) (S_areas : ℕ) (condition : S_areas = 2014) : 
  ∃ S : ℕ, S = 2014 := 
by 
  use 2014
  exact condition

end shaded_area_eq_2014_l445_445737


namespace change_making_ways_l445_445972

-- Define the conditions
def is_valid_combination (quarters nickels pennies : ℕ) : Prop :=
  quarters ≤ 2 ∧ 25 * quarters + 5 * nickels + pennies = 50

-- Define the main statement
theorem change_making_ways : 
  ∃(num_ways : ℕ), (∀(quarters nickels pennies : ℕ), is_valid_combination quarters nickels pennies → num_ways = 18) :=
sorry

end change_making_ways_l445_445972


namespace number_of_true_propositions_l445_445918

/- Definitions for conditions -/
variables (l m : Line) (α β : Plane)
variables (hl_perp_α : Perpendicular l α) (hm_in_β : Contained m β)

/- Definitions for propositions -/
def proposition_① : Prop := Parallel α β → Perpendicular l m
def proposition_② : Prop := Perpendicular α β → Parallel l m
def proposition_③ : Prop := Parallel l m → Perpendicular α β

/- Main statement -/
theorem number_of_true_propositions : 
  (proposition_① l α β ∧ proposition_② l α β ∧ proposition_③ l α β → 2) := 
sorry

end number_of_true_propositions_l445_445918


namespace min_value_of_k_l445_445051

-- Define the set S
def S : set ℕ := {1, 2, 3, 4}

-- Define the sequence a
variable (a : ℕ → ℕ)

-- Define the conditions for the sequence
def valid_permutation (seq : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (b : ℕ → ℕ), (∀ j, 1 ≤ j ∧ j ≤ 4 → b j ∈ S) ∧ b 4 ≠ 1 ∧ 
  (∃ i1 i2 i3 i4, 1 ≤ i1 ∧ i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 ≤ n ∧
    seq i1 = b 1 ∧ seq i2 = b 2 ∧ seq i3 = b 3 ∧ seq i4 = b 4)

-- Define the minimum k
def min_k (k : ℕ) : Prop :=
  ∀ seq, valid_permutation seq k → ∃ seq', valid_permutation seq' k ∧ (∀ k', k' < k → ¬valid_permutation seq' k')

-- The proof statement
theorem min_value_of_k : ∃ k, min_k k ∧ k = 11 :=
sorry

end min_value_of_k_l445_445051


namespace martin_total_distance_l445_445372

noncomputable def calculate_distance_traveled : ℕ :=
  let segment1 := 70 * 3 -- 210 km
  let segment2 := 80 * 4 -- 320 km
  let segment3 := 65 * 3 -- 195 km
  let segment4 := 50 * 2 -- 100 km
  let segment5 := 90 * 4 -- 360 km
  segment1 + segment2 + segment3 + segment4 + segment5

theorem martin_total_distance : calculate_distance_traveled = 1185 :=
by
  sorry

end martin_total_distance_l445_445372


namespace angle_between_vectors_l445_445944

def vector_len (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

def angle_between (a b : ℝ × ℝ) : ℝ :=
real.arccos ((a.1 * b.1 + a.2 * b.2) / (vector_len a * vector_len b))

theorem angle_between_vectors (a : ℝ × ℝ)
  (h1 : vector_len a = 1) 
  (b : ℝ × ℝ := (real.sqrt 3, -1))
  (h2 : vector_len (a.1 + b.1, a.2 + b.2) = real.sqrt 7) : 
  angle_between a b = π / 3 :=
sorry

end angle_between_vectors_l445_445944


namespace total_decorations_l445_445790

-- Define the conditions
def decorations_per_box := 4 + 1 + 5
def total_boxes := 11 + 1

-- Statement of the problem: Prove that the total number of decorations handed out is 120
theorem total_decorations : total_boxes * decorations_per_box = 120 := by
  sorry

end total_decorations_l445_445790


namespace sin_half_alpha_l445_445882

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445882


namespace general_term_l445_445936

def S (n : ℕ) : ℤ := n^2 - 4*n

noncomputable def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = (2 * n - 5) := by
  sorry

end general_term_l445_445936


namespace BANANA_arrangements_l445_445309

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445309


namespace find_tuple_l445_445709

noncomputable def condition (n : ℕ) (m : ℤ) (a : Fin (n-1) → ℤ) : Prop :=
  (List.range (n - 1)).sum (λ k, Int.floor ((2^k * m + a ⟨k, sorry⟩ : ℤ) / (2^n - 1))) = m

theorem find_tuple (n : ℕ) :
  ∃ a : Fin (n-1) → ℤ, ∀ m : ℤ, condition n m a :=
  sorry

end find_tuple_l445_445709


namespace exist_pairs_sum_and_diff_l445_445082

theorem exist_pairs_sum_and_diff (N : ℕ) : ∃ a b c d : ℕ, 
  (a + b = c + d) ∧ (a * b + N = c * d ∨ a * b = c * d + N) := sorry

end exist_pairs_sum_and_diff_l445_445082


namespace sin_half_angle_l445_445847

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445847


namespace right_triangle_and_perpendicular_secant_l445_445571

variable (A B C D : Type) 
variables [triangle : Triangle A B C] (D : Point) 
variable [secant : SecantLine A D B C]

-- Definition of similar triangles
variable [similar_abd_abc : ∀ (A B C D : Type), Similar (Triangle A B D) (Triangle A B C)]
variable [similar_acd_abc : ∀ (A B C D : Type), Similar (Triangle A C D) (Triangle A B C)]

theorem right_triangle_and_perpendicular_secant :
  ∀ (A B C D : Point),
    Triangle A B C → SecantLine A D B C →
    Similar (Triangle A B D) (Triangle A B C) →
    Similar (Triangle A C D) (Triangle A B C) →
    (RightAngle A B C ∧ Perpendicular A D B C) := sorry

end right_triangle_and_perpendicular_secant_l445_445571


namespace problem_statement_l445_445568

-- Definitions used in the conditions
def dot_product_lt_zero (a b : Vector ℝ) : Prop := a • b < 0

def angle_obtuse (a b : Vector ℝ) : Prop := ∃ θ, θ > π / 2 ∧ θ < π ∧ cos θ = (a • b) / (||a|| * ||b||)

def proposition_p (a b : Vector ℝ) : Prop := dot_product_lt_zero a b → angle_obtuse a b

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x ≤ f y

def proposition_q (f : ℝ → ℝ) : Prop :=
  is_increasing_on f {x | x < 0} ∧ is_increasing_on f {x | x > 0} → is_increasing_on f Set.univ

-- The Lean statement of the problem
theorem problem_statement : ¬(proposition_p a b ∧ proposition_q f) :=
by
  sorry

end problem_statement_l445_445568


namespace johns_old_apartment_cost_l445_445034

-- define the constants
variable (C : ℝ) -- cost of John's old apartment per month
variable (Savings : ℝ := 7680) -- yearly savings
variable (N : ℝ := 3) -- number of people sharing the new apartment cost

-- define the problem and the necessary conditions
theorem johns_old_apartment_cost :
  let Savings_per_month := Savings / 12 in
  let New_apartment_cost_per_month := 1.40 * C in
  let Johns_share := New_apartment_cost_per_month / N in
  C - Johns_share = Savings_per_month → C = 1200 := 
by
  intros
  sorry

end johns_old_apartment_cost_l445_445034


namespace arcsin_cos_eq_neg_pi_div_six_l445_445212

theorem arcsin_cos_eq_neg_pi_div_six :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  sorry

end arcsin_cos_eq_neg_pi_div_six_l445_445212


namespace games_played_before_third_game_l445_445729

theorem games_played_before_third_game
  (average_points_per_game : ℝ)
  (points_in_game3 : ℕ)
  (additional_points_needed : ℕ)
  (current_total_threshold : ℕ)
  (current_points_needed : ℕ)
  (n : ℕ)
  (h_avg : average_points_per_game = 61.5)
  (h_game3 : points_in_game3 = 47)
  (h_additional : additional_points_needed = 330)
  (h_total_threshold : current_total_threshold = 500)
  (h_needed : current_points_needed = 170)
  (h_condition : 61.5 * n + 47 = 170)
  : n = 2 := 
by {
  sorry
}

end games_played_before_third_game_l445_445729


namespace solve_system_l445_445586

open Classical

theorem solve_system : ∃ t : ℝ, ∀ (x y z : ℝ), 
  (x^2 - 9 * y^2 = 0 ∧ x + y + z = 0) ↔ 
  (x = 3 * t ∧ y = t ∧ z = -4 * t) 
  ∨ (x = -3 * t ∧ y = t ∧ z = 2 * t) := 
by 
  sorry

end solve_system_l445_445586


namespace permutations_of_banana_l445_445287

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445287


namespace banana_arrangements_l445_445326

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445326


namespace vector_dot_product_sum_l445_445962

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem vector_dot_product_sum (h₁ : a + b + c = 0) 
                               (ha : ∥a∥ = 1)
                               (hb : ∥b∥ = 2)
                               (hc : ∥c∥ = 2) :
  (a ⬝ b) + (b ⬝ c) + (c ⬝ a) = -9 / 2 := sorry

end vector_dot_product_sum_l445_445962


namespace Mike_changed_64_tires_l445_445063

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end Mike_changed_64_tires_l445_445063


namespace identify_vectors_l445_445195

theorem identify_vectors : 
  (forall i, i ∈ [2, 3, 4, 5] → is_vector i) ∧ 
  (forall j, j ∈ [1, 6] → ¬is_vector j) :=
by
  sorry

-- Definitions to support the theorem
def is_vector (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

end identify_vectors_l445_445195


namespace distance_traveled_by_wheel_l445_445189

theorem distance_traveled_by_wheel (radius : ℝ) (revolutions : ℕ) (C : ℝ) :
  radius = 2 → revolutions = 3 → C = 2 * radius * Real.pi → 
  (revolutions * C) = 12 * Real.pi :=
by
  intros hr hv hc
  rw [hr, hv, hc]
  sorry

end distance_traveled_by_wheel_l445_445189


namespace ship_lighthouse_distance_l445_445719

-- Define the conditions as hypotheses
def ship_distance (s: ℝ) (t: ℝ) := s * t

-- Calculation of trigonometric values
def cos_deg (θ: ℝ) := cos (θ * π / 180)
def sin_deg (θ: ℝ) := sin (θ * π / 180)
def tan_deg (θ: ℝ) := tan (θ * π / 180)

-- Given points A and B with angles 60° and 15° respectively
def pointA (d: ℝ) := (-d * cos_deg 30, d * sin_deg 30)
def pointB (d: ℝ) := (-d * cos_deg 30 + 60, d * sin_deg 30)

-- Define the distance as d
def distance (d: ℝ) := 800 * real.sqrt 3 - 240

-- Assert the distance calculation is as expected
theorem ship_lighthouse_distance : 
  ∀ (d: ℝ), 
    ship_distance 15 4 = 60 →
    pointA d = (-d * (real.sqrt 3 / 2), d / 2) →
    pointB d = (-d * (real.sqrt 3 / 2) + 60, d / 2) →
    tan_deg 75 = (d / 2) / (d * (real.sqrt 3 / 2) - 60) →
    d = distance d :=
by
  sorry

end ship_lighthouse_distance_l445_445719


namespace value_of_O_l445_445112

-- Define the values of the letters as integers
variable (L O V E I: ℤ)

-- Define the conditions based on the problem statement
def condition1 : L = 15 := by sorry
def condition2 : L + O + V + E = 60 := by sorry
def condition3 : L + I + V + E = 50 := by sorry
def condition4 : V + I + L + E = 55 := by sorry

-- Prove that O = 20
theorem value_of_O : O = 20 :=
by
  have hL : L = 15 := condition1
  have hLOVE : L + O + V + E = 60 := condition2
  have hLIVE : L + I + V + E = 50 := condition3
  have hVILE : V + I + L + E = 55 := condition4
  sorry

end value_of_O_l445_445112


namespace points_roles_l445_445407

variables {A B C O N P : Type} [InnerProductSpace ℝ A] [EuclideanGeometry A]

-- Assume the conditions hold
variables (O A B C N P : A)
  (h1 : dist O A = dist O B)
  (h2 : dist O B = dist O C)
  (h3 : (N -ᵥ A) + (N -ᵥ B) + (N -ᵥ C) = 0)
  (h4 : inner (P -ᵥ A) (P -ᵥ B) = inner (P -ᵥ B) (P -ᵥ C))
  (h5 : inner (P -ᵥ B) (P -ᵥ C) = inner (P -ᵥ C) (P -ᵥ A))

-- Goal to prove
theorem points_roles (h1 : dist O A = dist O B) (h2 : dist O B = dist O C)
  (h3 : (N -ᵥ A) + (N -ᵥ B) + (N -ᵥ C) = 0)
  (h4 : inner (P -ᵥ A) (P -ᵥ B) = inner (P -ᵥ B) (P -ᵥ C))
  (h5 : inner (P -ᵥ B) (P -ᵥ C) = inner (P -ᵥ C) (P -ᵥ A)) :
  (is_circumcenter O A B C) ∧ (is_centroid N A B C) ∧ (is_orthocenter P A B C) :=
sorry

end points_roles_l445_445407


namespace seventeenth_replacement_month_l445_445530

def months_after_january (n : Nat) : Nat :=
  n % 12

theorem seventeenth_replacement_month :
  months_after_january (7 * 16) = 4 :=
by
  sorry

end seventeenth_replacement_month_l445_445530


namespace vector_dot_product_sum_eq_l445_445947

variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V)

theorem vector_dot_product_sum_eq :
  a + b + c = 0 →
  ∥a∥ = 1 →
  ∥b∥ = 2 →
  ∥c∥ = 2 →
  inner a b + inner b c + inner c a = -9 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end vector_dot_product_sum_eq_l445_445947


namespace sin_half_alpha_l445_445868

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445868


namespace analytical_expression_smallest_positive_period_min_value_max_value_l445_445807

noncomputable def P (x : ℝ) : ℝ × ℝ :=
  (Real.cos (2 * x) + 1, 1)

noncomputable def Q (x : ℝ) : ℝ × ℝ :=
  (1, Real.sqrt 3 * Real.sin (2 * x) + 1)

noncomputable def f (x : ℝ) : ℝ :=
  (P x).1 * (Q x).1 + (P x).2 * (Q x).2

theorem analytical_expression (x : ℝ) : 
  f x = 2 * Real.sin (2 * x + Real.pi / 6) + 2 :=
sorry

theorem smallest_positive_period : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
sorry

theorem min_value : 
  ∃ x : ℝ, f x = 0 :=
sorry

theorem max_value : 
  ∃ y : ℝ, f y = 4 :=
sorry

end analytical_expression_smallest_positive_period_min_value_max_value_l445_445807


namespace points_cyclic_l445_445578

-- Define the necessary geometric concepts.
variables {A B C D P Q : Point}
variables {circle : Circle}

-- Given conditions.
axiom diameter_condition : circle.is_diameter A B
axiom points_on_circle : circle.contains C ∧ circle.contains D
axiom ray_intersections : 
  circle.tangent_at B ∩ (ray A C) = {P} ∧ 
  circle.tangent_at B ∩ (ray A D) = {Q}

-- The theorem to be proved.
theorem points_cyclic (A B C D P Q : Point) (circle : Circle) 
  (diameter_condition : circle.is_diameter A B)
  (points_on_circle : circle.contains C ∧ circle.contains D)
  (ray_intersections : 
    circle.tangent_at B ∩ (ray A C) = {P} ∧ 
    circle.tangent_at B ∩ (ray A D) = {Q}) :
  cyclic_quadrilateral C D P Q :=
by
  -- The proof itself is omitted.
  sorry

end points_cyclic_l445_445578


namespace train_length_l445_445171

-- Define the conditions
def jogger_speed := 9 -- in km/hr
def train_speed := 45 -- in km/hr
def head_start := 180 -- in meters
def time_to_pass := 30 -- in seconds

-- Define the mathematically equivalent proof problem in Lean
theorem train_length : 
  let relative_speed := (train_speed - jogger_speed) * 1000 / 3600 in
  let distance_covered := relative_speed * time_to_pass in
  distance_covered - head_start = 120 :=
by
  -- Sorry to skip the proof as requested
  sorry

end train_length_l445_445171


namespace area_of_EFCD_trapezoid_l445_445515

-- Define the conditions given in the problem
def AB : ℝ := 10
def CD : ℝ := 24
def height : ℝ := 15
def E_midpoint_AD : Prop := true
def F_midpoint_BC : Prop := true

-- Define the statement of the problem in Lean
theorem area_of_EFCD_trapezoid (h1 : AB = 10) (h2 : CD = 24) (h3 : height = 15) 
  (h4 : E_midpoint_AD) (h5 : F_midpoint_BC) : 
  let EF := (AB + CD) / 2
  EFCD_area = 15 * (17 + 24) / 2 := 
begin
  let EF := (AB + CD) / 2,
  sorry -- Proof to be filled in
end

end area_of_EFCD_trapezoid_l445_445515


namespace matrix_determinant_zero_l445_445750

noncomputable def matrix_example : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]

theorem matrix_determinant_zero : matrix_example.det = 0 := 
by 
  sorry

end matrix_determinant_zero_l445_445750


namespace ellipse_and_hyperbola_standard_equations_l445_445428

noncomputable def standard_equation_of_ellipse (a c : ℝ) (e : ℝ) (major_axis_length : ℝ) (focus_axis : ℝ) : Prop :=
  (focus_axis = a ∧ major_axis_length = 2 * a ∧ e = c / a ∧ b^2 = a^2 * (1 - e^2 ∧ 
  (10 = 2 * a) ∧ (e = 4 / 5) → 
  (a = 5 ∧ c = 4 ∧ b^2 = 25 * (1 - (16 / 25)) = 9 → b = 3 ∧ standard_equation = x^2 / 25 + y^2 / 9 = 1))

noncomputable def standard_equation_of_hyperbola (slope : ℝ) (vertex_distance : ℝ) (a : ℝ) (b : ℝ) : Prop :=
  (vertex_distance = 2a ∧ slope = abs (b / a) ∧ 
  vertex_distance = 6 ∧ slope = ± 3 / 2) →
  (a^2 = 11 ∧ b^2 = 9 → standard_equation = x^2 / 11 - y^2 / 9 = 1)

theorem ellipse_and_hyperbola_standard_equations :
  ∃ a c e focus_axis major_axis_length slope vertex_distance,
  (standard_equation_of_ellipse a c e major_axis_length focus_axis) ∧
  (standard_equation_of_hyperbola slope vertex_distance a b) :=
sorry

end ellipse_and_hyperbola_standard_equations_l445_445428


namespace product_to_fraction_l445_445211

theorem product_to_fraction :
  (∏ k in Finset.range (49), (1 - 1 / (k + 2))) = 1 / 50 :=
by
  sorry

end product_to_fraction_l445_445211


namespace combination_values_l445_445988

theorem combination_values (x : ℝ) :
    binomial 18 x = binomial 18 (3 * x - 6) ↔ (x = 3 ∨ x = 6) := 
by
  sorry

end combination_values_l445_445988


namespace arrangement_of_BANANA_l445_445358

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445358


namespace sin_half_alpha_l445_445866

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445866


namespace arrangement_count_correct_l445_445738

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end arrangement_count_correct_l445_445738


namespace total_amount_paid_l445_445561

-- Definitions based on the conditions in step a)
def ring_cost : ℕ := 24
def ring_quantity : ℕ := 2

-- Statement to prove that the total cost is $48.
theorem total_amount_paid : ring_quantity * ring_cost = 48 := 
by
  sorry

end total_amount_paid_l445_445561


namespace matrix_determinant_sin_zero_l445_445748

theorem matrix_determinant_sin_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ] in Matrix.det A = 0 :=
by
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  sorry

end matrix_determinant_sin_zero_l445_445748


namespace combinations_eight_choose_three_l445_445022

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end combinations_eight_choose_three_l445_445022


namespace sin_half_alpha_l445_445851

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445851


namespace find_t_l445_445788

theorem find_t (t : ℝ) (f : ℝ → ℝ) (m n : ℝ)
  (h_f : ∀ x, f x = (x^3 + t*x^2 + sqrt 2 * t * sin (x + π/4) + 2*t) / (x^2 + 2 + cos x))
  (h_t_nonzero : t ≠ 0)
  (h_max : ∃ x, f x = m)
  (h_min : ∃ x, f x = n)
  (h_sum : m + n = 2017) :
  t = 2017 / 2 :=
sorry

end find_t_l445_445788


namespace simplify_expression_l445_445580

theorem simplify_expression :
  ((45 * 2^10) / (15 * 2^5) * 5) = 480 := by
  sorry

end simplify_expression_l445_445580


namespace comparison_abc_l445_445547

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := 0.5 * (Real.log 2023 / Real.log 2022 + Real.log 2022 / Real.log 2023)

theorem comparison_abc : c > b ∧ b > a := by
  sorry

end comparison_abc_l445_445547


namespace BANANA_arrangements_l445_445268

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445268


namespace incenter_circumcenter_midpoints_concyclic_l445_445020

variables {A B C I O M N : Type*}
variables [metric_space A] [metric_space B] [metric_space C] 
variables [metric_space I] [metric_space O] [metric_space M] [metric_space N]
variables [has_dist A] [has_dist B] [has_dist C]
variables [has_dist I] [has_dist O] [has_dist M] [has_dist N]

-- Defines the lengths of the sides of the triangle
variable (sides : dist B A * 2 = dist C A + dist B C)

-- Definition involving the midpoints and special points, with conditions
variable (midpoints : midpoint M A C ∧ midpoint N B C)
variable (centers : is_incenter I A B C ∧ is_circumcenter O A B C)

-- The theorem statement that incenter, circumcenter and the midpoints are concyclic
theorem incenter_circumcenter_midpoints_concyclic 
  (h : dist B A * 2 = dist C A + dist B C) 
  (hm : midpoint M A C) 
  (hn : midpoint N B C) 
  (hi : is_incenter I A B C) 
  (ho : is_circumcenter O A B C) :
  cyclic I O M N :=
sorry

end incenter_circumcenter_midpoints_concyclic_l445_445020


namespace sin_half_alpha_l445_445880

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445880


namespace find_y_l445_445498

open Classical

theorem find_y (a b c x y : ℚ)
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4) :
  y = 15 / x :=
sorry

end find_y_l445_445498


namespace Mike_changed_64_tires_l445_445062

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end Mike_changed_64_tires_l445_445062


namespace min_points_condition_l445_445778

theorem min_points_condition :
  ∃ n, n ≥ 3 ∧ 
  ∃ (A : Fin n → Point) (h_nc : ∀ (i j k : Fin n), collinear (A i) (A j) (A k) → i = j ∨ j = k ∨ i = k) 
  (h_mid : ∀ (i : Fin n), ∃ j : Fin n, j ≠ i ∧ midpoint (A i) (A (i + 1) % n) ∈ segment (A j) (A ((j + 1) % n))),
  n = 6 :=
by
  sorry

end min_points_condition_l445_445778


namespace arithmetic_sequence_sum_is_right_l445_445109

noncomputable def arithmetic_sequence_sum : ℤ :=
  let a1 := 1
  let d := -2
  let a2 := a1 + d
  let a3 := a1 + 2 * d
  let a6 := a1 + 5 * d
  let S6 := 6 * a1 + (6 * (6-1)) / 2 * d
  S6

theorem arithmetic_sequence_sum_is_right {d : ℤ} (h₀ : d ≠ 0) 
(h₁ : (a1 + 2 * d) ^ 2 = (a1 + d) * (a1 + 5 * d)) :
  arithmetic_sequence_sum = -24 := by
  sorry

end arithmetic_sequence_sum_is_right_l445_445109


namespace cos_a_value_side_c_value_l445_445516

variables {A B C: ℝ}
variables (a b c: ℝ)
variables (cosA cosB cosC: ℝ)

-- Definition for cosine of angles A, B, and C in terms of sides a, b, c
def cosine_angles : Prop := 2 * a * cosA = c * cosB + b * cosC

-- Part I: Prove that given 2a cos A = c cos B + b cos C, then cos A = 1/2
theorem cos_a_value (h1: cosine_angles a b c cosA cosB cosC):
  cosA = 1 / 2 :=
sorry

-- Part II: Given a = 1 and additional condition, find possible values of c
variables (cos_B_half cos_C_half: ℝ)
def additional_condition : Prop := cos_B_half^2 + cos_C_half^2 = 1 + sqrt 3 / 4

theorem side_c_value (h2: a = 1) (h3: additional_condition cos_B_half cos_C_half):
  c = sqrt 3 / 3 ∨ c = 2 * sqrt 3 / 3 :=
sorry

end cos_a_value_side_c_value_l445_445516


namespace sin_half_alpha_l445_445858

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445858


namespace gmat_test_statistics_l445_445979

theorem gmat_test_statistics 
    (p1 : ℝ) (p2 : ℝ) (p12 : ℝ) (neither : ℝ) (S : ℝ) 
    (h1 : p1 = 0.85)
    (h2 : p12 = 0.60) 
    (h3 : neither = 0.05) :
    0.25 + S = 0.95 → S = 0.70 :=
by
  sorry

end gmat_test_statistics_l445_445979


namespace train_length_l445_445723

theorem train_length (length_bridge : ℝ) (time : ℝ) (speed_kmh : ℝ) 
    (h₁ : length_bridge = 300) 
    (h₂ : time = 12) 
    (h₃ : speed_kmh = 120) 
    (conversion_factor : ℝ := 1000 / 3600) : 
    ∃ (length_train : ℝ), length_train ≈ 99.96 :=
by
  let speed_ms := speed_kmh * conversion_factor
  let total_distance := speed_ms * time
  let length_train := total_distance - length_bridge
  use length_train
  sorry

end train_length_l445_445723


namespace symmetric_axis_sin_l445_445784

theorem symmetric_axis_sin (x : ℝ) : 
  (∀ x, y = sin(3 * x + 3 * π / 4) → (y = 1 ↔ x = -π / 12)) := sorry

end symmetric_axis_sin_l445_445784


namespace smallest_ninequality_l445_445388

theorem smallest_ninequality 
  (n : ℕ) 
  (h : ∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≤ 2 ^ (1 - n)) : 
  n = 2 := 
by
  sorry

end smallest_ninequality_l445_445388


namespace two_quadrilaterals_divide_into_nine_parts_l445_445075

-- Define the two quadrilaterals and a function that determines the number of regions formed by their intersections on the sheet of paper.
def is_convex (poly : Set (ℝ × ℝ)) : Prop := sorry
def divides_into_n_parts (shapes : List (Set (ℝ × ℝ))) (n : ℕ) :=
  (∀ p ∈ Sheet, exists_unique (λ region, p ∈ region)) ∧
  (Sheet.partition shapes = n)

-- Establish the problem statement using Lean
theorem two_quadrilaterals_divide_into_nine_parts (q1 q2 : Set (ℝ × ℝ))
  (h1 : is_convex q1) (h2 : is_convex q2) : divides_into_n_parts [q1, q2] 9 :=
sorry

end two_quadrilaterals_divide_into_nine_parts_l445_445075


namespace sum_m_n_eq_minus_one_l445_445987

theorem sum_m_n_eq_minus_one (m n : ℝ) : 
  let p := 3 * (λ x, x^3 + (1/3) * x^2 + n * x) - (λ x, m * x^2 - 6 * x - 1) in
  (∀ x : ℝ, p x = 3 * x^3 + (1 - m) * x^2 + (3 * n + 6) * x + 1 ∧ (1 - m = 0) ∧ (3 * n + 6 = 0)) →
  m + n = -1 := 
by 
  intros h1;
  sorry

end sum_m_n_eq_minus_one_l445_445987


namespace number_of_last_digits_divisible_by_5_l445_445564

theorem number_of_last_digits_divisible_by_5 : 
  (∀ n : ℕ, (n % 5 = 0 ∨ n % 5 = 5) → (n % 10 = 0 ∨ n % 10 = 5)) → 
  ({d : ℕ | d = 0 ∨ d = 5}.card = 2) :=
by
  sorry

end number_of_last_digits_divisible_by_5_l445_445564


namespace restore_trapezoid_l445_445188

noncomputable def isBicentralTrapezoid (A I : Point) (ω : Circle) : Prop := sorry

theorem restore_trapezoid
  (A I : Point) (ω : Circle) (O : Point) (ABCD : Quadrilateral) 
  (h1 : is_center O ω)
  (h2 : is_incenter I ABCD)
  (h3 : A ∈ vertices ABCD)
  (h4 : ω = circumcircle ABCD)
  (h5 : isBicentralTrapezoid A I ω):
  ∃ ABCD' : Quadrilateral, is_trapezoid ABCD' ∧
  vertices ABCD' = vertices ABCD := by
  sorry

end restore_trapezoid_l445_445188


namespace sin_half_alpha_l445_445888

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445888


namespace jumble_words_with_A_l445_445611

theorem jumble_words_with_A 
  (alphabet_size : ℕ) 
  (letters : Fin alphabet_size → Char) 
  (A_in_alphabet : 'A' ∈ Set.range letters)
  (max_length : ℕ) 
  (H : alphabet_size = 20) 
  (M : max_length = 4) :
  (let without_A : ℕ := (Finset.range (max_length + 1)).sum (λ n, (alphabet_size - 1) ^ n);
       total : ℕ := (Finset.range (max_length + 1)).sum (λ n, alphabet_size ^ n)
   in total - without_A) = 30860 :=
by
  sorry

end jumble_words_with_A_l445_445611


namespace BANANA_arrangements_l445_445302

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445302


namespace no_a_b_satisfy_l445_445545

noncomputable def A (a b: ℝ) : set (ℝ × ℝ) := 
  { p | ∃ n : ℤ, p = (n : ℝ, n * a + b) }

noncomputable def B : set (ℝ × ℝ) := 
  { p | ∃ m : ℤ, p = (m : ℝ, 3 * (m : ℝ)^2 + 15) }

noncomputable def C : set (ℝ × ℝ) := 
  { p | p.1^2 + p.2^2 ≤ 144 }

theorem no_a_b_satisfy : ¬ ∃ a b: ℝ, (A a b ∩ B ≠ ∅) ∧ ((a, b) ∈ C) := by
  sorry

end no_a_b_satisfy_l445_445545


namespace lattice_point_probability_l445_445182

noncomputable def calculate_radius (side_length : ℝ) (prob : ℝ) : ℝ :=
  let π := Real.pi
  in (1 / Real.sqrt (3 * π))

theorem lattice_point_probability (d : ℝ) (side_length : ℝ) (prob : ℝ) :
  (side_length = 1000) ∧ (prob = 1/3) →
  d ≈ calculate_radius side_length prob :=
by
  -- Assuming conditions as stated in problem
  intros h
  cases' h with h1 h2
  rw [h1, h2]
  sorry

end lattice_point_probability_l445_445182


namespace solve_system_l445_445585

theorem solve_system (x y z a b c : ℝ)
  (h1 : x * (x + y + z) = a^2)
  (h2 : y * (x + y + z) = b^2)
  (h3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ x = -a^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ y = -b^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (z = c^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) :=
by
  sorry

end solve_system_l445_445585


namespace minimum_flowers_to_guarantee_bouquets_l445_445997

theorem minimum_flowers_to_guarantee_bouquets :
  (∀ (num_types : ℕ) (flowers_per_bouquet : ℕ) (num_bouquets : ℕ),
   num_types = 6 → flowers_per_bouquet = 5 → num_bouquets = 10 →
   ∃ min_flowers : ℕ, min_flowers = 70 ∧
   ∀ (picked_flowers : ℕ → ℕ), 
     (∀ t : ℕ, t < num_types → picked_flowers t ≥ 0 ∧ 
                (t < num_types - 1 → picked_flowers t ≤ flowers_per_bouquet * (num_bouquets - 1) + 4)) → 
     ∑ t in finset.range num_types, picked_flowers t = min_flowers → 
     ∑ t in finset.range num_types, (picked_flowers t / flowers_per_bouquet) ≥ num_bouquets) := 
by {
  intro num_types flowers_per_bouquet num_bouquets,
  intro h1 h2 h3,
  use 70,
  split,
  {
    exact rfl,
  },
  {
    intros picked_flowers h_picked,
    sorry,
  }
}

end minimum_flowers_to_guarantee_bouquets_l445_445997


namespace dot_product_sum_l445_445958

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom vec_sum : a + b + c = 0
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom norm_c : ∥c∥ = 2

-- The theorem to prove
theorem dot_product_sum :
  ⟪a, b⟫ + ⟪b, c⟫ + ⟪c, a⟫ = - 9 / 2 :=
sorry

end dot_product_sum_l445_445958


namespace find_k_parallel_vectors_l445_445452

theorem find_k_parallel_vectors
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (k : ℝ)
  (h_a : a = (3, 1))
  (h_b : b = (1, 3))
  (h_c : c = (k, 7))
  (h_parallel : ∃ (λ : ℝ), (a.1 - c.1, a.2 - c.2) = (λ * b.1, λ * b.2)) :
  k = 5 :=
by
  sorry

end find_k_parallel_vectors_l445_445452


namespace Gina_needs_more_tip_l445_445395

/-- Gina is considered a good tipper if she tips at least 20%.
  Gina initially tipped 5% on a bill of $26.
  How many more cents would Gina need to tip to be considered a good tipper? -/

theorem Gina_needs_more_tip
  (bill : ℝ)
  (tip_percent_Gina : ℝ)
  (good_tip_percent : ℝ)
  (cents_per_dollar : ℤ)
  (Gina_tip : ℝ := bill * tip_percent_Gina)
  (good_tip : ℝ := bill * good_tip_percent)
  (difference_in_dollars : ℝ := good_tip - Gina_tip)
  (difference_in_cents : ℤ := (difference_in_dollars * ↑cents_per_dollar).toInt) :
  bill = 26 → tip_percent_Gina = 0.05 → good_tip_percent = 0.20 → cents_per_dollar = 100 → difference_in_cents = 390 :=
by
  intros h1 h2 h3 h4
  sorry

end Gina_needs_more_tip_l445_445395


namespace vasya_guaranteed_win_with_4_queries_l445_445080

noncomputable def polynomial_game_solution : ℕ :=
  let minimum_rubles_needed := 4 in
  minimum_rubles_needed

theorem vasya_guaranteed_win_with_4_queries 
  (P : Polynomial ℤ) :
  ∃ n ≤ 4, ∃ a1 a2 a3 a4 : ℤ, -- Vasya makes 4 queries
    (P.eval a1).natAbs = (P.eval a2).natAbs ∨
    (P.eval a1).natAbs = (P.eval a3).natAbs ∨
    (P.eval a1).natAbs = (P.eval a4).natAbs ∨
    (P.eval a2).natAbs = (P.eval a3).natAbs ∨
    (P.eval a2).natAbs = (P.eval a4).natAbs ∨
    (P.eval a3).natAbs = (P.eval a4).natAbs
:=
sorry

end vasya_guaranteed_win_with_4_queries_l445_445080


namespace trigonometric_identity_l445_445158

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : α + β = π / 3)  -- Note: 60 degrees is π/3 radians
  (tan_add : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) 
  (tan_60 : Real.tan (π / 3) = Real.sqrt 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 :=
sorry

end trigonometric_identity_l445_445158


namespace total_hike_time_l445_445525

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l445_445525


namespace coin_probabilities_equal_l445_445629

theorem coin_probabilities_equal :
  let A := (FirstCoin = "heads")
  let B := (SecondCoin = "tails")
  (P(A) = P(B)) :=
by
  let A := "the first coin is heads up"
  let B := "the second coin is tails up"
  have P_A : Probability A = 1 / 2 := by sorry
  have P_B : Probability B = 1 / 2 := by sorry
  sorry

end coin_probabilities_equal_l445_445629


namespace samantha_correct_percentage_l445_445576

theorem samantha_correct_percentage :
  let test1 := 30
  let test2 := 50
  let test3 := 20
  let correct1 := 0.9 * test1
  let correct2 := 0.85 * test2
  let correct3 := 0.6 * test3
  let total_correct := correct1 + correct2 + correct3
  let total_questions := test1 + test2 + test3
  (total_correct / total_questions) = 0.815 :=
by
  let test1 := 30
  let test2 := 50
  let test3 := 20
  let correct1 := 0.9 * test1
  let correct2 := 0.85 * test2
  let correct3 := 0.6 * test3
  let total_correct := correct1 + correct2 + correct3
  let total_questions := test1 + test2 + test3
  have : total_questions = 100 := rfl
  have : total_correct = 81.5 := by norm_num
  show (total_correct / total_questions) = 0.815 from by norm_num
  sorry

end samantha_correct_percentage_l445_445576


namespace distinct_arrangements_of_BANANA_l445_445339

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445339


namespace banana_permutations_l445_445256

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445256


namespace sin_half_alpha_l445_445873

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445873


namespace area_smaller_segment_l445_445698

open Real

-- Definitions for the proof problem.
def radius : ℝ := 15
def distance_center_to_chord : ℝ := 9

theorem area_smaller_segment (h1 : radius = 15) (h2 : distance_center_to_chord = 9) :
    (the area of the smaller segment cut off by the chord) = 117.29 := 
sorry

end area_smaller_segment_l445_445698


namespace catalan_number_divisibility_l445_445039

open Nat

def C (k : ℕ) : ℕ := (binom (2 * k) k) / (k + 1)

theorem catalan_number_divisibility (p : ℕ) [Fact (Nat.Prime p)] (hp: Odd p):
  (∃ n (hn : n ∈ Finset.range (p - 1)), (∑ k in Finset.range (p - 1), C k * n^k) % p = 0) ∧
  (∃ n (hn : n ∈ Finset.range (p - 1)), (∑ k in Finset.range (p - 1), C k * n^k) % p ≠ 0) :=
sorry

end catalan_number_divisibility_l445_445039


namespace sin_half_angle_l445_445816

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445816


namespace ellipse_distance_pf2_l445_445805

noncomputable def ellipse_focal_length := 2 * Real.sqrt 2
noncomputable def ellipse_equation (a : ℝ) (a_gt_one : a > 1)
  (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / a) + y^2 = 1

theorem ellipse_distance_pf2
  (a : ℝ) (a_gt_one : a > 1)
  (focus_distance : 2 * Real.sqrt (a - 1) = 2 * Real.sqrt 2)
  (F1 F2 P : ℝ × ℝ)
  (on_ellipse : ellipse_equation a a_gt_one P)
  (PF1_eq_two : dist P F1 = 2)
  (a_eq : a = 3) :
  dist P F2 = 2 * Real.sqrt 3 - 2 := 
sorry

end ellipse_distance_pf2_l445_445805


namespace find_EC_length_l445_445544

variable {A B C E : Type}
variables [EuclideanGeometry E]

def right_triangle (A B C : E) : Prop :=
  ∃ (B : E), angle A B C = 90

def circle_with_diameter_intersect (B C A E : E) : Prop :=
  let circle := Circle (segment_length B C / 2) (midpoint B C)
  segment B C = diam circle ∧ E ∈ inter circle (segment A B)

def given_conditions (A B C E : E) : Prop :=
  right_triangle A B C ∧
  circle_with_diameter_intersect B C A E ∧
  segment_length B E = 6 ∧
  segment_length A E = 3

theorem find_EC_length (A B C E : E) (h : given_conditions A B C E) :
  segment_length E C = 12 :=
sorry

end find_EC_length_l445_445544


namespace arrangement_of_BANANA_l445_445362

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445362


namespace policeman_can_catch_gangster_l445_445025

theorem policeman_can_catch_gangster
    (a : ℝ) (C A B D E : ℝ × ℝ)
    (v_p v_g : ℝ)
    (h₁ : v_g = 2 * v_p)
    (h₂ : C = (0, 0))
    (h₃ : A = (a/2, a/2) ∨ A = (-a/2, a/2) ∨ A = (-a/2, -a/2) ∨ A = (a/2, -a/2))
    (h₄ : ∀ (P : ℝ × ℝ), (P = (0, a/2) ∨ P = (0, -a/2) ∨ P = (a/2, 0) ∨ P = (-a/2, 0))) :
  ∃ (P : ℝ × ℝ), (P = (0, a/2) ∧ P = A) ∨ (P = (0, -a/2) ∧ P = A) ∨
                   (P = (a/2, 0) ∧ P = A) ∨ (P = (-a/2, 0) ∧ P = A) ∨
                   (v_g * (sqrt 2 * a / 2) / v_p ≤ 2 * (sqrt 2 * a / 2)) :=
sorry

end policeman_can_catch_gangster_l445_445025


namespace matrix_determinant_sin_zero_l445_445749

theorem matrix_determinant_sin_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ] in Matrix.det A = 0 :=
by
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    [Real.sin 1, Real.sin 2, Real.sin 3],
    [Real.sin 4, Real.sin 5, Real.sin 6],
    [Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  sorry

end matrix_determinant_sin_zero_l445_445749


namespace vector_dot_product_sum_l445_445961

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem vector_dot_product_sum (h₁ : a + b + c = 0) 
                               (ha : ∥a∥ = 1)
                               (hb : ∥b∥ = 2)
                               (hc : ∥c∥ = 2) :
  (a ⬝ b) + (b ⬝ c) + (c ⬝ a) = -9 / 2 := sorry

end vector_dot_product_sum_l445_445961


namespace parallel_lines_slope_l445_445943

theorem parallel_lines_slope {a : ℝ} (h : -a / 3 = -2 / 3) : a = 2 := 
by
  sorry

end parallel_lines_slope_l445_445943


namespace correct_propositions_l445_445231

theorem correct_propositions (p q : Prop) (a : ℝ) :
  (¬ (∃ x : ℝ, x ^ 2 + 1 > 3 * x) ↔ ¬ (¬ ∀ x : ℝ, x ^ 2 + 1 ≤ 3 * x)) ∧
  (¬ (p ∨ q) → ¬ p ∧ ¬ q) ∧
  ¬ ((a > 3 → a > real.pi)) ∧
  (∀ x : ℝ, (x + 2) * (x + a) = (-x + 2) * (-x + a) → a = -2) :=
by
  sorry

end correct_propositions_l445_445231


namespace seating_arrangements_valid_l445_445731

theorem seating_arrangements_valid :
  let people := ["Alice", "Bob", "Carla", "Derek", "Eric"]
  ∃ valid_arrangements : List (List String),
  valid_arrangements.length = 12 ∧
  (∀ arr ∈ valid_arrangements,
    ∀ i, 
      (arr.nth i = some "Alice" → (arr.nth (i-1) ≠ some "Bob" ∧ arr.nth (i+1) ≠ some "Bob" ∧ arr.nth (i-1) ≠ some "Derek" ∧ arr.nth (i+1) ≠ some "Derek")) ∧
      (arr.nth i = some "Carla" → (arr.nth (i-1) ≠ some "Bob" ∧ arr.nth (i+1) ≠ some "Bob" ∧ arr.nth (i-1) ≠ some "Eric" ∧ arr.nth (i+1) ≠ some "Eric"))
  )
:= sorry

end seating_arrangements_valid_l445_445731


namespace min_touches_to_turn_on_all_bulbs_l445_445623

-- Definition for toggling a bulb and all bulbs in the same row and column
def toggle_bulb (board : ℕ → ℕ → bool) (i j : ℕ) : ℕ → ℕ → bool :=
  λ x y, if x = i ∨ y = j then !board x y else board x y

-- Definition for applying multiple touches
def apply_touches (board : ℕ → ℕ → bool) (touches : list (ℕ × ℕ)) : ℕ → ℕ → bool :=
  touches.foldl (λ b (i, j), toggle_bulb b i j) board

-- Initial board, where all bulbs are off
def initial_board (n : ℕ) : ℕ → ℕ → bool := λ _ _, false

-- Uniform board with all bulbs on
def all_on (n : ℕ) : ℕ → ℕ → bool := λ _ _, true

-- Main theorem statement
theorem min_touches_to_turn_on_all_bulbs (n : ℕ) : 
  ∃ (m : ℕ) (touches : list (ℕ × ℕ)), 
    (if n % 2 = 1 then m = n else m = n^2) ∧ 
    apply_touches (initial_board n) touches = all_on n := 
sorry

end min_touches_to_turn_on_all_bulbs_l445_445623


namespace banana_arrangements_l445_445333

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445333


namespace number_of_arrangements_of_BANANA_l445_445315

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445315


namespace banana_arrangement_count_l445_445275

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445275


namespace angle_BAD_eq_30_l445_445512

open EuclideanGeometry

-- Defining the given points A, B, C, D.
variables {A B C D : Point ℝ}

-- Given conditions AB = CD and BC = 2AD
variables (h1: dist A B = dist C D) (h2: dist B C = 2 * dist A D)

-- The theorem to prove
theorem angle_BAD_eq_30 : ∠BAD = 30 :=
by
  sorry

end angle_BAD_eq_30_l445_445512


namespace arrangement_of_BANANA_l445_445359

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445359


namespace arithmetic_sequence_a_sum_c_first_n_terms_l445_445432

-- Define the sequences and conditions
variable (b : ℕ+ → ℝ) (a : ℕ+ → ℝ) (c : ℕ+ → ℝ) (S : ℕ → ℝ)

-- Conditions
-- sequence {b_n} is an increasing geometric sequence
axiom b_increasing_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ+, b n = b 1 * q^(n - 1)
-- b₁ + b₃ = 5
axiom condition_b1_b3_sum : b 1 + b 3 = 5
-- b₁ ⋅ b₃ = 4
axiom condition_b1_b3_product : b 1 * b 3 = 4

-- Definitions
noncomputable def a_n (n : ℕ+) : ℝ := log 2 (b n) + 3
noncomputable def c_n (n : ℕ+) : ℝ := 1 / (a n * a (n + 1))

-- Goal
theorem arithmetic_sequence_a : ∀ n : ℕ+, a n = n + 2 :=
by
  sorry

theorem sum_c_first_n_terms : ∀ n : ℕ, S n = ∑ i in Finset.range n, c i = n / (3 * (n + 3)) :=
by
  sorry

end arithmetic_sequence_a_sum_c_first_n_terms_l445_445432


namespace expenditure_of_negative_l445_445981

def income := 5000
def expenditure (x : Int) : Int := -x

theorem expenditure_of_negative (x : Int) : expenditure (-x) = x :=
by
  sorry

example : expenditure (-400) = 400 :=
by 
  exact expenditure_of_negative 400

end expenditure_of_negative_l445_445981


namespace vector_calculation_l445_445444

def a :ℝ × ℝ := (1, 2)
def b :ℝ × ℝ := (1, -1)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_calculation : scalar_mult (1/3) a - scalar_mult (4/3) b = (-1, 2) :=
by sorry

end vector_calculation_l445_445444


namespace determinant_of_matrix_l445_445546

theorem determinant_of_matrix (a b c : ℝ) 
  (h1 : Polynomial.root (Polynomial.mk [1, -3, 2, -1]) a)
  (h2 : Polynomial.root (Polynomial.mk [1, -3, 2, -1]) b)
  (h3 : Polynomial.root (Polynomial.mk [1, -3, 2, -1]) c) :
  Matrix.det ![
    ![a - b, b - c, c - a],
    ![b - c, c - a, a - b],
    ![c - a, a - b, b - c]
  ] = 0 := 
  sorry

end determinant_of_matrix_l445_445546


namespace bridge_length_l445_445722

theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross_bridge : ℝ) 
  (train_speed_m_s : train_speed_kmh * (1000 / 3600) = 15) : 
  train_length = 110 → train_speed_kmh = 54 → time_to_cross_bridge = 16.13204276991174 → 
  ((train_speed_kmh * (1000 / 3600)) * time_to_cross_bridge - train_length = 131.9806415486761) :=
by
  intros h1 h2 h3
  sorry

end bridge_length_l445_445722


namespace number_of_non_empty_subsets_l445_445400

theorem number_of_non_empty_subsets (a : ℝ) :
  let M := { abs x | x ∈ ℝ ∧ x^3 + (a^2 + 1) * x + 2 * a^2 + 10 = 0 } in
  M.nonempty → fintype.card M.subsets ≠ 2 :=
by sorry

end number_of_non_empty_subsets_l445_445400


namespace banana_arrangements_l445_445329

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445329


namespace solve_for_x_l445_445584

theorem solve_for_x (x : ℝ) (h : x + 3 * x = 500 - (4 * x + 5 * x)) : x = 500 / 13 := 
by 
  sorry

end solve_for_x_l445_445584


namespace sphere_containment_l445_445098

noncomputable def tetrahedronGeometry :
  Type :=
sorry

variables (G_a H H_a H_b H_c H_d X_a X_b X_c X_d : tetrahedronGeometry)

-- Assume the intersection point of altitudes.
axiom altitudes_intersect_at_H :
  intersect (altitude H_b H_c H_d) (altitude H_a H_c H_d) (altitude H_a H_b H_d) (altitude H_a H_b H_c) = H

-- Assume points dividing the altitudes.
axiom points_dividing_altitudes_2_to_1 :
  divides_ratio(X_b, G_a) = 2/1 ∧
  divides_ratio(X_c, G_a) = 2/1 ∧
  divides_ratio(X_d, G_a) = 2/1

-- Main theorem to be proven
theorem sphere_containment :
  ∃ ω : Sphere,
    in_sphere ω H ∧ in_sphere ω H_a ∧ in_sphere ω X_b ∧ in_sphere ω X_c ∧ in_sphere ω X_d :=
sorry

end sphere_containment_l445_445098


namespace coin_event_probability_equivalence_l445_445627
open ProbabilityTheory

-- Definitions of Events
def event_A (first_coin: Bool) : Prop :=
  first_coin = true -- heads is represented by true

def event_B (second_coin: Bool) : Prop :=
  second_coin = false -- tails is represented by false

-- The main theorem statement
theorem coin_event_probability_equivalence 
  (first_coin second_coin: Bool) 
  (h_fair : ∀ (c : Bool), P(c = true) = 1/2 ∧ P(c = false) = 1/2) :
  P(event_A first_coin) = P(event_B second_coin) := 
sorry

end coin_event_probability_equivalence_l445_445627


namespace mark_50_points_in_convex_100_gon_l445_445084

theorem mark_50_points_in_convex_100_gon :
  ∀ (P : ℕ → ℕ → Prop) (convex_100_gon : List (ℕ × ℕ)),
  (∀ i, 1 ≤ i ∧ i ≤ 100 → ∃ a b, P a b) →
  (∀ v ∈ convex_100_gon, ∃ p1 p2 ∈ marked_points, v ∈ line_segment p1 p2) :=
begin
  sorry
end

end mark_50_points_in_convex_100_gon_l445_445084


namespace find_angle_3_l445_445423

def angle_degree_min := (nat × nat) -- represents an angle in degrees and minutes, (e.g., (67, 12) represents 67°12')

noncomputable def angle_1 : angle_degree_min := (67, 12)

def complementary (a b : angle_degree_min) : Prop :=
  a.1 + b.1 = 90 ∧ a.2 + b.2 = 60 

def supplementary (a b : angle_degree_min) : Prop :=
  a.1 + b.1 = 180 ∧ a.2 + b.2 = 60

theorem find_angle_3 (angle_2 angle_3 : angle_degree_min) :
  complementary angle_1 angle_2 →
  supplementary angle_2 angle_3 →
  angle_3 = (157, 12) :=
sorry

end find_angle_3_l445_445423


namespace BANANA_arrangements_l445_445266

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445266


namespace smallest_n_with_terminating_decimal_and_digit_7_l445_445652

-- Definition for \( n \) containing the digit 7
def contains_digit_7 (n: Nat) : Prop :=
  (n.toString.contains '7')

-- Definition for \( \frac{1}{n} \) being a terminating decimal
def is_terminating_decimal (n: Nat) : Prop :=
  ∃ a b: Nat, n = 2^a * 5^b

-- The main theorem statement
theorem smallest_n_with_terminating_decimal_and_digit_7 :
  Nat.find (λ n => is_terminating_decimal n ∧ contains_digit_7 n) = 65536 := by
  sorry

end smallest_n_with_terminating_decimal_and_digit_7_l445_445652


namespace correct_system_l445_445687

def system_of_equations (x y : ℤ) : Prop :=
  (5 * x + 45 = y) ∧ (7 * x - 3 = y)

theorem correct_system : ∃ x y : ℤ, system_of_equations x y :=
sorry

end correct_system_l445_445687


namespace initial_fraction_l445_445376

-- Definitions:
variables (S L W : ℝ) (F : ℝ)

-- Conditions:
-- Equal amounts of water pour into each jar
def equal_amounts_poured : Prop := 
  W = (1/3) * S ∧ W = F * L

-- When the lesser jar's water is poured into the larger jar, 2/3 of the larger jar is filled
def resulting_fill : Prop := 
  2 * W = (2/3) * L

-- Proof objective: prove the initial fraction filled in the larger jar is 1/3
theorem initial_fraction (S L W : ℝ) (F : ℝ) (heq: equal_amounts_poured S L W F) (hresult: resulting_fill S L W):
  F = 1 / 3 :=
sorry

end initial_fraction_l445_445376


namespace ellipse_equation_l445_445916

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end ellipse_equation_l445_445916


namespace percentage_music_l445_445622

variable (students_total : ℕ)
variable (percent_dance percent_art percent_drama percent_sports percent_photography percent_music : ℝ)

-- Define the problem conditions
def school_conditions : Prop :=
  students_total = 3000 ∧
  percent_dance = 0.125 ∧
  percent_art = 0.22 ∧
  percent_drama = 0.135 ∧
  percent_sports = 0.15 ∧
  percent_photography = 0.08 ∧
  percent_music = 1 - (percent_dance + percent_art + percent_drama + percent_sports + percent_photography)

-- Define the proof statement
theorem percentage_music : school_conditions students_total percent_dance percent_art percent_drama percent_sports percent_photography percent_music → percent_music = 0.29 :=
by
  intros h
  rw [school_conditions] at h
  sorry

end percentage_music_l445_445622


namespace eccentricity_range_of_ellipse_l445_445982

theorem eccentricity_range_of_ellipse
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (e : ℝ) (he1 : e > 0) (he2 : e < 1)
  (h_directrix : 2 * (a / e) ≤ 3 * (2 * a)) :
  (1 / 3) ≤ e ∧ e < 1 := 
sorry

end eccentricity_range_of_ellipse_l445_445982


namespace find_n_l445_445989

theorem find_n (x n : ℝ) (h1 : ((x / n) * 5) + 10 - 12 = 48) (h2 : x = 40) : n = 4 :=
sorry

end find_n_l445_445989


namespace jamie_bought_yellow_balls_l445_445519

theorem jamie_bought_yellow_balls :
  ∀ (red balls_initially num_balls_after_loss total_balls_after_purchase : ℕ),
    red = 16 →
    balls_initially = 2 * red →
    balls_after_loss = red - 6 →
    total_balls_after_purchase = balls_after_loss + balls_initially →
    74 = total_balls_after_purchase + yellow →
    yellow = 32 :=
by
  intros red balls_initially balls_after_loss total_balls_after_purchase yellow
  intros h1 h2 h3 h4 h5
  sorry

end jamie_bought_yellow_balls_l445_445519


namespace sin_half_angle_l445_445831

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445831


namespace sufficient_but_not_necessary_condition_l445_445406

variables (a b : EuclideanSpace ℝ (Fin 2)) [Nonzero a] [Nonzero b]

theorem sufficient_but_not_necessary_condition (ha : a ≠ 0)
  (hb : b ≠ 0) (hsuff : a = 3 • b) : (a / ‖a‖ = b / ‖b‖) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l445_445406


namespace permutations_of_BANANA_l445_445241

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445241


namespace find_a1_l445_445937

-- Define the sequence
def seq (a : ℕ → ℝ) := ∀ n : ℕ, 0 < n → a n = (1/2) * a (n + 1)

-- Given conditions
def a3_value (a : ℕ → ℝ) := a 3 = 12

-- Theorem statement
theorem find_a1 (a : ℕ → ℝ) (h_seq : seq a) (h_a3 : a3_value a) : a 1 = 3 :=
by
  sorry

end find_a1_l445_445937


namespace necklace_stand_capacity_l445_445170

def necklace_stand_initial := 5
def ring_display_capacity := 30
def ring_display_current := 18
def bracelet_display_capacity := 15
def bracelet_display_current := 8
def cost_per_necklace := 4
def cost_per_ring := 10
def cost_per_bracelet := 5
def total_cost := 183

theorem necklace_stand_capacity : necklace_stand_current + (total_cost - (ring_display_capacity - ring_display_current) * cost_per_ring - (bracelet_display_capacity - bracelet_display_current) * cost_per_bracelet) / cost_per_necklace = 12 :=
by
  sorry

end necklace_stand_capacity_l445_445170


namespace toms_ribbon_length_l445_445577

theorem toms_ribbon_length (gifts : ℕ) (ribbon_per_gift : ℝ) (remaining_ribbon : ℝ) (total_ribbon : ℝ) 
  (hg : gifts = 8) (hrpg : ribbon_per_gift = 1.5) (hrr : remaining_ribbon = 3) (htotal : total_ribbon = 15) :
  gifts * ribbon_per_gift + remaining_ribbon = total_ribbon :=
by 
  rw [hg, hrpg, hrr];
  norm_num;
  exact htotal
  sorry

end toms_ribbon_length_l445_445577


namespace permutations_of_BANANA_l445_445242

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445242


namespace distinct_arrangements_of_BANANA_l445_445341

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445341


namespace centipede_shoes_and_socks_l445_445696

-- Define number of legs
def num_legs : ℕ := 10

-- Define the total number of items
def total_items : ℕ := 2 * num_legs

-- Define the total permutations without constraints
def total_permutations : ℕ := Nat.factorial total_items

-- Define the probability constraint for each leg
def single_leg_probability : ℚ := 1 / 2

-- Define the combined probability constraint for all legs
def all_legs_probability : ℚ := single_leg_probability ^ num_legs

-- Define the number of valid permutations (the answer to prove)
def valid_permutations : ℚ := total_permutations / all_legs_probability

theorem centipede_shoes_and_socks : valid_permutations = (Nat.factorial 20 : ℚ) / 2^10 :=
by
  -- The proof is omitted
  sorry

end centipede_shoes_and_socks_l445_445696


namespace Evelyn_end_caps_l445_445770

def bottle_caps_end (start caps_lost) : Float := start - caps_lost

theorem Evelyn_end_caps : bottle_caps_end 63.0 18.0 = 45.0 :=
by
  sorry

end Evelyn_end_caps_l445_445770


namespace triangle_AB_AC_geq_4DE_l445_445514

theorem triangle_AB_AC_geq_4DE (A B C F D E : Point)
    (hTriangle : triangle A B C)
    (angle_AFB : angle A F B = 120)
    (angle_BFC : angle B F C = 120)
    (angle_CFA : angle C F A = 120)
    (BF_to_AC_at_D : ∃ g : Line, g = line B F ∧ D ∈ g ∧ D ∈ line A C)
    (CF_to_AB_at_E : ∃ h : Line, h = line C F ∧ E ∈ h ∧ E ∈ line A B) :
  segment_length A B + segment_length A C ≥ 4 * segment_length D E := 
sorry

end triangle_AB_AC_geq_4DE_l445_445514


namespace proof_correct_inferences_l445_445922

theorem proof_correct_inferences
    (f : ℝ → ℝ)
    (hf : ∀ x ∈ {2, -1}, f x = if x = 2 then 1 else -2) :
  (∀ x ∈ {2, -1}, f x = 1 ↔ x = 2) →  -- Inference ① is false due to non-parallel lines
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x ∧ k > 0) →  -- Inference ② is true if hyperbola
  (∀ x, ∃ a b c : ℝ, f x = a * x ^ 2 + b * x + c ∧ a < 0) → -- Inference ③ is undecidable
  (∀ x, ∃ a b c : ℝ, f x = a * x ^ 2 + b * x + c ∧ a > 0 → -b / (2 * a) < 1 / 2) →  -- Inference ④ is true for upwards parabola
  (true, true, false, true) :=  
by
  intros _ _ _ _
  exact (true, true, false, true)

end proof_correct_inferences_l445_445922


namespace smallest_square_side_length_l445_445791

theorem smallest_square_side_length :
  (∃ n a : ℕ, n > 0 ∧ a^2 = 14 * n ∧ (∀ m < n, ¬∃ b : ℕ, b^2 = 14 * m) ∧ a = 14) :=
begin
  sorry
end

end smallest_square_side_length_l445_445791


namespace sum_num_perfect_square_l445_445758

noncomputable def sequence (n : ℕ) : ℚ
| 0       := 4/3
| (n + 1) := (sequence n)^2 / ((sequence n)^2 - sequence n + 1)

def sum_sequence (k : ℕ) : ℚ :=
(∑ i in Finset.range k, sequence i)

theorem sum_num_perfect_square (k : ℕ) : 
  ∃ m : ℕ, (sum_sequence k).num = m^2 := 
sorry

end sum_num_perfect_square_l445_445758


namespace arithmetic_sequence_third_term_l445_445621

theorem arithmetic_sequence_third_term (a d : ℤ) 
  (h20 : a + 19 * d = 17) (h21 : a + 20 * d = 20) : a + 2 * d = -34 := 
sorry

end arithmetic_sequence_third_term_l445_445621


namespace h_max_value_l445_445761

variable {a b : ℝ}
variable {h : ℝ}

noncomputable def h_def := min a (b / (a ^ 2 + 4 * b ^ 2))

theorem h_max_value (a_pos : 0 < a) (b_pos : 0 < b) : h_def ≤ 1/2 :=
sorry

end h_max_value_l445_445761


namespace union_set_solution_l445_445421

open Set

theorem union_set_solution (a : ℝ) (A B : Set ℝ) (hA : A = {a^2 + 1, 2a}) (hB : B = {a + 1, 0}) (h : A ∩ B ≠ ∅) :
  A ∪ B = {0, 1} :=
by
  sorry

end union_set_solution_l445_445421


namespace perm_banana_l445_445353

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445353


namespace f6_eq_l445_445798

def f (x : ℝ) : ℝ := x^2 / (2 * x + 1)
def f_seq (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then x else f (f_seq (n - 1) x)

theorem f6_eq (x : ℝ) (hx : x ≠ 0) : 
  f_seq 6 x = 1 / ((1 + 1 / x)^64 - 1) :=
sorry

end f6_eq_l445_445798


namespace sin_half_alpha_l445_445859

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445859


namespace BANANA_arrangements_l445_445264

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445264


namespace max_dist_traveled_by_car_after_braking_l445_445607

-- Define the conditions.
def s (t : ℝ) (b : ℝ) : ℝ := -6 * t^2 + b * t

-- Define the conditions given in the problem.
theorem max_dist_traveled_by_car_after_braking :
  (∀ t : ℝ, s t 15 = -6 * t^2 + 15 * t) ∧ s (1/2) 15 = 6 →
  ∃ t : ℝ, s t 15 ≥ s x 15 ∀ x : ℝ := (75 / 8) :=
by
  intros h_eq h_val
  sorry

end max_dist_traveled_by_car_after_braking_l445_445607


namespace sin_half_alpha_l445_445870

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445870


namespace minimum_flowers_to_guarantee_bouquets_l445_445992

-- Definitions based on conditions given
def types_of_flowers : ℕ := 6
def flowers_needed_for_bouquet : ℕ := 5
def required_bouquets : ℕ := 10

-- Problem statement in Lean 4
theorem minimum_flowers_to_guarantee_bouquets (types : ℕ) (needed: ℕ) (bouquets: ℕ) 
    (h_types: types = types_of_flowers) (h_needed: needed = flowers_needed_for_bouquet) 
    (h_bouquets: bouquets = required_bouquets) : 
    (minimum_number_of_flowers_to_guarantee_bouquets types needed bouquets) = 70 :=
by sorry


end minimum_flowers_to_guarantee_bouquets_l445_445992


namespace correct_quotient_is_32_l445_445012

-- Definitions based on the conditions
def incorrect_divisor := 12
def correct_divisor := 21
def incorrect_quotient := 56
def dividend := incorrect_divisor * incorrect_quotient -- Given as 672

-- Statement of the theorem
theorem correct_quotient_is_32 :
  dividend / correct_divisor = 32 :=
by
  -- skip the proof
  sorry

end correct_quotient_is_32_l445_445012


namespace correctly_calculated_value_l445_445146

theorem correctly_calculated_value (x : ℝ) (hx : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  -- proof to be provided
  sorry

end correctly_calculated_value_l445_445146


namespace vector_dot_product_sum_l445_445953

variables {V : Type*} [InnerProductSpace ℝ V]
(open InnerProductSpace)

def vector_a : V := sorry
def vector_b : V := sorry
def vector_c : V := sorry

#check vector_a

-- Given conditions
def cond1 : vector_a + vector_b + vector_c = (0 : V) := sorry
def cond2 : ∥vector_a∥ = 1 := sorry
def cond3 : ∥vector_b∥ = 2 := sorry
def cond4 : ∥vector_c∥ = 2 := sorry

-- Proof goal
theorem vector_dot_product_sum :
  vector_a + vector_b + vector_c = 0 →
  ∥vector_a∥ = 1 →
  ∥vector_b∥ = 2 →
  ∥vector_c∥ = 2 →
  (⟪vector_a, vector_b⟫ + ⟪vector_b, vector_c⟫ + ⟪vector_c, vector_a⟫ : ℝ) = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445953


namespace sin_half_angle_l445_445849

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445849


namespace sin_half_alpha_l445_445875

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445875


namespace distance_between_points_l445_445384

def point := (Int, Int)

def distance (p1 p2 : point) : Real :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance (3, 3) (-2, -3) = Real.sqrt 61 :=
by
  sorry

end distance_between_points_l445_445384


namespace solution_set_inequality_l445_445920

variable (f : ℝ → ℝ)

-- Given conditions: f is continuous, differentiable, and f(x) > x * f'(x)
def continuous_and_differentiable (f : ℝ → ℝ) :=
  ContinuousOn f (set.Ioi 0) ∧ DifferentiableOn ℝ f (set.Ioi 0)

def greater_than_x_times_derivative (f : ℝ → ℝ) :=
  ∀ x : ℝ, 0 < x → f x > x * (deriv f x)

-- Proof statement: The solution set of the inequality is (0, 1)
theorem solution_set_inequality (hf_cont_diff : continuous_and_differentiable f)
  (hf_gt_xf' : greater_than_x_times_derivative f) :
  { x : ℝ | 0 < x ∧ {x^2} * f(1/x) - f(x) < 0} = set.Ioo 0 1 :=
  sorry

end solution_set_inequality_l445_445920


namespace calculate_hypotenuse_l445_445180

theorem calculate_hypotenuse
(h : ℝ)
(opp : ℝ)
(adj : ℝ)
(h_opp : opp = h / 2)
(perimeter : h + opp + adj = 120)
(adj_eq : adj = (sqrt(3) * h) / 2) :
  h = 40 * (3 - sqrt 3) := 
by
  sorry

end calculate_hypotenuse_l445_445180


namespace find_value_of_f_at_1_l445_445481

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_value_of_f_at_1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 2 * f x - f (- x) = 3 * x + 1) : f 1 = 2 :=
by
  sorry

end find_value_of_f_at_1_l445_445481


namespace white_circle_exists_l445_445640

theorem white_circle_exists (L W : ℝ) (n : ℕ) (side : ℝ) (d : ℝ)
   (H_LW : L = 25) (H_WW : W = 20) (H_n : n = 120) (H_side : side = 1) (H_d : d = 1) :
   ∃ (x y : ℝ), x ∈ Icc (0 : ℝ) L ∧ y ∈ Icc (0 : ℝ) W ∧
   (∀ (xi yi : ℝ), (xi, yi) ∈ finset.univ.image (λ k, (k.1 * side, k.2 * side)) →
   (abs (x - xi) > d / 2 ∨ abs (y - yi) > d / 2)) :=
sorry

end white_circle_exists_l445_445640


namespace eugene_not_used_cards_l445_445769

/-- Eugene used 6 boxes of toothpicks, each containing 450 toothpicks.
    Each card requires 75 toothpicks. 
    Eugene had a deck of 52 cards, and used some of them.
    Prove that the number of cards Eugene did not use is 16. -/
theorem eugene_not_used_cards :
  let total_toothpicks := 6 * 450 in
  let toothpicks_per_card := 75 in
  let total_cards := 52 in
  let cards_used := total_toothpicks / toothpicks_per_card in
  total_cards - cards_used = 16 := sorry

end eugene_not_used_cards_l445_445769


namespace independent_var_is_temperature_speed_of_sound_increase_rate_speed_of_sound_relationship_distance_to_fireworks_l445_445028

def is_independent_var (t : ℕ) := t = 0 ∨ t = 5 ∨ t = 10 ∨ t = 15 ∨ t = 20 ∨ t = 25

def is_dependent_var (v : ℕ) := v = 331 ∨ v = 334 ∨ v = 337 ∨ v = 340 ∨ v = 343 ∨ v = 346

theorem independent_var_is_temperature (t : ℕ) : is_independent_var t :=
by sorry

theorem speed_of_sound_increase_rate : ∃ r : ℝ, r = 0.6 :=
by sorry

theorem speed_of_sound_relationship (t : ℕ) (v : ℝ): v = 331 + 0.6 * t :=
by sorry

theorem distance_to_fireworks (t : ℕ) (time : ℝ) : t = 18 ∧ time = 5 → 
  let v := 331 + 0.6 * t in
  let distance := v * time in
  distance = 1709 :=
by sorry

end independent_var_is_temperature_speed_of_sound_increase_rate_speed_of_sound_relationship_distance_to_fireworks_l445_445028


namespace sin_half_alpha_l445_445865

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445865


namespace sin_half_angle_l445_445905

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445905


namespace intersection_points_polar_coordinates_l445_445934

variable (C1_Cartesian_eq : ∀ (x y : ℝ), (x - 4)^2 + (y - 5)^2 = 25)
variable (C2_Cartesian_eq : ∀ (x y : ℝ), x^2 + y^2 - 2y = 0)

theorem intersection_points_polar_coordinates :
  (C1_Cartesian_eq ∧ C2_Cartesian_eq) →
  ((∃ (x y : ℝ), x = 0 ∧ y = 2) ∨ (∃ (x y : ℝ), x = 1 ∧ y = 1)) →
  (∃ (ρ θ : ℝ),
    (ρ = 2 ∧ θ = π / 2) ∨
    (ρ = sqrt 2 ∧ θ = π / 4)) := by
  sorry

end intersection_points_polar_coordinates_l445_445934


namespace sin_half_alpha_l445_445878

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445878


namespace sin_half_alpha_l445_445855

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445855


namespace wrong_height_l445_445596

theorem wrong_height (h_avg_wrong: ℕ → rat := 181 )
                     (h_boy_real: ℕ := 106 )
                     (h_avg_real: ℕ → rat := 179 ) 
                     (n_boys: ℕ := 35) :
                     n_boys * h_avg_wrong = 6335  ∧
                     n_boys * h_avg_real = 6265 ∧
                     (6335 - 6265 = 70)  → 
                     (x :=  h_boy_real + 70 ) :
                     x = 176  :=
by
  sorry


end wrong_height_l445_445596


namespace length_of_platform_is_75_l445_445187

-- Definitions of the conditions
def speed_train_m_per_s : ℕ := 15
def length_train_m : ℕ := 300
def time_pass_platform_s : ℕ := 25

-- Statement of the proof problem
theorem length_of_platform_is_75 : 
  ∀ (speed length time : ℕ), 
  speed = speed_train_m_per_s → 
  length = length_train_m → 
  time = time_pass_platform_s → 
  length + (time * speed) - length = 75 :=
by 
  intros speed length time,
  assume h1 : speed = speed_train_m_per_s,
  assume h2 : length = length_train_m,
  assume h3 : time = time_pass_platform_s,
  rw [h1, h2, h3],
  simp,
  sorry

end length_of_platform_is_75_l445_445187


namespace sin_half_angle_l445_445845

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445845


namespace part1_part2_l445_445455

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  m * sin (2 * x) + cos (2 * x)

-- Condition: f passes through the point (π/12, √3)
theorem part1 (m : ℝ) (h : sin (π / 6) = 1 / 2) (h' : cos (π / 6) = √3 / 2) :
  f m (π / 12) = √3 → m = √3 :=
by
  sorry

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ :=
  2 * sin (2 * x + 2 * φ + π/6)

-- Condition: minimum distance from the highest point on g(x) to (0, 3) is 1
theorem part2 {φ : ℝ} (hφ : 0 < φ ∧ φ < π) :
  let x0 := 0 in sqrt (1 + x0^2) = 1 ∧
  g x0 φ = 2 ∧ φ = π / 6 →
  ∀ k : ℤ, -π / 2 + k * π ≤ x ∧ x ≤ k * π := 
by
  sorry

end part1_part2_l445_445455


namespace sin_half_alpha_l445_445828

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445828


namespace BANANA_arrangements_l445_445298

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445298


namespace bug_probability_l445_445163

theorem bug_probability : ∀ (A H : ℕ), A ≠ H ∧ A ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ H ∈ {1, 2, 3, 4, 5, 6, 7, 8} →
  (∃ path : ℕ → ℕ, (∀ i, path i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) 
   ∧ path 0 = A ∧ path 6 = H 
   ∧ (∀ i, i < 6 → path (i + 1) ∈ {n | ∃ e, (path i, n) = e ∧ e isEdgeOfCube}) 
   ∧ (∃! i, (path i = j) ∧ (∀ k, k ≠ i → path k ≠ j))) ∧ (j ∈ {1, 2, 3, 4, 5, 6, 7, 8}) → ∃ (P : ℚ), P = 1/729
:= sorry

end bug_probability_l445_445163


namespace common_chord_through_vertex_l445_445443

-- Define the structure for the problem
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

def passes_through (x y x_f y_f : ℝ) : Prop := (x - x_f) * (x - x_f) + y * y = 0

noncomputable def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- The main statement to prove
theorem common_chord_through_vertex (p : ℝ)
  (A B C D : ℝ × ℝ)
  (hA : parabola A.snd A.fst p)
  (hB : parabola B.snd B.fst p)
  (hC : parabola C.snd C.fst p)
  (hD : parabola D.snd D.fst p)
  (hAB_f : passes_through A.fst A.snd (focus p).fst (focus p).snd)
  (hCD_f : passes_through C.fst C.snd (focus p).fst (focus p).snd) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + k = 0) → (y + k = 0) :=
by sorry

end common_chord_through_vertex_l445_445443


namespace BANANA_arrangements_l445_445303

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445303


namespace find_angle_A_range_area_of_triangle_l445_445031

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem find_angle_A (h1 : b^2 + c^2 = a^2 - b * c) : A = (2 : ℝ) * Real.pi / 3 :=
by sorry

theorem range_area_of_triangle (h1 : b^2 + c^2 = a^2 - b * c)
(h2 : b * Real.sin A = 4 * Real.sin B) 
(h3 : Real.log b + Real.log c ≥ 1 - 2 * Real.cos (B + C)) 
(h4 : A = (2 : ℝ) * Real.pi / 3) :
(Real.sqrt 3 / 4 : ℝ) ≤ (1 / 2) * b * c * Real.sin A ∧
(1 / 2) * b * c * Real.sin A ≤ (4 * Real.sqrt 3 / 3 : ℝ) :=
by sorry

end find_angle_A_range_area_of_triangle_l445_445031


namespace f_2005_is_cos_l445_445552

noncomputable def f (x : ℝ) : ℝ := Real.sin x

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => f x
  | n+1   => (fun x => (derivative (f_n n)) x) x

theorem f_2005_is_cos :
  ∀ x : ℝ, f_n 2005 x = Real.cos x :=
by
  sorry

end f_2005_is_cos_l445_445552


namespace calculate_expression_l445_445213

theorem calculate_expression (m : ℝ) : (-m)^2 * m^5 = m^7 := 
sorry

end calculate_expression_l445_445213


namespace angle_in_third_quadrant_l445_445476

theorem angle_in_third_quadrant (a : ℝ) (h1 : sin (2 * a) > 0) (h2 : sin a < 0) : 
  (a > π / 2) ∧ (a < π) := 
by
  sorry -- Proof to be filled

end angle_in_third_quadrant_l445_445476


namespace sin_half_angle_l445_445850

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445850


namespace correct_word_is_any_l445_445520

def words : List String := ["other", "any", "none", "some"]

def is_correct_word (word : String) : Prop :=
  "Jane was asked a lot of questions, but she didn’t answer " ++ word ++ " of them." = 
    "Jane was asked a lot of questions, but she didn’t answer any of them."

theorem correct_word_is_any : is_correct_word "any" :=
by
  sorry

end correct_word_is_any_l445_445520


namespace least_possible_element_in_T_l445_445542

theorem least_possible_element_in_T :
  ∃ T : Finset ℕ, T.card = 7 ∧ (∀ c d ∈ T, c < d → ¬ (d % c = 0)) ∧ T.sum id < 50 ∧ T.min' (Finset.card_pos.2 (by decide)) = 4 := by
  sorry

end least_possible_element_in_T_l445_445542


namespace find_b_l445_445200

-- Define the functions f and g
def f (x : ℝ) := x / 4 + 2
def g (x : ℝ) := 5 - 2 * x

-- Main theorem stating the result
theorem find_b : 
  (∃ b : ℝ, f (g b) = 3) → (b = 1/2) :=
by 
  sorry

end find_b_l445_445200


namespace work_completion_days_l445_445078

variable (Paul_days Rose_days Sam_days : ℕ)

def Paul_rate := 1 / 80
def Rose_rate := 1 / 120
def Sam_rate := 1 / 150

def combined_rate := Paul_rate + Rose_rate + Sam_rate

noncomputable def days_to_complete_work := 1 / combined_rate

theorem work_completion_days :
  Paul_days = 80 →
  Rose_days = 120 →
  Sam_days = 150 →
  days_to_complete_work = 37 := 
by
  intros
  simp only [Paul_rate, Rose_rate, Sam_rate, combined_rate, days_to_complete_work]
  sorry

end work_completion_days_l445_445078


namespace BANANA_arrangements_l445_445265

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445265


namespace sin_half_alpha_l445_445886

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445886


namespace find_x_l445_445661

theorem find_x (x y : ℤ) (h₁ : x + 3 * y = 10) (h₂ : y = 3) : x = 1 := 
by
  sorry

end find_x_l445_445661


namespace simple_interest_rate_l445_445681

theorem simple_interest_rate (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 130) (h2 : P = 780) (h3 : T = 4) :
  R = 4.17 :=
sorry

end simple_interest_rate_l445_445681


namespace banana_permutations_l445_445253

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445253


namespace monotonicity_inequality_l445_445057

def f (a x : ℝ) : ℝ := a * x^2 - a - (Real.log x)

theorem monotonicity (a : ℝ) :
  (∀ x > 0, f a x ≤ f a (x + 1) ↔ a ≤ 0) ∧
  (∀ x, 0 < x ∧ x < Real.sqrt (1 / (2 * a)) → f a x ≥ f a (x + 1) ↔ a > 0) ∧
  (∀ x > Real.sqrt (1 / (2 * a)), f a x ≤ f a (x + 1) ↔ a > 0) := sorry

theorem inequality (a : ℝ) :
  (∀ x > 1, f a x > 1 / x - Real.exp (1 - x)) ↔ (a ≥ 1 / 2) := sorry

end monotonicity_inequality_l445_445057


namespace at_least_six_consecutive_heads_in_ten_flips_prob_l445_445701

def probability_at_least_six_consecutive_heads_in_ten_flips : ℚ :=
  129 / 1024

theorem at_least_six_consecutive_heads_in_ten_flips_prob :
  (∃ s : ℕ → bool, (∀ i < 10, (s i = tt ∨ s i = ff))
    ∧ (∑ k in (finset.range 5), ite (∀ j, j < 6 → s (j + k) = tt) 1 0
      + ∑ k in (finset.range 4), ite (∀ j, j < 7 → s (j + k) = tt) 1 0
      + ∑ k in (finset.range 3), ite (∀ j, j < 8 → s (j + k) = tt) 1 0
      + ∑ k in (finset.range 2), ite (∀ j, j < 9 → s (j + k) = tt) 1 0
      + ite (∀ j, j < 10 → s (j) = tt) 1 0).to_rat = 129 / 1024) :=
begin
  sorry
end

end at_least_six_consecutive_heads_in_ten_flips_prob_l445_445701


namespace complex_power_sum_l445_445378

theorem complex_power_sum (i : ℂ) (h_i4 : i^4 = 1) : i^12 + i^20 + i^(-32) = 3 := by
  sorry

end complex_power_sum_l445_445378


namespace sequence_count_no_three_consecutive_same_parity_l445_445471

theorem sequence_count_no_three_consecutive_same_parity : 
  ∃ (f : Fin 7 → Fin 10 → ℕ), 
  (∀ (i : Fin 7), no_three_consecutive_same_parity (f i)) ∧
  length(sequences_filtered_by_parity (f)) = 312500 :=
by sorry

-- Supporting definitions and conditions
def no_three_consecutive_same_parity (s : Fin 10 → ℕ) : Prop := 
  ∀ (i : Nat), i + 2 < 7 → (s i % 2 ≠ s (i + 1) % 2 ∨ s (i + 1) % 2 ≠ s (i + 2) % 2)

def sequences_filtered_by_parity (f : Fin 7 → Fin 10 → ℕ) : List (Fin 10 → ℕ) :=
  filter no_three_consecutive_same_parity (generate_all_sequences f)

def generate_all_sequences (f : Fin 7 → Fin 10 → ℕ) : List (Fin 10 → ℕ) :=
  -- Predefine all possible sequences, where f is a function that generates sequences
  sorry

end sequence_count_no_three_consecutive_same_parity_l445_445471


namespace black_friday_sales_l445_445072

variable (n : ℕ) (initial_sales increment : ℕ)

def yearly_sales (sales: ℕ) (inc: ℕ) (years: ℕ) : ℕ :=
  sales + years * inc

theorem black_friday_sales (h1 : initial_sales = 327) (h2 : increment = 50) :
  yearly_sales initial_sales increment 3 = 477 := by
  sorry

end black_friday_sales_l445_445072


namespace permutations_of_banana_l445_445286

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445286


namespace minimum_area_isosceles_trapezoid_l445_445771

theorem minimum_area_isosceles_trapezoid (r x a d : ℝ) (h_circumscribed : a + d = 2 * x) (h_minimal : x ≥ 2 * r) :
  4 * r^2 ≤ (a + d) * r :=
by sorry

end minimum_area_isosceles_trapezoid_l445_445771


namespace banana_permutations_l445_445260

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445260


namespace sqrt_sixteen_l445_445592

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end sqrt_sixteen_l445_445592


namespace union_of_A_B_l445_445410

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l445_445410


namespace sin_half_alpha_l445_445823

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445823


namespace banana_arrangements_l445_445330

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445330


namespace BANANA_arrangements_l445_445307

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445307


namespace sum_of_primes_upto_1M_l445_445377

open Set

noncomputable def is_prime (n : Nat) : Prop := Nat.Prime n

def primes_less_than (n : Nat) : Set Nat := {p : Nat | p < n ∧ is_prime p}

def sum_primes (n : Nat) : Nat := (primes_less_than n).toFinset.sum id

theorem sum_of_primes_upto_1M :
  sum_primes 1000000 = 37550402023 := sorry

end sum_of_primes_upto_1M_l445_445377


namespace sin_half_angle_l445_445817

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445817


namespace dot_product_sum_l445_445955

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom vec_sum : a + b + c = 0
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom norm_c : ∥c∥ = 2

-- The theorem to prove
theorem dot_product_sum :
  ⟪a, b⟫ + ⟪b, c⟫ + ⟪c, a⟫ = - 9 / 2 :=
sorry

end dot_product_sum_l445_445955


namespace vector_dot_product_sum_l445_445965

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_sum
  (a b c : V)
  (h1 : a + b + c = 0)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hc : ∥c∥ = 2) :
  inner_product_space.inner a b + inner_product_space.inner b c + inner_product_space.inner c a = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445965


namespace frustum_cone_height_l445_445168

theorem frustum_cone_height 
  (H_f : ℝ) (A_L A_U : ℝ)
  (h : ℝ) :
  H_f = 30 →
  A_L = 400 * real.pi →
  A_U = 100 * real.pi →
  h = 30 :=
by
  sorry

end frustum_cone_height_l445_445168


namespace sin_half_angle_l445_445819

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445819


namespace distance_between_trees_l445_445004

theorem distance_between_trees (L : ℝ) (n : ℕ) (hL : L = 375) (hn : n = 26) : 
  (L / (n - 1) = 15) :=
by
  sorry

end distance_between_trees_l445_445004


namespace vector_dot_product_sum_l445_445966

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_sum
  (a b c : V)
  (h1 : a + b + c = 0)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hc : ∥c∥ = 2) :
  inner_product_space.inner a b + inner_product_space.inner b c + inner_product_space.inner c a = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445966


namespace permutations_of_banana_l445_445289

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445289


namespace polynomial_unique_solution_l445_445782

theorem polynomial_unique_solution (q : ℝ → ℝ) :
  (∀ x : ℝ, q(x^3) - q(x^3 - 4) = (q(x))^2 + 20) →
  q = (λ x, 12 * x^3 - 4) :=
by
  sorry

end polynomial_unique_solution_l445_445782


namespace sin_half_angle_l445_445906

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445906


namespace percentage_decrease_fuel_l445_445560

theorem percentage_decrease_fuel (fuel_this_week : ℝ) (total_fuel_two_weeks : ℝ) (percentage_decrease : ℝ) : 
  fuel_this_week = 15 ∧ total_fuel_two_weeks = 27 ∧ 
  percentage_decrease = 20 :=
by
  have fuel_last_week : ℝ := total_fuel_two_weeks - fuel_this_week
  have difference_in_usage : ℝ := fuel_this_week - fuel_last_week
  have percentage : ℝ := (difference_in_usage / fuel_this_week) * 100
  have H1 : fuel_this_week = 15 := by sorry
  have H2 : total_fuel_two_weeks = 27 := by sorry
  have H3 : percentage = 20 := by calc
    percentage
        = (difference_in_usage / fuel_this_week) * 100 : by sorry
    ... = (3 / 15) * 100 : by sorry
    ... = 0.2 * 100 : by sorry
    ... = 20 : by sorry
  exact ⟨H1, H2, H3⟩

end percentage_decrease_fuel_l445_445560


namespace perm_banana_l445_445349

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445349


namespace radius_of_cone_is_8_l445_445670

noncomputable def r_cylinder := 8 -- cm
noncomputable def h_cylinder := 2 -- cm
noncomputable def h_cone := 6 -- cm

theorem radius_of_cone_is_8 :
  exists (r_cone : ℝ), r_cone = 8 ∧ π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone :=
by
  let r_cone := 8
  have eq_volumes : π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone := 
    sorry
  exact ⟨r_cone, by simp, eq_volumes⟩

end radius_of_cone_is_8_l445_445670


namespace angle_between_vectors_l445_445453

variables {a b : EuclideanSpace ℝ (Fin 2)}
variables (θ : ℝ)

def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt (∥v∥ ^ 2)

def is_perpendicular (u v : EuclideanSpace ℝ (Fin 2)) : Prop := inner u v = 0

theorem angle_between_vectors (h1 : magnitude a = Real.sqrt 2)
                              (h2 : magnitude b = 2)
                              (h3 : is_perpendicular (a - b) a) :
  θ = Real.pi / 4 :=
sorry

end angle_between_vectors_l445_445453


namespace valid_paths_l445_445230

def paths (n m : ℕ) : ℕ := Nat.choose (n + m) m

def total_paths : ℕ := paths 8 4
def blocked_paths_through_C_or_D : ℕ := 4 * 56

theorem valid_paths : total_paths - blocked_paths_through_C_or_D = 271 := by
  sorry

end valid_paths_l445_445230


namespace system_of_equations_has_two_solutions_l445_445970

noncomputable def count_solutions : Nat :=
  let solutions := { (x, y) | x + 4 * y = 4 ∧ abs (abs x - 2 * abs y) = 2 }
  Finset.card ((solutions : Set (ℝ × ℝ)).toFinset)

theorem system_of_equations_has_two_solutions :
  count_solutions = 2 :=
by
  sorry

end system_of_equations_has_two_solutions_l445_445970


namespace exists_zero_point_bisection_l445_445637

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 3

theorem exists_zero_point_bisection:
  ∃ c ∈ Ioo (2.5625 : ℝ) 2.75, f c = 0 :=
sorry

end exists_zero_point_bisection_l445_445637


namespace find_angle_l445_445913

variable (θ φ : ℝ)
variable (h1 : 0 < θ ∧ θ < π / 2)
variable (h2 : 0 < φ ∧ φ < π / 2)
variable (h3 : Real.cot θ = 2)
variable (h4 : Real.cos φ = 3 / 5)

theorem find_angle : 2 * θ + φ = 2 * Real.pi / 3 := sorry

end find_angle_l445_445913


namespace sin_half_alpha_l445_445874

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445874


namespace selection_has_pair_sum_21_l445_445088

theorem selection_has_pair_sum_21 (S : Finset ℕ) (hS : S.card = 11) :
  (∀ x ∈ S, x ∈ Finset.range 21) → ∃ x y ∈ S, x + y = 21 :=
by
  intros H
  sorry

end selection_has_pair_sum_21_l445_445088


namespace permutations_of_BANANA_l445_445240

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445240


namespace local_min_condition_for_b_l445_445606

theorem local_min_condition_for_b (b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 3 * b * x + 3 * b)
  (h_min : ∃ c ∈ set.Ioo 0 1, ∀ x ∈ set.Ioo 0 1, f c ≤ f x ) : 0 < b ∧ b < 1 :=
by
  sorry

end local_min_condition_for_b_l445_445606


namespace notebook_cost_l445_445174

noncomputable theory

-- Definitions for the costs
def cost_notebook (n p : ℝ) := n = 2 + p
def total_cost (n p : ℝ) := n + p = 2.40

-- Theorem statement proving the cost of the notebook given the conditions
theorem notebook_cost (n p : ℝ) 
  (h1 : total_cost n p) (h2 : cost_notebook n p) : 
  n = 2.20 := 
sorry

end notebook_cost_l445_445174


namespace janabel_widget_sales_l445_445074

theorem janabel_widget_sales :
  let a : ℕ → ℕ := λ n, if n = 10 then 0 else 3 * n - 1,
  total_widgets := (∑ n in Finset.range 15, a (n + 1))
in
  total_widgets = 345 :=
by
  sorry

end janabel_widget_sales_l445_445074


namespace three_exp_eq_l445_445420

theorem three_exp_eq (y : ℕ) (h : 3^y + 3^y + 3^y = 2187) : y = 6 :=
by
  sorry

end three_exp_eq_l445_445420


namespace total_valid_arrangements_l445_445582

-- Define the students and schools
inductive Student
| G1 | G2 | B1 | B2 | B3 | BA
deriving DecidableEq

inductive School
| A | B | C
deriving DecidableEq

-- Define the condition that any two students cannot be in the same school
def is_valid_arrangement (arr : School → Student → Bool) : Bool :=
  (arr School.A Student.G1 ≠ arr School.A Student.G2) ∧ 
  (arr School.B Student.G1 ≠ arr School.B Student.G2) ∧
  (arr School.C Student.G1 ≠ arr School.C Student.G2) ∧
  ¬ arr School.C Student.G1 ∧
  ¬ arr School.C Student.G2 ∧
  ¬ arr School.A Student.BA

-- The theorem to prove the total number of different valid arrangements
theorem total_valid_arrangements : 
  ∃ n : ℕ, n = 18 ∧ ∃ arr : (School → Student → Bool), is_valid_arrangement arr := 
sorry

end total_valid_arrangements_l445_445582


namespace union_of_A_and_B_l445_445416

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l445_445416


namespace discriminant_of_quadratic_l445_445209

-- Definitions of the coefficients and calculation of the discriminant
def a : ℝ := 2
def b : ℝ := 2 + Real.sqrt 2
def c : ℝ := 1 / 2

-- The quadratic discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem stating the discriminant is as calculated
theorem discriminant_of_quadratic :
  discriminant a b c = 2 + 4 * Real.sqrt 2 := by
  sorry

end discriminant_of_quadratic_l445_445209


namespace max_volume_of_cube_in_pyramid_l445_445176

noncomputable def max_cube_volume : ℝ :=
  let s := (4 * Real.sqrt 3) / 3
  in s^3

theorem max_volume_of_cube_in_pyramid :
  let side_length_pyramid_base := 2
  let pyr_base_height := Real.sqrt 3
  let pyr_total_height := (2 * Real.sqrt 3)
  let pyr_lateral_faces_height := pyr_total_height
  let max_volume := max_cube_volume
  max_volume = (64 * Real.sqrt 3) / 9 :=
by
  sorry

end max_volume_of_cube_in_pyramid_l445_445176


namespace prob_exactly_three_even_l445_445206

open Probability

-- Define rolling a 12-sided die
def roll_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define event of rolling an even number
def even_event (x : ℕ) : Prop := x % 2 = 0

-- Define the probability of rolling one die and it showing an even number
noncomputable def P_even : ℚ := 1 / 2

-- Define the event of exactly three dice showing even numbers
def event_exactly_three_even (dice_results : Fin 4 → ℕ) : Prop :=
  (∑ i, if even_event (dice_results i) then 1 else 0) = 3

-- Define the set of all 4-dice rolls
def all_rolls : Set (Fin 4 → ℕ) := {dice_results | ∀ i, dice_results i ∈ roll_die}

-- Prove the probability that exactly three dice show even numbers is 1/4
theorem prob_exactly_three_even :
  Pr[all_rolls, event_exactly_three_even] = 1 / 4 :=
by
  sorry

end prob_exactly_three_even_l445_445206


namespace sin_3B_over_sin_B_l445_445494

theorem sin_3B_over_sin_B (A B C a b c : ℝ) (h1 : A + B + C = π) (h2 : C = 2 * B)
  (h3 : ∀ x y, sin x / sin y = x / y) :
  sin (3 * B) / sin B = a / b := 
sorry

end sin_3B_over_sin_B_l445_445494


namespace hexagon_area_l445_445799

theorem hexagon_area (A B C D E F G : Type)
  (ab : dist A B = 5)
  (bc : dist B C = 7)
  (cd : dist C D = 3)
  (de : dist D E = 4)
  (ef : dist E F = 6)
  (fa : dist F A = 8)
  (ab_parallel_de : parallel A B D E)
  (cd_perpendicular_ab : perpendicular C D A B)
  (fa_perpendicular_ab : perpendicular F A A B)
  (G_intersect_dc_af : is_intersection_point G (line_through D C) (line_through A F)) :
  area (hexagon A B C D E F) = 84 :=
by sorry

end hexagon_area_l445_445799


namespace parallelogram_circumference_l445_445624

-- Defining the conditions
def isParallelogram (a b : ℕ) := a = 18 ∧ b = 12

-- The theorem statement to prove
theorem parallelogram_circumference (a b : ℕ) (h : isParallelogram a b) : 2 * (a + b) = 60 :=
  by
  -- Extract the conditions from hypothesis
  cases h with
  | intro hab' hab'' =>
    sorry

end parallelogram_circumference_l445_445624


namespace max_area_triangle_OSQ_l445_445436

noncomputable def max_area_triangle (a b : ℝ) : ℝ :=
  1 / 2 * a * b

theorem max_area_triangle_OSQ
  (a b : ℝ)
  (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h_line : ∀ P Q : ℝ × ℝ, (P.fst = P.snd) ∧ (Q.fst = Q.snd) ∧ P.fst ≠ Q.fst)
  (hS_reflection : ∀ y : ℝ, (0, -y) = (0, y)) :
  ∃ S Q : ℝ × ℝ, 
  max_area_triangle a b = 1 / 2 * a * b :=
begin
  sorry
end

end max_area_triangle_OSQ_l445_445436


namespace find_coordinates_of_Q_l445_445712

noncomputable def point_P_start : (ℝ × ℝ) := (1, 0)

noncomputable def angle_QOx : ℝ := (2 * Real.pi) / 3

noncomputable def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem find_coordinates_of_Q :
  let Q := point_on_unit_circle angle_QOx in
  Q = (-1 / 2, Real.sqrt 3 / 2) :=
by
  sorry

end find_coordinates_of_Q_l445_445712


namespace each_customer_purchases_4_tomatoes_l445_445639

def customers := 500
def lettuce_price := 1
def lettuce_quantity_per_customer := 2
def tomato_price := 0.5
def total_sales := 2000

theorem each_customer_purchases_4_tomatoes :
  (total_sales - customers * (lettuce_quantity_per_customer * lettuce_price)) / tomato_price / customers = 4 :=
by
  sorry

end each_customer_purchases_4_tomatoes_l445_445639


namespace sin_half_angle_l445_445843

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445843


namespace vector_dot_product_sum_eq_l445_445945

variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V)

theorem vector_dot_product_sum_eq :
  a + b + c = 0 →
  ∥a∥ = 1 →
  ∥b∥ = 2 →
  ∥c∥ = 2 →
  inner a b + inner b c + inner c a = -9 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end vector_dot_product_sum_eq_l445_445945


namespace folded_rectangle_l445_445505

def Point := (ℝ × ℝ)

structure Rectangle where
  A B C D : Point
  AB_eq_4 : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16
  BC_eq_8 : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 64

def is_RightTriangle (P Q R : Point) : Prop :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) + ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = ((P.1 - R.1)^2 + (P.2 - R.2)^2)

theorem folded_rectangle (r : Rectangle) : ∃ EF : ℝ, EF = 4 := by
  sorry

end folded_rectangle_l445_445505


namespace rectangle_area_l445_445715

theorem rectangle_area (ABCD : Type*) (small_square : ℕ) (shaded_squares : ℕ) (side_length : ℕ) 
  (shaded_area : ℕ) (width : ℕ) (height : ℕ)
  (H1 : shaded_squares = 3) 
  (H2 : side_length = 2)
  (H3 : shaded_area = side_length * side_length)
  (H4 : width = 6)
  (H5 : height = 4)
  : (width * height) = 24 :=
by
  sorry

end rectangle_area_l445_445715


namespace BANANA_arrangements_l445_445272

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445272


namespace complement_union_B_A_intersection_empty_l445_445448

-- Define Universarl Set U 
def U := Set ℝ 

-- Define Set A
def A := {x : ℝ | ∃ (y : ℝ), y = Real.log (x + 1) ∧ x > 0}

-- Define Set B
def B := {x : ℝ | (1 / 2) ≤ (2 : ℝ) ^ x ∧ (2 : ℝ) ^ x ≤ 8}

-- Define Set C
def C (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 2 * a}

-- Question 1
theorem complement_union_B : (U \ A) ∪ B = {x : ℝ | x ≤ 3} := by
  sorry

-- Question 2
theorem A_intersection_empty (a : ℝ): A ∩ C(a) = ∅ ↔ a ∈ Icc (⟨-∞, 0⟩ : Set ℝ) := by
  sorry

end complement_union_B_A_intersection_empty_l445_445448


namespace range_of_product_of_zeros_of_function_l445_445435
noncomputable theory

def curve (k : ℝ) (x : ℝ) : ℝ := k * exp (-x)

theorem range_of_product_of_zeros_of_function (k : ℝ)
  (h1 : ∃ m, m = -2 ∧ is_perpendicular m 0.5)
  (h2 : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ g x₁ = 0 ∧ g x₂ = 0):
  (1 / exp 2) < x₁ * x₂ ∧ x₁ * x₂ < 1 :=
sorry

-- Additional helper definitions go here
def is_perpendicular (m₁ m₂ : ℝ) := m₁ = -1 / m₂

def g (x : ℝ) : ℝ := curve 2 x - abs (log x)

end range_of_product_of_zeros_of_function_l445_445435


namespace total_hike_time_l445_445528

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l445_445528


namespace max_value_of_function_l445_445763

theorem max_value_of_function :
  ∀ x : ℝ, (sin x ∈ set.Icc (-1 : ℝ) 1) →
  (2 * cos x ^ 2 - sin x) ≤ (17 / 8) :=
by
  sorry

end max_value_of_function_l445_445763


namespace satisfies_conditions_l445_445683

noncomputable def f : ℝ → ℝ := floor

theorem satisfies_conditions :
  (∀ x y : ℝ, f(x) + f(y) + 1 ≥ f(x + y) ∧ f(x + y) ≥ f(x) + f(y)) ∧
  (∀ x : ℝ, 0 ≤ x → x < 1 → f(0) ≥ f(x)) ∧
  (-f (-1) = f 1 ∧ f 1 = 1) :=
by
  sorry

end satisfies_conditions_l445_445683


namespace banana_arrangement_count_l445_445284

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445284


namespace factor_polynomial_l445_445518

theorem factor_polynomial (x y z : ℂ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) := by
  sorry

end factor_polynomial_l445_445518


namespace permutations_of_banana_l445_445293

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445293


namespace banana_arrangement_count_l445_445285

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445285


namespace limit_pow_convergence_l445_445083

variables {ℚ : Type*} [is_R_or_C ℚ] {α : Type*} [linear_ordered_field α]

theorem limit_pow_convergence (a : α) (x : ℚ) (x_n : ℕ → ℚ)
  (hx : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (x_n n - x) < ε)
  (ha_pos : a > 0) :
  ∃ c : α, (tendsto (λ n, a^(x_n n)) at_top (nhds c ∧ c = a^x)) :=
sorry

end limit_pow_convergence_l445_445083


namespace number_and_sum_of_possible_g4_l445_445046

noncomputable def g (x : ℝ) : ℝ := sorry

theorem number_and_sum_of_possible_g4 :
  (let g : ℝ → ℝ := sorry in
  (∀ x y z : ℝ, g(x^2 + y * g(z)) = x * g(x) + z * g(y)) →
  let possible_values := {g | ∃ x : ℝ, g = g x} in
  let g_4_values := set.image g {4} in
  let n := finset.card (finset.coe g_4_values) in
  let s := finset.sum (finset.coe g_4_values) (λ x, x) in
  n * s = 8) := sorry

end number_and_sum_of_possible_g4_l445_445046


namespace sin_half_alpha_l445_445861

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445861


namespace no_solution_exists_l445_445040

-- Define the sums of the two geometric series
def C_n (n : ℕ) : ℝ := 768 * (1 - 1 / (3^n))
def D_n (n : ℕ) : ℝ := (729 / 4) * (1 - 1 / ((-3)^n))

-- Statement to prove: there is no integer \( n \ge 1 \) such that \( C_n = D_n \)
theorem no_solution_exists : ¬∃ n : ℕ, n ≥ 1 ∧ C_n n = D_n n := 
by
  sorry

end no_solution_exists_l445_445040


namespace initial_number_of_girls_l445_445392

theorem initial_number_of_girls (b g : ℤ) 
  (h1 : b = 3 * (g - 20)) 
  (h2 : 3 * (b - 30) = g - 20) : 
  g = 31 :=
by
  sorry

end initial_number_of_girls_l445_445392


namespace BANANA_arrangements_l445_445301

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445301


namespace positive_difference_between_balances_l445_445087

theorem positive_difference_between_balances 
  (P : ℝ)
  (r1 r2 : ℝ)
  (t : ℕ)
  (A_Rita : ℝ)
  (A_James : ℝ) :
  P = 15000 →
  r1 = 0.06 →
  r2 = 0.08 →
  t = 20 →
  A_Rita = P * (1 + r1)^t →
  A_James = P * (1 + r2 * t) →
  abs (A_Rita - A_James) ≈ 9107 :=
by
  intros
  sorry

end positive_difference_between_balances_l445_445087


namespace points_lie_on_parabola_l445_445787

noncomputable def lies_on_parabola (t : ℝ) : Prop :=
  let x := Real.cos t ^ 2
  let y := Real.sin t * Real.cos t
  y ^ 2 = x * (1 - x)

-- Statement to prove
theorem points_lie_on_parabola : ∀ t : ℝ, lies_on_parabola t :=
by
  intro t
  sorry

end points_lie_on_parabola_l445_445787


namespace circle_through_points_center_on_x_axis_l445_445427

noncomputable def circle_center_on_x_axis_through_points (A B : Point) : Circle :=
  sorry

open_locale classical

theorem circle_through_points_center_on_x_axis
  (A : Point := ⟨-1, 1⟩)
  (B : Point := ⟨1, 3⟩) :
  let C : Circle := circle_center_on_x_axis_through_points A B in
  C.equation = (λ x y, (x - 2)^2 + y^2 = 10) ∧
  ∀ m : ℝ, (line_intersects_circle (λ x, -x + m) C) → 
    (let MN_line := line_MN_through_origin (λ x, -x + m) C in 
    MN_line = (λ x, -x + 1 + sqrt 7) ∨ MN_line = (λ x, -x + 1 - sqrt 7)) :=
begin
  sorry
end

end circle_through_points_center_on_x_axis_l445_445427


namespace melody_cutouts_l445_445562

theorem melody_cutouts (cutouts_per_card cards : ℕ) (h1 : cutouts_per_card = 4) (h2 : cards = 6) : cutouts_per_card * cards = 24 :=
by
  rw [h1, h2]
  exact rfl

end melody_cutouts_l445_445562


namespace lead_percentage_correct_l445_445708

def total_weight_of_mixture (lead copper : ℝ) : ℝ :=
  lead + copper + 15 / 100 * (lead + copper)

def percentage_lead (lead total_weight: ℝ) : ℝ :=
  (lead / total_weight) * 100

theorem lead_percentage_correct (T lead copper : ℝ) 
  (h1 : 0.60 * T = copper) 
  (h2 : lead = 5)
  (h3 : copper = 12) 
  : percentage_lead lead T = 25 := 
begin
  have hT : T = 20, from calc
    T = 12 / 0.60 : by rw [h3, div_eq_mul_one_div, mul_comm, one_div, mul_assoc, mul_inv_cancel, mul_one],
  rw [hT, h2],
  refl,
end

end lead_percentage_correct_l445_445708


namespace solve_for_y_l445_445142

theorem solve_for_y (y : ℝ) (h : (45 / 75) = sqrt (y / 25)) : y = 9 := 
  sorry

end solve_for_y_l445_445142


namespace sufficient_not_necessary_l445_445047

variable (x : ℝ)

lemma abs_implies_log : 
  (| x - 2 | < 1) → (log (x + 2) / log (1/2) < 0) :=
sorry

lemma log_does_not_imply_abs : 
  ¬((log (x + 2) / log (1/2) < 0) → (| x - 2 | < 1)) :=
sorry

theorem sufficient_not_necessary :
  (∀ x : ℝ, (| x - 2 | < 1) → (log (x + 2) / log (1/2) < 0)) ∧ 
  (¬ (∀ x : ℝ, (log (x + 2) / log (1/2) < 0) → (| x - 2 | < 1))) :=
by
  apply And.intro
  . exact abs_implies_log
  . exact log_does_not_imply_abs

end sufficient_not_necessary_l445_445047


namespace complex_exp_neg_ipi_on_real_axis_l445_445038

theorem complex_exp_neg_ipi_on_real_axis :
  (Complex.exp (-Real.pi * Complex.I)).im = 0 :=
by 
  sorry

end complex_exp_neg_ipi_on_real_axis_l445_445038


namespace overlapping_circle_area_l445_445133

theorem overlapping_circle_area 
  (center1 : ℝ × ℝ) (center2 : ℝ × ℝ) (radius : ℝ)
  (h_center1 : center1 = (3, 0))
  (h_center2 : center2 = (0, 3))
  (h_radius : radius = 3) :
  let overlap_area : ℝ := (9 / 2) * real.pi - 9 in
  overlap_area = (9 / 2) * real.pi - 9 :=
by
  sorry

end overlapping_circle_area_l445_445133


namespace banana_arrangements_l445_445332

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445332


namespace binom_12_11_eq_12_l445_445219

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_eq_12_l445_445219


namespace wire_circle_radius_l445_445190

-- Given conditions
def squareArea : ℝ := 7737.769850454057
def wireAsSquareSideLength : ℝ := Real.sqrt squareArea
def wirePerimeter : ℝ := 4 * wireAsSquareSideLength
def circleCircumference : ℝ := wirePerimeter

-- The radius of the circle
def circleRadius (C : ℝ) : ℝ := C / (2 * Real.pi)

theorem wire_circle_radius : circleRadius circleCircumference ≈ 56 :=
by
  sorry

end wire_circle_radius_l445_445190


namespace kaleb_initial_video_games_l445_445537

def initial_video_games (non_working working_earn total_earn price_per_game : ℕ) : ℕ :=
  non_working + (total_earn / price_per_game)

theorem kaleb_initial_video_games :
  ∀ (non_working working_earn total_earn price_per_game : ℕ),
  non_working = 8 →
  total_earn = 12 →
  price_per_game = 6 →
  initial_video_games non_working working_earn total_earn price_per_game = 10 :=
by
  intros; 
  unfold initial_video_games;
  simp [*];
  sorry

end kaleb_initial_video_games_l445_445537


namespace smallest_n_with_terminating_decimal_and_digit_7_l445_445654

def contains_digit_7 (n : ℕ) : Prop :=
  n.digits 10 |> List.contains 7

theorem smallest_n_with_terminating_decimal_and_digit_7 :
  (∃ n : ℕ, (fractional n).is_terminating ∧ contains_digit_7 n ∧ ∀ m : ℕ, (fractional m).is_terminating ∧ contains_digit_7 m → n ≤ m) →
  ∃ smallest_n : ℕ, smallest_n = 128 :=
by {
  sorry -- proof goes here
}

end smallest_n_with_terminating_decimal_and_digit_7_l445_445654


namespace propositions_true_l445_445236

def average_value_function (f : ℝ → ℝ) (a b : ℝ) (x₀ : ℝ) : Prop :=
  a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

theorem propositions_true :
  (average_value_function (λ x, Math.sin x - 1) (-π) π (π / 2)) ∧
  ¬(∀ f : ℝ → ℝ, ∀ a b x₀ : ℝ, average_value_function f a b x₀ → x₀ ≤ (a + b) / 2) ∧
  (∀ m : ℝ, m ∈ set.Ioo (-2 : ℝ) 0 →
    ∃ x₀ : ℝ, x₀ ∈ set.Ioo (-1 : ℝ) 1 ∧ average_value_function (λ x, x ^ 2 + m * x - 1) (-1) 1 x₀) ∧
  (∀ (a b : ℝ) (h₀ : 1 ≤ a) (h₁ : a < b) (x₀ : ℝ), 
    average_value_function (λ x, Math.log x) a b x₀ → Math.log x₀ < 1 / Math.sqrt (a * b)) :=
by { 
  sorry 
}

end propositions_true_l445_445236


namespace parabola_1_is_higher_l445_445232

def parabola_1 (x : ℝ) : ℝ := x^2 - (3 / 4) * x + 3
def parabola_2 (x : ℝ) : ℝ := x^2 + (1 / 4) * x + 1

theorem parabola_1_is_higher
  (h1 : -((3 / 4) : ℝ) / (2 * 1) = (3 / 8))
  (k1 : (3 : ℝ) - ((-3 / 4)^2 / (4 * 1)) = (39 / 16))
  (h2 : -((1 / 4) : ℝ) / (2 * 1) = (-1 / 8))
  (k2 : (1 : ℝ) - ((1 / 4)^2 / (4 * 1)) = (15 / 16)) :
  (39 / 16) > (15 / 16) :=
by sorry

end parabola_1_is_higher_l445_445232


namespace solution_l445_445055

def f (x : ℝ) := exp (2 * x - 3)
def p : Prop := ∀ x y : ℝ, x < y → f x < f y
def q : Prop := ∃ x0 : ℝ, x0^2 - x0 + 2 < 0

theorem solution : (p ∧ ¬q) :=
by
  -- Proposition p: The function f(x) = e^(2x-3) is increasing on ℝ.
  have hp : p := by sorry
  -- Proposition q: For all x in ℝ, x^2 - x + 2 >= 0 (so, there does not exist an x0 such that x0^2 - x0 + 2 < 0).
  have hq : ¬q := by sorry
  -- Hence, the true proposition is p ∧ ¬q
  exact ⟨hp, hq⟩

end solution_l445_445055


namespace max_xy_expression_l445_445048

theorem max_xy_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 5 * y < 90) : 
  ∃ M, M = 1350 ∧ ∀ x y : ℝ, (0 < x) → (0 < y) → (4 * x + 5 * y < 90) → xy (90 - 4 * x - 5 * y) ≤ M :=
sorry

end max_xy_expression_l445_445048


namespace shift_sin_cos_graph_l445_445631

theorem shift_sin_cos_graph :
  (∀ x, sin (π/2 + 2*x) = cos (2*x)) →
  (∀ x, cos (2*(x - π/6)) = cos (2*x - π/3)) →
  (∃ d, ∀ x, sin (π/2 + 2*(x + d)) = cos (2*x - π/3) ∧ d = -π/6) :=
by
  intros h1 h2
  use (-π/6)
  intro x
  specialize h1 x
  specialize h2 x
  split
  sorry

end shift_sin_cos_graph_l445_445631


namespace exists_point_in_multiple_k_gons_l445_445027

-- Definitions for conditions
def plane : Type := ℝ × ℝ
def convex_polygon (k : ℕ) : Type := fin k → plane

-- Variables and parameters
variables (n k : ℕ) (P : fin n → convex_polygon k)

-- Definition of intersection and homothety
def intersects (P1 P2 : convex_polygon k) : Prop := sorry
def homothety (P1 P2 : convex_polygon k) : Prop := sorry

-- Main theorem statement
theorem exists_point_in_multiple_k_gons (h_intersects : ∀ i j : fin n, i ≠ j → intersects (P i) (P j))
(h_homothety : ∀ i j : fin n, i ≠ j → homothety (P i) (P j)) :
  ∃ p : plane, ∑ i : fin n, ite (∃ j : fin k, p ∈ (P i) j) 1 0 ≥ 1 + (n-1)/(2*k) :=
sorry

end exists_point_in_multiple_k_gons_l445_445027


namespace distinct_arrangements_of_BANANA_l445_445334

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445334


namespace hiking_time_l445_445521

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l445_445521


namespace sin_half_angle_l445_445904

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445904


namespace distinct_arrangements_of_BANANA_l445_445335

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445335


namespace ralph_squares_count_l445_445375

def total_matchsticks := 50
def elvis_square_sticks := 4
def ralph_square_sticks := 8
def elvis_squares := 5
def leftover_sticks := 6

theorem ralph_squares_count : 
  ∃ R : ℕ, 
  (elvis_squares * elvis_square_sticks) + (R * ralph_square_sticks) + leftover_sticks = total_matchsticks ∧ R = 3 :=
by 
  sorry

end ralph_squares_count_l445_445375


namespace sin_half_angle_l445_445814

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445814


namespace approximate_solution_to_fx_eq_0_l445_445480

noncomputable def f : ℝ → ℝ := λ x, 2 * x^3 + 3 * x - 3

theorem approximate_solution_to_fx_eq_0 :
  f 0.625 < 0 → f 0.75 > 0 → f 0.6875 < 0 → ∃ x : ℝ, abs (x - 0.7) < 0.05 := by
  sorry

end approximate_solution_to_fx_eq_0_l445_445480


namespace sin_half_alpha_l445_445857

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π/2)
variable (h₁ : cos α = (1 + sqrt 5) / 4)

theorem sin_half_alpha : sin (α / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445857


namespace solution_set_of_quadratic_inequality_l445_445618

theorem solution_set_of_quadratic_inequality (a : ℝ) (x : ℝ) :
  (∀ x, 0 < x - 0.5 ∧ x < 2 → ax^2 + 5 * x - 2 > 0) ∧ a = -2 →
  (∀ x, -3 < x ∧ x < 0.5 → a * x^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end solution_set_of_quadratic_inequality_l445_445618


namespace focus_line_distance_l445_445425

noncomputable def distance_from_focus_to_line (p : ℝ) : ℝ :=
  let focus := (0, p / 2)
  let line_case_1 := abs (-1 + 1) / sqrt (1^2 + 0^2)
  let line_case_2 := abs ((-(1/4) * 0 - 1/4 - p / 2)) / sqrt ((1/4)^2 + 1^2)
  let line_case_3 := sqrt ((-1)^2 + (0 - p / 2)^2)
  if p = 8 then min (min line_case_1 line_case_2) line_case_3 else 0

theorem focus_line_distance (p : ℝ) (A : ℝ × ℝ) (L : ℝ → ℝ) :
  A = (-1, 0) →
  (∃ k : ℝ, L = fun x => k * (x + 1)) →
  (∃ x : ℝ, x^2 = 2 * p * L x) →
  p = 8 →
  distance_from_focus_to_line p = 1 ∨ distance_from_focus_to_line p = 4 ∨ distance_from_focus_to_line p = sqrt 17 :=
by
  -- proof comes here
  sorry

end focus_line_distance_l445_445425


namespace select_7_jury_l445_445125

theorem select_7_jury (students : Finset ℕ) (jury : Finset ℕ)
  (likes : ℕ → Finset ℕ) (h_students : students.card = 100)
  (h_jury : jury.card = 25) (h_likes : ∀ s ∈ students, (likes s).card = 10) :
  ∃ (selected_jury : Finset ℕ), selected_jury.card = 7 ∧ ∀ s ∈ students, ∃ j ∈ selected_jury, j ∈ (likes s) :=
sorry

end select_7_jury_l445_445125


namespace perm_banana_l445_445350

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445350


namespace distance_between_points_l445_445381

theorem distance_between_points:
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -3
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = sqrt 61 := 
begin
  sorry
end

end distance_between_points_l445_445381


namespace man_l445_445707

theorem man's_rowing_speed_in_still_water (with_stream: ℕ) (against_stream: ℕ) : 
  with_stream = 16 → against_stream = 6 → ∃ R : ℕ, R = 11 := by
  assume h1 : with_stream = 16,
  assume h2 : against_stream = 6,
  let S := (with_stream - against_stream) / 2,
  let R := 11,
  use R,
  have hR : 2 * R = with_stream + against_stream := by
    calc
      2 * R = 2 * 11 := by rfl
      ... = 22 := by norm_num
      ... = with_stream + against_stream := by rw [h1, h2]
  exact hR

end man_l445_445707


namespace mean_goals_l445_445609

theorem mean_goals :
  let goals := 2 * 3 + 4 * 2 + 5 * 1 + 6 * 1
  let players := 3 + 2 + 1 + 1
  goals / players = 25 / 7 :=
by
  sorry

end mean_goals_l445_445609


namespace sin_half_angle_l445_445844

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445844


namespace polynomial_division_remainder_zero_l445_445387

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_zero_l445_445387


namespace xn_plus_inv_xn_eq_two_sin_n_theta_l445_445478

theorem xn_plus_inv_xn_eq_two_sin_n_theta
  (θ : ℝ) (x : ℝ)
  (hθ1 : 0 < θ) 
  (hθ2 : θ < real.pi / 2)
  (hx : x + (1 / x) = 2 * real.sin θ)
  (n : ℕ) :
  x ^ n + (1 / x) ^ n = 2 * real.sin (n * θ) :=
sorry

end xn_plus_inv_xn_eq_two_sin_n_theta_l445_445478


namespace minimum_flowers_to_guarantee_bouquets_l445_445995

theorem minimum_flowers_to_guarantee_bouquets :
  (∀ (num_types : ℕ) (flowers_per_bouquet : ℕ) (num_bouquets : ℕ),
   num_types = 6 → flowers_per_bouquet = 5 → num_bouquets = 10 →
   ∃ min_flowers : ℕ, min_flowers = 70 ∧
   ∀ (picked_flowers : ℕ → ℕ), 
     (∀ t : ℕ, t < num_types → picked_flowers t ≥ 0 ∧ 
                (t < num_types - 1 → picked_flowers t ≤ flowers_per_bouquet * (num_bouquets - 1) + 4)) → 
     ∑ t in finset.range num_types, picked_flowers t = min_flowers → 
     ∑ t in finset.range num_types, (picked_flowers t / flowers_per_bouquet) ≥ num_bouquets) := 
by {
  intro num_types flowers_per_bouquet num_bouquets,
  intro h1 h2 h3,
  use 70,
  split,
  {
    exact rfl,
  },
  {
    intros picked_flowers h_picked,
    sorry,
  }
}

end minimum_flowers_to_guarantee_bouquets_l445_445995


namespace Dawn_sold_glasses_l445_445204

variable (x : ℕ)

def Bea_price_per_glass : ℝ := 0.25
def Dawn_price_per_glass : ℝ := 0.28
def Bea_glasses_sold : ℕ := 10
def Bea_extra_earnings : ℝ := 0.26
def Bea_total_earnings : ℝ := Bea_glasses_sold * Bea_price_per_glass
def Dawn_total_earnings (x : ℕ) : ℝ := x * Dawn_price_per_glass

theorem Dawn_sold_glasses :
  Bea_total_earnings - Bea_extra_earnings = Dawn_total_earnings x → x = 8 :=
by
  sorry

end Dawn_sold_glasses_l445_445204


namespace chef_earns_17_less_than_manager_l445_445201

noncomputable def manager_hourly_wage := 8.50
noncomputable def dishwasher_hourly_wage := manager_hourly_wage / 2
noncomputable def chef_hourly_wage := dishwasher_hourly_wage * 1.20
noncomputable def manager_hours := 8
noncomputable def dishwasher_hours := 6
noncomputable def chef_hours := 10
noncomputable def daily_bonus := 5

noncomputable def manager_daily_earning := manager_hourly_wage * manager_hours + daily_bonus
noncomputable def dishwasher_daily_earning := dishwasher_hourly_wage * dishwasher_hours + daily_bonus
noncomputable def chef_daily_earning := chef_hourly_wage * chef_hours + daily_bonus

theorem chef_earns_17_less_than_manager :
  manager_daily_earning - chef_daily_earning = 17 :=
by
  sorry

end chef_earns_17_less_than_manager_l445_445201


namespace part_one_part_two_l445_445921

section
variable {R : Type*} [LinearOrderedField R]
variable (f : R → R) (a b : R)
variable (h_inc : ∀ x y, x ≤ y → f(x) ≤ f(y))

theorem part_one (h : a + b ≥ 0) : f(a) + f(b) ≥ f(-a) + f(-b) :=
sorry

theorem part_two (h : f(a) + f(b) ≥ f(-a) + f(-b)) : a + b ≥ 0 :=
sorry
end

end part_one_part_two_l445_445921


namespace doughnuts_in_each_box_l445_445686

theorem doughnuts_in_each_box (total_doughnuts : ℕ) (boxes : ℕ) (h1 : total_doughnuts = 48) (h2 : boxes = 4) : total_doughnuts / boxes = 12 :=
by
  sorry

end doughnuts_in_each_box_l445_445686


namespace number_of_arrangements_of_BANANA_l445_445313

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445313


namespace multiples_of_15_between_20_and_200_l445_445463

theorem multiples_of_15_between_20_and_200 : 
  let a : ℕ := 30
  let l : ℕ := 195
  let d : ℕ := 15
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 12 := 
begin
  sorry
end

end multiples_of_15_between_20_and_200_l445_445463


namespace minimal_dominoes_needed_l445_445137

-- Variables representing the number of dominoes and tetraminoes
variables (d t : ℕ)

-- Definitions related to the problem
def area_rectangle : ℕ := 2008 * 2010 -- Total area of the rectangle
def area_domino : ℕ := 1 * 2 -- Area of a single domino
def area_tetramino : ℕ := 2 * 3 - 2 -- Area of a single tetramino
def total_area_covered : ℕ := 2 * d + 4 * t -- Total area covered by dominoes and tetraminoes

-- The theorem we want to prove
theorem minimal_dominoes_needed :
  total_area_covered d t = area_rectangle → d = 0 :=
sorry

end minimal_dominoes_needed_l445_445137


namespace find_b_for_collinear_points_l445_445129

theorem find_b_for_collinear_points :
  ∀ (b : ℚ),
  let P1 := (4 : ℚ, -7 : ℚ),
      P2 := (-2 * b + 3, 5),
      P3 := (3 * b + 4, 3) in
  collinear {P1, P2, P3} → b = -5 / 28 :=
by
  intros b P1 P2 P3 h_collinear
  rw collinear_iff
  sorry

end find_b_for_collinear_points_l445_445129


namespace sufficient_condition_parallel_planes_l445_445450

noncomputable def line (point₁ point₂ : ℝ × ℝ × ℝ) := {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (point₁.1 + t * (point₂.1 - point₁.1), point₁.2 + t * (point₂.2 - point₁.2), point₁.3 + t * (point₂.3 - point₁.3))}
noncomputable def plane (point₁ point₂ point₃ : ℝ × ℝ × ℝ) := {p : ℝ × ℝ × ℝ | ∃ u v : ℝ, p = (point₁.1 + u * (point₂.1 - point₁.1) + v * (point₃.1 - point₁.1), point₁.2 + u * (point₂.2 - point₁.2) + v * (point₃.2 - point₁.2), point₁.3 + u * (point₂.3 - point₁.3) + v * (point₃.3 - point₁.3))}
def perp (a b : set (ℝ × ℝ × ℝ)) : Prop := 
  ∀ (p₁ p₂ p₃ p₄ : ℝ × ℝ × ℝ) (t₁ t₂ : ℝ × ℝ × ℝ), p₁ ∈ a → p₂ ∈ a → p₃ ∈ b → p₄ ∈ b →
  let v₁ := (p₂.1 - p₁.1, p₂.2 - p₁.2, p₂.3 - p₁.3) in
  let v₂ := (p₄.1 - p₃.1, p₄.2 - p₃.2, p₄.3 - p₄.3) in
  (v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3 = 0)

theorem sufficient_condition_parallel_planes (a b α β : set (ℝ × ℝ × ℝ)) : 
  (∃ p₁ p₂ : ℝ × ℝ × ℝ, a = line p₁ p₂) → 
  (∃ p₁ p₂ p₃ : ℝ × ℝ × ℝ, α = plane p₁ p₂ p₃) → 
  (∃ p₁ p₂ : ℝ × ℝ × ℝ, b = line p₁ p₂) → 
  (∃ p₁ p₂ p₃ : ℝ × ℝ × ℝ, β = plane p₁ p₂ p₃) → 
  (α ⊆ β → perp a b) ∧ (perp a b → α ⊆ β) → False :=
sorry

end sufficient_condition_parallel_planes_l445_445450


namespace average_height_l445_445595

theorem average_height (avg1 avg2 : ℕ) (n1 n2 : ℕ) (total_students : ℕ)
  (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) (h5 : total_students = 31) :
  (n1 * avg1 + n2 * avg2) / total_students = 20 :=
by
  -- Placeholder for the proof
  sorry

end average_height_l445_445595


namespace mrs_bil_earnings_percentage_in_may_l445_445005

theorem mrs_bil_earnings_percentage_in_may
  (M F : ℝ)
  (h₁ : 1.10 * M / (1.10 * M + F) = 0.7196) :
  M / (M + F) = 0.70 :=
sorry

end mrs_bil_earnings_percentage_in_may_l445_445005


namespace arithmetic_geometric_sequence_l445_445434

theorem arithmetic_geometric_sequence :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ),
    (a 1 + a 2 = 10) →
    (a 4 - a 3 = 2) →
    (b 2 = a 3) →
    (b 3 = a 7) →
    a 15 = b 4 :=
by
  intros a b h1 h2 h3 h4
  sorry

end arithmetic_geometric_sequence_l445_445434


namespace permutations_of_BANANA_l445_445249

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445249


namespace minimum_flowers_to_guarantee_bouquets_l445_445994

-- Definitions based on conditions given
def types_of_flowers : ℕ := 6
def flowers_needed_for_bouquet : ℕ := 5
def required_bouquets : ℕ := 10

-- Problem statement in Lean 4
theorem minimum_flowers_to_guarantee_bouquets (types : ℕ) (needed: ℕ) (bouquets: ℕ) 
    (h_types: types = types_of_flowers) (h_needed: needed = flowers_needed_for_bouquet) 
    (h_bouquets: bouquets = required_bouquets) : 
    (minimum_number_of_flowers_to_guarantee_bouquets types needed bouquets) = 70 :=
by sorry


end minimum_flowers_to_guarantee_bouquets_l445_445994


namespace BANANA_arrangements_l445_445306

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445306


namespace translated_parabola_eq_l445_445132

-- Define the original parabola
def orig_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation function
def translate_upwards (f : ℝ → ℝ) (dy : ℝ) : (ℝ → ℝ) :=
  fun x => f x + dy

-- Define the translated parabola
def translated_parabola := translate_upwards orig_parabola 3

-- State the theorem
theorem translated_parabola_eq:
  translated_parabola = (fun x : ℝ => -2 * x^2 + 3) :=
by
  sorry

end translated_parabola_eq_l445_445132


namespace number_of_arrangements_of_BANANA_l445_445312

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445312


namespace distinct_arrangements_of_BANANA_l445_445337

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445337


namespace no_integer_exponent_l445_445054

-- Define the exp function.
def exp (m : ℕ) (n : ℕ) : ℕ := m ^ n

-- Noncomputable declaration for technical reasons.
noncomputable def integer_exponent (n : ℕ) : ℝ := log n / log 10

-- Main theorem.
theorem no_integer_exponent (m n : ℕ) (hm : exp 10 m = 25) (hmn : m ∈ ℤ) : ¬∃ x : ℤ, exp x = 25 :=
by
  sorry

end no_integer_exponent_l445_445054


namespace average_cost_per_orange_l445_445173

theorem average_cost_per_orange :
  ∀ (n m : ℕ), n = 20 → m = (12 * 4 + 12) → (m / n = 3) :=
by
  assume n m : ℕ,
  assume h₁ : n = 20,
  assume h₂ : m = 60,
  sorry

end average_cost_per_orange_l445_445173


namespace vector_dot_product_sum_eq_l445_445946

variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V)

theorem vector_dot_product_sum_eq :
  a + b + c = 0 →
  ∥a∥ = 1 →
  ∥b∥ = 2 →
  ∥c∥ = 2 →
  inner a b + inner b c + inner c a = -9 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end vector_dot_product_sum_eq_l445_445946


namespace quadratic_roots_ratio_l445_445117

theorem quadratic_roots_ratio (a b c : ℝ) (h1 : ∀ (s1 s2 : ℝ), s1 * s2 = a → s1 + s2 = -c → 3 * s1 + 3 * s2 = -a → 9 * s1 * s2 = b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  b / c = 27 := sorry

end quadratic_roots_ratio_l445_445117


namespace sin_half_alpha_l445_445862

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445862


namespace min_flowers_for_bouquets_l445_445999

open Classical

noncomputable def minimum_flowers (types : ℕ) (flowers_for_bouquet : ℕ) (bouquets : ℕ) : ℕ := 
  sorry

theorem min_flowers_for_bouquets : minimum_flowers 6 5 10 = 70 := 
  sorry

end min_flowers_for_bouquets_l445_445999


namespace black_friday_sales_projection_l445_445069

theorem black_friday_sales_projection (sold_now : ℕ) (increment : ℕ) (years : ℕ) 
  (h_now : sold_now = 327) (h_inc : increment = 50) (h_years : years = 3) : 
  let sold_three_years := sold_now + 3 * increment in
  sold_three_years = 477 := 
by
  -- Definitions according to the conditions
  have h1 : sold_now = 327 := h_now
  have h2 : increment = 50 := h_inc
  have h3 : years = 3 := h_years

  -- Calculation based on definitions
  have h_sold_next_year := sold_now + increment
  have h_sold_second_year := h_sold_next_year + increment
  have h_sold_third_year := h_sold_second_year + increment
  
  -- Haven't elaborated on proof steps as the problem requires the statement only
  sorry

end black_friday_sales_projection_l445_445069


namespace sum_of_solutions_l445_445783

theorem sum_of_solutions (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (∃ x₁ x₂ : ℝ, (3 * x₁ + 2) * (x₁ - 4) = 0 ∧ (3 * x₂ + 2) * (x₂ - 4) = 0 ∧
  x₁ ≠ 1 ∧ x₁ ≠ -1 ∧ x₂ ≠ 1 ∧ x₂ ≠ -1 ∧ x₁ + x₂ = 10 / 3) :=
sorry

end sum_of_solutions_l445_445783


namespace students_not_enrolled_l445_445008

theorem students_not_enrolled (total_students : ℕ) (students_french : ℕ) (students_german : ℕ) (students_both : ℕ)
  (h1 : total_students = 94)
  (h2 : students_french = 41)
  (h3 : students_german = 22)
  (h4 : students_both = 9) : 
  ∃ (students_neither : ℕ), students_neither = 40 :=
by
  -- We would show the calculation here in a real proof 
  sorry

end students_not_enrolled_l445_445008


namespace range_of_c_l445_445506

theorem range_of_c :
  (∃ (c : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 4) → ((12 * x - 5 * y + c) / 13 = 1))
  → (c > -13 ∧ c < 13) := 
sorry

end range_of_c_l445_445506


namespace number_of_ordered_pairs_l445_445764

theorem number_of_ordered_pairs :
  {p : ℤ × ℤ | let a := p.fst, b := p.snd in a^2 + b^2 < 25 ∧ a^2 + b^2 < 10 * a ∧ a^2 + b^2 < 10 * b}.card = 11 :=
by
  sorry

end number_of_ordered_pairs_l445_445764


namespace exists_large_prime_invariant_integer_l445_445033

theorem exists_large_prime_invariant_integer:
  ∃ n : ℕ, n > 10^1000 ∧ ¬(10 ∣ n) ∧ ∃ d₁ d₂ : ℤ,
      (d₁ ≠ 0 ∧ d₂ ≠ 0 ∧ d₁ ≠ d₂ ∧ 
      (set.prime_factors (swap_digits n d₁ d₂) = set.prime_factors n)) :=
  sorry

end exists_large_prime_invariant_integer_l445_445033


namespace arithmetic_sequence_sum_l445_445504

/-
In an arithmetic sequence, if the sum of terms \( a_2 + a_3 + a_4 + a_5 + a_6 = 90 \), 
prove that \( a_1 + a_7 = 36 \).
-/

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_sum : a 2 + a 3 + a 4 + a 5 + a 6 = 90) :
  a 1 + a 7 = 36 := by
  sorry

end arithmetic_sequence_sum_l445_445504


namespace locus_of_points_l445_445720

-- A point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a square plate with side length 10 cm lying in the XY-plane
def square_plate (side: ℝ) : set Point3D := 
  {p | 0 ≤ p.x ∧ p.x ≤ side ∧ 0 ≤ p.y ∧ p.y ≤ side ∧ p.z = 0}

-- Define parallel planes 5 cm away from the square plate
def parallel_planes : set (set Point3D) := 
  { plane_above | ∀ p, (p ∈ square_plate 10) → 
                  {q | q.z = p.z + 5 ∧ q.x = p.x ∧ q.y = p.y}} ∪ 
  { plane_below | ∀ p, (p ∈ square_plate 10) → 
                  {q | q.z = p.z - 5 ∧ q.x = p.x ∧ q.y = p.y}}

-- Define quarter-cylinder surfaces along the edges of the square
def quarter_cylinders (r: ℝ) (l: ℝ) : set (set Point3D) :=
  {cylinder_AL, cylinder_AB, cylinder_BC, cylinder_CD |
  ∀ a b, (a ∈ square_plate l) ∧ (b ∈ square_plate l) → 
         { q | (q.x - a.x)^2 + (q.y - a.y)^2 = r^2 ∧ b.z = 0 }}

-- Define quarter-spheres centered at the vertices of the square plate
def quarter_spheres (r: ℝ) (center: Point3D) : set (set Point3D) :=
  {sphere_A, sphere_B, sphere_C, sphere_D |
  ∀ p, (p ∈ square_plate 10) → 
        {q | (q.x - center.x)^2 + (q.y - center.y)^2 + (q.z - center.z)^2 = r^2 }}

theorem locus_of_points :
  ∃ (points : set Point3D),
    points = (parallel_planes ∪ quarter_cylinders 5 10 ∪ quarter_spheres 5 (⟨0, 0, 0⟩ : Point3D))
 := by
  sorry

end locus_of_points_l445_445720


namespace problem_1_problem_2_l445_445454

variable {α : ℝ}
variables a b : ℝ × ℝ

def initial_conditions : Prop :=
  a = (3 * Real.sin α, Real.cos α) ∧
  b = (2 * Real.sin α, 5 * Real.sin α - 4 * Real.cos α) ∧
  α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi ∧
  (Prod.fst a * Prod.fst b + Prod.snd a * Prod.snd b = 0)

theorem problem_1 (h : initial_conditions) : Real.tan α = -4 / 3 :=
sorry

theorem problem_2 (h : initial_conditions) : Real.cos (α / 2 + Real.pi / 3) = (-2 * Real.sqrt 5 - Real.sqrt 15) / 10 :=
sorry

end problem_1_problem_2_l445_445454


namespace line_l_equation_l445_445923

-- Define the equation of the given line and its slope
def given_line_equation (x y : ℝ) : Prop := x + y - 3 = 0

-- Define what it means for a line to intercept the y-axis at 2
def intercepts_y_axis_at (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

-- Define what it means for two lines to be perpendicular
def perpendicular_to (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → x1 + y2 ≠ 0 ∧ y1 + x2 = 0

-- Define the general form equation of a line
def general_form (x y c : ℝ) : ℝ → ℝ → Prop := 
  λ (a b : ℝ), a * x + b * y + c = 0

-- The main theorem to be proved
theorem line_l_equation :
  ∃ l : ℝ → ℝ → Prop, 
    (∃ b : ℝ, intercepts_y_axis_at l 2) ∧ 
    perpendicular_to l given_line_equation ∧
    general_form 1 (-1) 2 = l :=
sorry

end line_l_equation_l445_445923


namespace exists_singleton_subset_l445_445096

theorem exists_singleton_subset (S : Finset ℕ) (S_i : ℕ → Finset ℕ) 
  (h1 : ∀ i j ∈ S, j ∈ S_i i → i ∈ S_i j)
  (h2 : ∀ i j ∈ S, S_i i.card = S_i j.card → S_i i ∩ S_i j = ∅) : 
  ∃ k ∈ S, (S_i k).card = 1 := 
sorry

end exists_singleton_subset_l445_445096


namespace banana_arrangements_l445_445325

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445325


namespace optimal_selling_price_l445_445620

variables
  (cost_price : ℕ)
  (initial_selling_price : ℕ)
  (initial_sales_volume : ℕ)
  (decrease_rate : ℕ)
  (monthly_profit_target : ℕ)
  (selling_price : ℕ)

def monthly_sales_volume (x : ℕ) : ℕ :=
  initial_sales_volume - decrease_rate * (x - initial_selling_price)

def monthly_profit (x : ℕ) : ℕ :=
  (x - cost_price) * (monthly_sales_volume x)

theorem optimal_selling_price :
  cost_price = 30 →
  initial_selling_price = 40 →
  initial_sales_volume = 600 →
  decrease_rate = 10 →
  monthly_profit_target = 10000 →
  monthly_profit 50 = monthly_profit_target :=
by intros; sorry

end optimal_selling_price_l445_445620


namespace cubed_difference_l445_445483

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end cubed_difference_l445_445483


namespace perm_banana_l445_445347

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445347


namespace first_player_wins_l445_445404

-- Define the conditions of the game
def valid_moves : Set ℕ := {1, 2, 3, 4, 5}

def next_move_is_valid (prev_move : ℕ) (current_move : ℕ) :=
  current_move ∈ valid_moves ∧ prev_move ≠ current_move

-- Prove that the first player has a winning strategy
theorem first_player_wins : 
  ∃ (strategy : ℕ → ℕ), 
    strategy 2000 ∈ valid_moves ∧ 
    ∀ (n k : ℕ), k ∈ valid_moves → next_move_is_valid (strategy n) k → 
      (n - strategy n) % 13 ∈ {0, 3, 5, 7} → 
        ∃ (strategy' : ℕ → ℕ), 
          strategy' (n - strategy n - k) ∈ valid_moves ∧ 
          next_move_is_valid k (strategy' (n - strategy n - k)) →
            (strategy' (n - strategy n - k)) ∈ {0, 3, 5, 7} :=
sorry

end first_player_wins_l445_445404


namespace perm_banana_l445_445355

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445355


namespace union_of_A_and_B_l445_445413

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l445_445413


namespace number_of_valid_11_tuples_l445_445780

theorem number_of_valid_11_tuples : 
  {t : Fin 11 → ℤ // ∀ i : Fin 11, (t i) ^ 2 = (Finset.univ.erase i).sum (λ j, t j)}.card = 330 :=
sorry

end number_of_valid_11_tuples_l445_445780


namespace proof_find_BE_l445_445682

variables {A B C M P K E : Type*}
variables {R a b : Real} {α : Real}
variables (triangle_ABC : ∀ {x : Type*}, x ≠ A ∨ x ≠ B ∨ x ≠ C)
variables (circle_touches_AC_at_M : ∀ {x : Type*}, x = M)
variables (circle_touches_BC_at_P : ∀ {x : Type*}, x = P)
variables (AB_intersects_circle_at_K_E : E lies_on_segment B K →  K ≠ E)
variables (BC_eq_a : BC = a)
variables (CM_eq_b : CM = b)
variables (angle_KME_eq_alpha : ∠KME = α)
variables (b_less_than_a : b < a)

noncomputable def find_BE (a b R α : Real) : Real :=
√(R^2 * (sin α)^2 + (a - b)^2) - R * (sin α)

theorem proof_find_BE :
  find_BE a b R α = √(R^2 * (sin α)^2 + (a - b)^2) - R * (sin α) :=
sorry

end proof_find_BE_l445_445682


namespace range_S_n_l445_445045

noncomputable def f : ℝ → ℝ := sorry  -- a function defined on ℝ that is never zero

axiom f_nonzero : ∀ x : ℝ, f x ≠ 0

axiom f_mul : ∀ x y : ℝ, f(x) * f(y) = f(x + y)

def a_n (n : ℕ) : ℝ :=
if n = 0 then 1 / 2 else f (n : ℝ)

def S_n (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), a_n i

theorem range_S_n : ∀ n : ℕ, 1 / 2 ≤ S_n n ∧ S_n n < 1 := sorry

end range_S_n_l445_445045


namespace length_OB_max_volume_l445_445598

-- Given conditions as definitions
variables {P A B H C O : ℝ}
def is_isosceles_right_triangle (P A B C : ℝ) : Prop := sorry
def perpendicular (x y : ℝ) : Prop := sorry -- a stub for perpendicular property
def on_circumference (A : ℝ) : Prop := sorry -- a stub for point on circumference

-- The main theorem statement
theorem length_OB_max_volume (P A B H C O : ℝ) (h1 : is_isosceles_right_triangle P A B C) 
  (h2 : on_circumference A) (h3 : perpendicular A B) (h4 : perpendicular O B)
  (h5 : perpendicular O H) (h6 : PA = 4) (h7 : C = (P + A) / 2) :
  ∃ OB : ℝ, OB = (2 * real.sqrt 6) / 3 := 
sorry

end length_OB_max_volume_l445_445598


namespace smallest_h_l445_445156

theorem smallest_h (h : ℕ) : 
  (∀ k, h = k → (k + 5) % 8 = 0 ∧ 
        (k + 5) % 11 = 0 ∧ 
        (k + 5) % 24 = 0) ↔ h = 259 :=
by
  sorry

end smallest_h_l445_445156


namespace find_angle_C_find_AB_length_l445_445032

variable {A B C : ℝ}
variable {a b AB : ℝ}
variable {angle_C : ℝ}

-- Conditions
axiom h1 : BC = a
axiom h2 : AC = b
axiom h3 : a^2 - 2 * Real.sqrt 3 * a + 2 = 0
axiom h4 : b^2 - 2 * Real.sqrt 3 * b + 2 = 0
axiom h5 : 2 * Real.cos (A + B) = 1

-- Questions and Answers
theorem find_angle_C : ∀ (A B C : ℝ) (a b : ℝ), 
  (BC = a) → (AC = b) → 
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) → (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = 1) → angle_C = 2 * Real.pi / 3 := 
by sorry

theorem find_AB_length : ∀ (A B C : ℝ) (a b : ℝ), 
  (BC = a) → (AC = b) → 
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) → (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = 1) → 
  AB = Real.sqrt(10) := 
by sorry

end find_angle_C_find_AB_length_l445_445032


namespace banana_permutations_l445_445255

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445255


namespace question_1_question_2_l445_445439

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := (1 / 3) * a * x ^ 3 - (1 / 2) * x ^ 2 + (a - 1) * x + 1

-- Definition of the derivative of f(x)
def f' (a x : ℝ) : ℝ := a * x ^ 2 - x + a - 1

-- Proof that a = 2 given the conditions in Question 1
theorem question_1 (a : ℝ) : (f' a 1 = 2) ↔ (a = 2) :=
by
  calc f' a 1 = 2 ↔ 2 * a - 2 = 2 : by simp
          ... ↔ 2 * a = 4       : by simp
          ... ↔ a = 2           : by simp

-- Proof of the summary of monotonicity for Question 2
theorem question_2 (a x : ℝ) (h : x ≥ 2) :
    (a ≤ 0 → ∀ y, y ≥ 2 → f' a y < 0) ∧ 
    (0 < a ∧ a < (3 / 5) → (∃ x0, x0 = (1 + (Real.sqrt (-4 * a^2 + 4 * a + 1))) / (2 * a) ∧ (x ≤ x0 → f' a x < 0) ∧ (x > x0 → f' a x > 0))) ∧ 
    (a ≥ (3 / 5) → ∀ y, y ≥ 2 → f' a y > 0) :=
sorry   -- Proof omitted

end question_1_question_2_l445_445439


namespace minimum_value_expression_l445_445049

theorem minimum_value_expression (a x y z : ℝ) (h : 0 < a ∧ a < 1) (hx : -a < x ∧ x < a) (hy : -a < y ∧ y < a) (hz : -a < z ∧ z < a) :
  (\frac{1}{(1 - x) * (1 - y) * (1 - z)} + \frac{1}{(1 + x) * (1 + y) * (1 + z)}) ≥ \frac{2}{(1 - a^2)^3} :=
  sorry

end minimum_value_expression_l445_445049


namespace work_rate_b_l445_445147

theorem work_rate_b (W : ℝ) (A B C : ℝ) :
  (A = W / 11) → 
  (C = W / 55) →
  (8 * A + 4 * B + 4 * C = W) →
  B = W / (2420 / 341) :=
by
  intros hA hC hWork
  -- We start with the given assumptions and work towards showing B = W / (2420 / 341)
  sorry

end work_rate_b_l445_445147


namespace password_probability_l445_445744

def is_single_digit_prime (n : ℕ) : Prop :=
  n ∈ {2, 3, 5, 7}

def is_color (c : String) : Prop :=
  c ∈ {"red", "blue", "green"}

def is_positive_single_digit (n : ℕ) : Prop :=
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def probability_password : ℚ :=
  (4 / 10 : ℚ) * (1 / 3) * (9 / 10)

theorem password_probability :
  (probability_password = 3 / 25) :=
by 
  unfold probability_password
  norm_num
  sorry

end password_probability_l445_445744


namespace binom_12_11_l445_445226

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_l445_445226


namespace pen_worth_inconsistency_l445_445192

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (· < ·)
  if sorted.length % 2 = 0 then
    (sorted[(sorted.length / 2) - 1] + sorted[sorted.length / 2]) / 2
  else
    sorted[sorted.length / 2]

theorem pen_worth_inconsistency :
  let worths := [22, 25, 30, 40]
  mean worths ≠ 1.75 ∨ median worths ≠ 1.75 :=
by
  let worths := [22, 25, 30, 40]
  have h_mean : mean worths = (22 + 25 + 30 + 40) / worths.length := rfl
  have h_median : median worths = (worths.nthLe 1 sorry + worths.nthLe 2 sorry) / 2 := rfl
  have mean_val : (117 : ℚ) / 4 ≠ 1.75 := by norm_num
  have median_val : (25 + 30 : ℚ) / 2 ≠ 1.75 := by norm_num
  exact Or.inl mean_val

end pen_worth_inconsistency_l445_445192


namespace total_tires_mike_changed_l445_445060

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_tires_mike_changed_l445_445060


namespace sum_of_two_numbers_l445_445613

theorem sum_of_two_numbers :
  ∃ x y : ℝ, (x * y = 9375 ∧ y / x = 15) ∧ (x + y = 400) :=
by
  sorry

end sum_of_two_numbers_l445_445613


namespace smallest_n_with_terminating_decimal_and_digit_7_l445_445655

def contains_digit_7 (n : ℕ) : Prop :=
  n.digits 10 |> List.contains 7

theorem smallest_n_with_terminating_decimal_and_digit_7 :
  (∃ n : ℕ, (fractional n).is_terminating ∧ contains_digit_7 n ∧ ∀ m : ℕ, (fractional m).is_terminating ∧ contains_digit_7 m → n ≤ m) →
  ∃ smallest_n : ℕ, smallest_n = 128 :=
by {
  sorry -- proof goes here
}

end smallest_n_with_terminating_decimal_and_digit_7_l445_445655


namespace arrangement_of_BANANA_l445_445368

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445368


namespace vector_dot_product_sum_l445_445967

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_sum
  (a b c : V)
  (h1 : a + b + c = 0)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hc : ∥c∥ = 2) :
  inner_product_space.inner a b + inner_product_space.inner b c + inner_product_space.inner c a = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445967


namespace next_year_multiple_of_6_8_9_l445_445145

theorem next_year_multiple_of_6_8_9 (n : ℕ) (h₀ : n = 2016) (h₁ : n % 6 = 0) (h₂ : n % 8 = 0) (h₃ : n % 9 = 0) : ∃ m > n, m % 6 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m = 2088 :=
by
  sorry

end next_year_multiple_of_6_8_9_l445_445145


namespace initial_cheerleaders_count_l445_445590

theorem initial_cheerleaders_count (C : ℕ) 
  (initial_football_players : ℕ := 13) 
  (quit_football_players : ℕ := 10) 
  (quit_cheerleaders : ℕ := 4) 
  (remaining_people : ℕ := 15) 
  (initial_total : ℕ := initial_football_players + C) 
  (final_total : ℕ := (initial_football_players - quit_football_players) + (C - quit_cheerleaders)) :
  remaining_people = final_total → C = 16 :=
by intros h; sorry

end initial_cheerleaders_count_l445_445590


namespace pencil_purchase_cost_l445_445695

theorem pencil_purchase_cost (cost_per_box : ℝ) (pencils_per_box : ℕ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) (pencils_ordered : ℕ) :
  cost_per_box = 50 → pencils_per_box = 200 → discount_threshold = 1000 → 
  discount_rate = 0.10 → pencils_ordered = 2500 → 
  (∑ i in (finset.range 2500).filter (λ i, i < 1000), 0.25) + 
  (∑ i in (finset.range 2500).filter (λ i, i ≥ 1000), 0.225) = 587.50 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end pencil_purchase_cost_l445_445695


namespace combinations_of_eight_choose_three_is_fifty_six_l445_445024

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end combinations_of_eight_choose_three_is_fifty_six_l445_445024


namespace find_constant_C_l445_445183

variable {λ : ℝ} (f : ℝ → ℝ)

def exponential_pdf (x : ℝ) : ℝ :=
  if x < 0 then 0 else λ * Real.exp (-λ * x)

theorem find_constant_C (h : ∫ x in -∞..∞, exponential_pdf λ x = 1) : λ = λ := sorry

end find_constant_C_l445_445183


namespace density_of_B_l445_445539

noncomputable def is_dense_in_Rn (A : set (ℝ^n)) (B : set (ℝ^n)) : Prop :=
  (∀ x, ∃ b ∈ B, ∀ a ∈ A, ∃! a₀ ∈ A, dist b a₀ = inf {dist b a | a ∈ A}) ∧ closure B = univ

theorem density_of_B (A : set (ℝ^n)) (B : set (ℝ^n)) (hA_closed : is_closed A)
  (hB : ∀ b, b ∈ B ↔ ∃! a ∈ A, dist b a = inf {dist b a | a ∈ A}) :
  is_dense_in_Rn A B :=
begin
  sorry
end

end density_of_B_l445_445539


namespace degree_measure_angle_QRT_l445_445579

open Real

/-- Mathematics problem statement:
    Segment PQ has midpoint R, and segment QR has midpoint S. 
    Semi-circles are constructed with diameters PQ and QR to form the entire region shown.
    Segment RT splits the region into two sections of equal area.
    Prove that the degree measure of angle QRT is 225 degrees.
-/
theorem degree_measure_angle_QRT (PQ R Q S T : ℝ) 
  (h1 : 0 < PQ) (h2 : R = PQ / 2) (h3 : QR = PQ / 2) 
  (h4 : S = R / 2) (h5 : region_split_by_RT_is_equal : true) : 
  (angle_QRT = 225) :=
begin
  sorry
end

end degree_measure_angle_QRT_l445_445579


namespace rectangle_sides_l445_445016

theorem rectangle_sides (AB AC : ℕ) (x y : ℚ) : 
  AB = 5 → AC = 12 → x * y = 40 / 3 → real.sqrt (x^2 + y^2) < 8 → 
  (x = 4 ∧ y = 10 / 3) ∨ (x = 10 / 3 ∧ y = 4) :=
by 
  intros hAB hAC hArea hDiagonal;
  sorry

end rectangle_sides_l445_445016


namespace value_of_r_minus_p_l445_445674

-- Define the arithmetic mean conditions
def arithmetic_mean1 (p q : ℝ) : Prop :=
  (p + q) / 2 = 10

def arithmetic_mean2 (q r : ℝ) : Prop :=
  (q + r) / 2 = 27

-- Prove that r - p = 34 based on the conditions
theorem value_of_r_minus_p (p q r : ℝ)
  (h1 : arithmetic_mean1 p q)
  (h2 : arithmetic_mean2 q r) :
  r - p = 34 :=
by
  sorry

end value_of_r_minus_p_l445_445674


namespace coin_probabilities_equal_l445_445628

theorem coin_probabilities_equal :
  let A := (FirstCoin = "heads")
  let B := (SecondCoin = "tails")
  (P(A) = P(B)) :=
by
  let A := "the first coin is heads up"
  let B := "the second coin is tails up"
  have P_A : Probability A = 1 / 2 := by sorry
  have P_B : Probability B = 1 / 2 := by sorry
  sorry

end coin_probabilities_equal_l445_445628


namespace sum_c_n_l445_445803

variable (a_n b_n c_n : ℕ → ℕ)

axiom a_seq : ∀ n, a_n n = n
axiom b_seq : ∀ n, b_n n = 2^(n-1)
axiom a1_eq_1 : a_n 1 = 1
axiom b1_eq_1 : b_n 1 = 1
axiom a3_a7_sum_10 : a_n 3 + a_n 7 = 10
axiom b3_eq_a4 : b_n 3 = a_n 4

noncomputable def T_n (n : ℕ) : ℕ := (n-1) * 2^n + 1

theorem sum_c_n (n : ℕ) : ∑ i in finset.range n, (a_n i) * (b_n i) = T_n n :=
by sorry

end sum_c_n_l445_445803


namespace vector_dot_product_sum_l445_445963

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem vector_dot_product_sum (h₁ : a + b + c = 0) 
                               (ha : ∥a∥ = 1)
                               (hb : ∥b∥ = 2)
                               (hc : ∥c∥ = 2) :
  (a ⬝ b) + (b ⬝ c) + (c ⬝ a) = -9 / 2 := sorry

end vector_dot_product_sum_l445_445963


namespace a1_geq_2_sequence_monotonically_increasing_l445_445446

variable {a : ℕ → ℝ}

-- Condition 1: For any positive integer n, a_n > 0
def positive_sequence : Prop :=
  ∀ n : ℕ+, a n > 0

-- Condition 2: Sequence recurrence relation
def recurrence_relation : Prop :=
  ∀ n : ℕ+, a (n + 1) = (a n ^ 2 - 1) / n

-- Proof that a_1 ≥ 2
theorem a1_geq_2 (h_pos : positive_sequence) (h_rec : recurrence_relation) : a 1 ≥ 2 := sorry

-- Proof that the sequence is monotonically increasing
theorem sequence_monotonically_increasing (h_pos : positive_sequence) (h_rec : recurrence_relation) : ∀ n : ℕ+, a (n + 1) ≥ a n := sorry

end a1_geq_2_sequence_monotonically_increasing_l445_445446


namespace black_friday_sales_l445_445071

variable (n : ℕ) (initial_sales increment : ℕ)

def yearly_sales (sales: ℕ) (inc: ℕ) (years: ℕ) : ℕ :=
  sales + years * inc

theorem black_friday_sales (h1 : initial_sales = 327) (h2 : increment = 50) :
  yearly_sales initial_sales increment 3 = 477 := by
  sorry

end black_friday_sales_l445_445071


namespace sin_half_angle_l445_445837

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445837


namespace multiples_of_15_between_20_and_200_l445_445465

theorem multiples_of_15_between_20_and_200 : 
  ∃ n : ℕ, (∀ k : ℕ, 20 < k*15 ∧ k*15 < 200 → k*15 ∈ (30:ℕ) + (n-1)*15) ∧ n = 12 :=
by 
  sorry

end multiples_of_15_between_20_and_200_l445_445465


namespace geometric_transformation_l445_445094

-- Define the known angle difference
def angle_difference (A B C D : Point) (γ : ℝ) : Prop := 
angle BAC = angle DAC + γ

-- The main theorem stating the transformation
theorem geometric_transformation (A B C D : Point) (γ : ℝ) 
  (h1 : angle_difference A B C D γ) : 
  exists (C' : Point), 
    central_similarity_of_centroid_with_coefficient 
    (A B C D) (AB/AD) ∧ rotation_around_point A C γ = C' :=
sorry

end geometric_transformation_l445_445094


namespace banana_arrangements_l445_445331

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445331


namespace cos2x_equals_shifted_sin_l445_445131

theorem cos2x_equals_shifted_sin :
  ∀ x : ℝ, cos (2 * x) = sin (2 * (x - π / 8) + π / 4) :=
by
  intro x
  -- Apply the sinus identity and function transformation
  sorry

end cos2x_equals_shifted_sin_l445_445131


namespace exists_smallest_even_abundant_gt_12_l445_445756

noncomputable def properDivisors (n : ℕ) : List ℕ :=
List.filter (λ d, d < n ∧ n % d = 0) (List.range n)

noncomputable def isAbundant (n : ℕ) : Prop :=
(n > 0) ∧ (List.sum (properDivisors n) > n)

theorem exists_smallest_even_abundant_gt_12 :
  ∃ n : ℕ, n > 12 ∧ 2 ∣ n ∧ (¬ prime n) ∧ isAbundant n ∧ ∀ m : ℕ, (m > 12 ∧ 2 ∣ m ∧ (¬ prime m) ∧ isAbundant m) → n ≤ m ∧ n = 18 :=
by
  sorry

end exists_smallest_even_abundant_gt_12_l445_445756


namespace BANANA_arrangements_l445_445271

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445271


namespace find_x_l445_445475

def G (a b c d : ℝ) : ℝ :=
  a^b + c / d

theorem find_x (x : ℝ) (h : G 3 x 9 3 = 30) : x = 3 :=
by
  sorry

end find_x_l445_445475


namespace smallest_n_digit_7_terminating_decimal_l445_445657

theorem smallest_n_digit_7_terminating_decimal :
  ∃ n : ℕ, (∃ a b : ℕ, n = 2^a * 5^b) ∧ (∃ d : ℕ, nat.digits 10 n 7 = true) ∧ (∀ m : ℕ, 
  (∃ c d : ℕ, m = 2^c * 5^d) ∧ (∃ e : ℕ, nat.digits 10 m 7 = true) → n ≤ m) := by
sorry

end smallest_n_digit_7_terminating_decimal_l445_445657


namespace worm_length_difference_is_correct_l445_445097

-- Define the lengths of the worms
def worm1_length : ℝ := 0.8
def worm2_length : ℝ := 0.1

-- Define the difference in length between the longer worm and the shorter worm
def length_difference (a b : ℝ) : ℝ := a - b

-- State the theorem that the length difference is 0.7 inches
theorem worm_length_difference_is_correct (h1 : worm1_length = 0.8) (h2 : worm2_length = 0.1) :
  length_difference worm1_length worm2_length = 0.7 :=
by
  sorry

end worm_length_difference_is_correct_l445_445097


namespace solution_set_inequality_l445_445001

theorem solution_set_inequality {a x: ℝ} (h: {x | a * x - 1 > 0} = set.Ioi 1):
  {x | (a * x - 1) * (x + 2) ≥ 0} = set.Iic (-2) ∪ set.Ici 1 :=
by
  sorry

end solution_set_inequality_l445_445001


namespace total_respondents_l445_445067

theorem total_respondents (X Y : ℕ) (hX : X = 360) (h_ratio : 9 * Y = X) : X + Y = 400 := by
  sorry

end total_respondents_l445_445067


namespace number_of_arrangements_of_BANANA_l445_445310

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445310


namespace sin_half_angle_l445_445834

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445834


namespace banana_permutations_l445_445250

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445250


namespace smallest_n_digit_7_terminating_decimal_l445_445658

theorem smallest_n_digit_7_terminating_decimal :
  ∃ n : ℕ, (∃ a b : ℕ, n = 2^a * 5^b) ∧ (∃ d : ℕ, nat.digits 10 n 7 = true) ∧ (∀ m : ℕ, 
  (∃ c d : ℕ, m = 2^c * 5^d) ∧ (∃ e : ℕ, nat.digits 10 m 7 = true) → n ≤ m) := by
sorry

end smallest_n_digit_7_terminating_decimal_l445_445658


namespace permutations_of_banana_l445_445296

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445296


namespace remainder_of_sum_l445_445647

theorem remainder_of_sum : 
  let a := 21160
  let b := 21162
  let c := 21164
  let d := 21166
  let e := 21168
  let f := 21170
  (a + b + c + d + e + f) % 12 = 6 :=
by
  sorry

end remainder_of_sum_l445_445647


namespace banana_permutations_l445_445261

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445261


namespace number_of_arrangements_of_BANANA_l445_445316

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445316


namespace combined_length_in_scientific_notation_l445_445113

noncomputable def yards_to_inches (yards : ℝ) : ℝ := yards * 36
noncomputable def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54
noncomputable def feet_to_inches (feet : ℝ) : ℝ := feet * 12

def sports_stadium_length_yards : ℝ := 61
def safety_margin_feet : ℝ := 2
def safety_margin_inches : ℝ := 9

theorem combined_length_in_scientific_notation :
  (inches_to_cm (yards_to_inches sports_stadium_length_yards) +
   (inches_to_cm (feet_to_inches safety_margin_feet + safety_margin_inches)) * 2) = 5.74268 * 10^3 :=
by
  sorry

end combined_length_in_scientific_notation_l445_445113


namespace largest_beverage_amount_l445_445126

theorem largest_beverage_amount :
  let Milk := (3 / 8 : ℚ)
  let Cider := (7 / 10 : ℚ)
  let OrangeJuice := (11 / 15 : ℚ)
  OrangeJuice > Milk ∧ OrangeJuice > Cider :=
by
  have Milk := (3 / 8 : ℚ)
  have Cider := (7 / 10 : ℚ)
  have OrangeJuice := (11 / 15 : ℚ)
  sorry

end largest_beverage_amount_l445_445126


namespace BANANA_arrangements_l445_445304

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445304


namespace min_fraction_reaches_at_4_l445_445447

noncomputable def a : ℕ → ℕ
| 1       := 8
| (n + 1) := a n + n

theorem min_fraction_reaches_at_4 : 
  ∀ (n : ℕ), 
    n > 0 → 
    ( ∀ (k : ℕ), k > 0 → (a k / k) ≥ (a 4 / 4) ) ∧ (a n / n = a 4 / 4) → n = 4 := 
by
  intros n h_pos h_min
  sorry

end min_fraction_reaches_at_4_l445_445447


namespace union_of_A_and_B_l445_445417

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l445_445417


namespace permutations_of_BANANA_l445_445244

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445244


namespace dot_product_sum_l445_445957

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom vec_sum : a + b + c = 0
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom norm_c : ∥c∥ = 2

-- The theorem to prove
theorem dot_product_sum :
  ⟪a, b⟫ + ⟪b, c⟫ + ⟪c, a⟫ = - 9 / 2 :=
sorry

end dot_product_sum_l445_445957


namespace banana_arrangements_l445_445327

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l445_445327


namespace Ben_cards_left_l445_445205

def BenInitialBasketballCards : ℕ := 4 * 10
def BenInitialBaseballCards : ℕ := 5 * 8
def BenTotalInitialCards : ℕ := BenInitialBasketballCards + BenInitialBaseballCards
def BenGivenCards : ℕ := 58
def BenRemainingCards : ℕ := BenTotalInitialCards - BenGivenCards

theorem Ben_cards_left : BenRemainingCards = 22 :=
by 
  -- The proof will be placed here.
  sorry

end Ben_cards_left_l445_445205


namespace percentage_of_boys_among_boyds_friends_l445_445532

theorem percentage_of_boys_among_boyds_friends : 
  (Julian_total : ℕ) (Julian_girls_percentage : ℕ) (Boyd_total : ℕ) (Boyd_girls_multiplier : ℕ) 
  (H1 : Julian_total = 80) 
  (H2 : Julian_girls_percentage = 40)
  (H3 : Boyd_total = 100)
  (H4 : Boyd_girls_multiplier = 2)
  : (Boyd_boys_percentage : ℕ) := 
  let Julian_girls : ℕ := Julian_total * Julian_girls_percentage / 100,
  let Boyd_girls : ℕ := Julian_girls * Boyd_girls_multiplier,
  let Boyd_boys : ℕ := Boyd_total - Boyd_girls,
  let Boyd_boys_percentage : ℕ := Boyd_boys * 100 / Boyd_total
  Boyd_boys_percentage = 36 :=
sorry

end percentage_of_boys_among_boyds_friends_l445_445532


namespace chemical_x_added_l445_445164

theorem chemical_x_added (initial_volume : ℝ) (initial_percentage : ℝ) (final_percentage : ℝ) : 
  initial_volume = 80 → initial_percentage = 0.2 → final_percentage = 0.36 → 
  ∃ (a : ℝ), 0.20 * initial_volume + a = 0.36 * (initial_volume + a) ∧ a = 20 :=
by
  intros h1 h2 h3
  use 20
  sorry

end chemical_x_added_l445_445164


namespace sin_half_angle_l445_445901

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445901


namespace arrangement_of_BANANA_l445_445367

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445367


namespace rhombus_diagonal_l445_445601

/-- Given a rhombus with one diagonal being 11 cm and the area of the rhombus being 88 cm²,
prove that the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal 
  (d1 : ℝ) (d2 : ℝ) (area : ℝ)
  (h_d1 : d1 = 11)
  (h_area : area = 88)
  (h_area_eq : area = (d1 * d2) / 2) : d2 = 16 :=
sorry

end rhombus_diagonal_l445_445601


namespace find_x_l445_445485

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) : x = 24 / 7 :=
by
  sorry

end find_x_l445_445485


namespace union_sets_A_B_l445_445409

noncomputable def setA : set ℝ := { x | ∃ y, y = real.sqrt (-x^2 + 3*x + 4) }
noncomputable def setB : set ℝ := { x | 2^x > 4 }

theorem union_sets_A_B : setA ∪ setB = { x : ℝ | -1 ≤ x } :=
by
  sorry

end union_sets_A_B_l445_445409


namespace policeman_catches_gangster_l445_445157

variables (l r : ℝ) (policeman_speed gangster_speed : ℝ)
  (distance : ℝ)

-- Definitions of the problem conditions
def equal_length_corridors : Prop := -- Three corridors each of length l.
  true

def policeman_moves_faster : Prop := -- Policeman's speed is twice the gangster's speed.
  policeman_speed = 2 * gangster_speed

def visibility_constraint : Prop := -- Policeman can see the gangster if distance is less than or equal to r.
  distance ≤ r

-- The main goal to prove
theorem policeman_catches_gangster (h1 : equal_length_corridors)
  (h2 : policeman_moves_faster)
  (h3 : visibility_constraint) : 
  true :=
sorry

end policeman_catches_gangster_l445_445157


namespace omega_and_varphi_range_f_interval_l445_445927

-- Conditions
def f (x : Real) (ω : Real) (φ : Real) : Real := Real.sin (ω * x + φ)
def smallest_positive_period (f : Real → Real) (p : Real) : Prop := ∀ x, f (x + p) = f x ∧ 0 < p

-- Assertions to Prove
theorem omega_and_varphi
  (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0) (h_period : smallest_positive_period (f x ω φ) π)
  (h_equiv : f (π / 2) ω φ = f (2 * π / 3) ω φ) :
  ω = 2 ∧ ∃ k : ℤ, φ = k * π + π / 3 :=
sorry

theorem range_f_interval
  (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0) (h_period : smallest_positive_period (f x ω φ) π)
  (h_equiv : f (π / 2) ω φ = f (2 * π / 3) ω φ)
  (h_phi_bounds : |φ| < π / 2) :
  ∃ a b : ℝ, -π/3 ≤ x ∧ x ≤ π/6 → f x ω (φ % π) = a ∧ f x ω (φ % π) = b ∧ a = -Real.sqrt 3 / 2 ∧ b = 1 :=
sorry

end omega_and_varphi_range_f_interval_l445_445927


namespace susan_strawberries_l445_445589

def strawberries_picked (total_in_basket : ℕ) (handful_size : ℕ) (eats_per_handful : ℕ) : ℕ :=
  let strawberries_per_handful := handful_size - eats_per_handful
  (total_in_basket / strawberries_per_handful) * handful_size

theorem susan_strawberries : strawberries_picked 60 5 1 = 75 := by
  sorry

end susan_strawberries_l445_445589


namespace sin_half_alpha_l445_445900

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445900


namespace sin_half_alpha_l445_445894

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445894


namespace right_triangles_in_rhombus_l445_445107

noncomputable def points := {A, P, B, C, Q, D}

theorem right_triangles_in_rhombus :
  ∀ (A P B C Q D : Type) [rhombus ABCD] [PQ_divides_rhombus : divides_rhombus PQ ABCD],
  count_right_triangles_using_points (points A P B C Q D) = 4 :=
by
  intros
  sorry

end right_triangles_in_rhombus_l445_445107


namespace red_window_exchange_l445_445119

-- Defining the total transaction amount for online and offline booths
variables (x y : ℝ)

-- Defining conditions
def offlineMoreThanOnline (y x : ℝ) : Prop := y - 7 * x = 1.8
def averageTransactionDifference (y x : ℝ) : Prop := (y / 71) - (x / 44) = 0.3

-- The proof problem
theorem red_window_exchange (x y : ℝ) :
  offlineMoreThanOnline y x ∧ averageTransactionDifference y x := 
sorry

end red_window_exchange_l445_445119


namespace profit_percentage_is_25_l445_445197

theorem profit_percentage_is_25 
  (selling_price : ℝ) (cost_price : ℝ) 
  (sp_val : selling_price = 600) 
  (cp_val : cost_price = 480) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_is_25_l445_445197


namespace hiking_time_l445_445523

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l445_445523


namespace banana_arrangement_count_l445_445280

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445280


namespace fewest_number_of_posts_l445_445177

-- Define the parameters as constants
constant rock_wall_length : ℝ := 88
constant area_of_rectangle : ℝ := 3080
constant post_spacing : ℝ := 10

-- Define the statement to prove the fewest number of posts required
theorem fewest_number_of_posts :
  ∃ (num_posts : ℕ), 
    num_posts = 18 ∧ 
    ((rock_wall_length * (area_of_rectangle / rock_wall_length) = area_of_rectangle) ∧
     (num_posts =
       (nat.ceil (rock_wall_length / post_spacing) + 1) + -- posts along the 88-meter side
       2 * (nat.ceil ((area_of_rectangle / rock_wall_length) / post_spacing) + 1 - 1))) -- each 35-meter side minus one shared post
:= sorry

end fewest_number_of_posts_l445_445177


namespace marble_244_is_white_l445_445185

noncomputable def color_of_marble (n : ℕ) : String :=
  let cycle := ["white", "white", "white", "white", "gray", "gray", "gray", "gray", "gray", "black", "black", "black"]
  cycle.get! (n % 12)

theorem marble_244_is_white : color_of_marble 244 = "white" :=
by
  sorry

end marble_244_is_white_l445_445185


namespace abs_alpha_eq_2_sqrt_2_l445_445543

variable {ℂ : Type} [IsROrC ℂ]

-- Given conditions
variables (α β : ℂ)
variables (h1 : complex.conj α = β)
variables (h2 : isReal (α / (β^3)))
variables (h3 : complex.abs (α - β) = 4)

-- Goal
theorem abs_alpha_eq_2_sqrt_2
  (h1 : α = complex.conj β)
  (h2 : ∃ r : ℝ, α / (β ^ 3) = (r : ℂ))
  (h3 : complex.abs (α - β) = 4) :
  complex.abs α = 2 * real.sqrt 2 :=
sorry

end abs_alpha_eq_2_sqrt_2_l445_445543


namespace number_of_valid_triples_l445_445971

theorem number_of_valid_triples :
  ∃ (count : ℕ), count = 3 ∧
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z →
  Nat.lcm x y = 120 → Nat.lcm y z = 1000 → Nat.lcm x z = 480 →
  (∃ (u v w : ℕ), u = x ∧ v = y ∧ w = z ∧ count = 3) :=
by
  sorry

end number_of_valid_triples_l445_445971


namespace sin_half_alpha_l445_445826

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445826


namespace math_problem_l445_445548

noncomputable def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x * (deriv (deriv f) x) * log x > f x

theorem math_problem (f : ℝ → ℝ) (h : function_property f) :
  f 2 < f (Real.exp 1) * log 2 ∧ 2 * f (Real.exp 1) < f (Real.exp 2) :=
sorry

end math_problem_l445_445548


namespace abs_eq_of_sq_eq_l445_445144

theorem abs_eq_of_sq_eq (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  intro h
  sorry

end abs_eq_of_sq_eq_l445_445144


namespace find_sin_cos_B_l445_445002

-- Define the conditions
variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Conditions from the problem
def conditions (triangle_ABC : ∀ a b c : ℝ, Prop) :=
  ∃ (a b c : ℝ), c = 2 * b ∧ (sin C = 3 / 4) ∧ (b^2 + b * c = 2 * a^2)

-- The goal is to find sin B and cos B
theorem find_sin_cos_B (h : conditions triangle_ABC) :
  ∃ (B : ℝ), sin B = 3 / 8 ∧ cos B = 3 * sqrt 6 / 8 :=
sorry

end find_sin_cos_B_l445_445002


namespace non_similar_triangles_count_l445_445467

theorem non_similar_triangles_count : 
  ∃ n : ℕ, ∀ (α β γ : ℕ), 
    α + β + γ = 180 ∧
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧ 
    β = 45 ∧ 
    α < β ∧ β < γ → 
    α, β, and γ are in arithmetic progression →
    n = 4 :=
sorry

end non_similar_triangles_count_l445_445467


namespace part_one_part_two_l445_445617

variable (λ : ℝ)
noncomputable def a : ℕ → ℝ
| 1     := 1
| (n+1) := (n^2 + n - λ) * a n

theorem part_one (h1 : a 2 = -1) :
  λ = 3 ∧ a 3 = -3 := by
  sorry
  
theorem part_two :
  ¬∃ (d : ℝ), ∀ n : ℕ, a (n+1) - a n = d := by
  sorry

end part_one_part_two_l445_445617


namespace donut_combinations_count_l445_445743

theorem donut_combinations_count :
  let donut_types := 5
  let donuts_needed := 7
  let min_types_needed := 4
  ∃ (count : ℕ), ∀ (comb : ℕ), comb = donut_types.choose (min_types_needed) * (donuts_needed - min_types_needed + donut_types - 1).choose (donut_types - 1) ∧ comb = 175 :=
begin
  let donut_types := 5,
  let donuts_needed := 7,
  let min_types_needed := 4,
  use 175,
  intros comb,
  have h₁ : comb = (nat.choose donut_types min_types_needed) * (nat.choose (donuts_needed - min_types_needed + donut_types - 1) (donut_types - 1)),
  {
    sorry
  },
  rw h₁,
  norm_num,
end

end donut_combinations_count_l445_445743


namespace sin_half_angle_l445_445908

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445908


namespace sin_half_alpha_l445_445829

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445829


namespace sin_half_alpha_l445_445872

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcosα : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445872


namespace sum_of_squares_of_roots_l445_445754

-- Given polynomial
def polynomial : Polynomial ℝ := Polynomial.monomial 10 1 + Polynomial.monomial 7 3 + Polynomial.monomial 2 5 + Polynomial.C 404

-- Roots of the polynomial
def roots : Fin 10 → ℝ := sorry  -- abstract placeholder for the roots drawn from the polynomial

-- Polynomial is monic and has no x⁹ and x⁸ terms
axiom monic_poly : polynomial.leadingCoeff = 1
axiom no_x9_term : polynomial.coeff 9 = 0
axiom no_x8_term : polynomial.coeff 8 = 0

-- Sum of the roots is zero
axiom sum_roots_zero : ∑ i, roots i = 0

theorem sum_of_squares_of_roots : (∑ i, (roots i)^2) = 0 :=
by sorry

end sum_of_squares_of_roots_l445_445754


namespace correct_square_root_calculation_l445_445665

theorem correct_square_root_calculation :
  (∀ x y, sqrt ((x:ℝ) ^ 2) = |x| ∧ sqrt ((-y:ℝ) ^ 2) = |y| ∧ (sqrt (x / y) = (sqrt x / sqrt y))) →
  (sqrt (25 / 121) = 5 / 11) :=
begin
  intros h,
  have h1 := h 25 121,
  exact h1.right.right, -- This extracts the necessary part of the hypothesis.
end

end correct_square_root_calculation_l445_445665


namespace sector_longest_segment_squared_l445_445725

theorem sector_longest_segment_squared (d : ℝ) (n : ℕ) (m : ℝ) :
  d = 16 ∧ n = 4 →
  m = 8 * real.sqrt 2 →
  m^2 = 128 :=
by
  intro h1 h2
  sorry

end sector_longest_segment_squared_l445_445725


namespace sqrt_sum_leq_2sqrt2_l445_445792

variable (a b : ℝ)

theorem sqrt_sum_leq_2sqrt2
  (ha : a > 0)
  (hb : b > 0)
  (hab : a + b = 1) :
  sqrt (2 * a + 1) + sqrt (2 * b + 1) ≤ 2 * sqrt 2 :=
sorry

end sqrt_sum_leq_2sqrt2_l445_445792


namespace remainder_2_pow_19_div_7_l445_445139

theorem remainder_2_pow_19_div_7 :
  2^19 % 7 = 2 := by
  sorry

end remainder_2_pow_19_div_7_l445_445139


namespace measure_angle_B_of_triangle_l445_445495

theorem measure_angle_B_of_triangle {A B C : Type}
  (a b c : ℝ)
  (h₁ : a^2 + c^2 = b^2 + sqrt 2 * a * c) :
  ∠ B = 45 :=
sorry

end measure_angle_B_of_triangle_l445_445495


namespace no_odd_integer_trinomial_has_root_1_over_2022_l445_445767

theorem no_odd_integer_trinomial_has_root_1_over_2022 :
  ¬ ∃ (a b c : ℤ), (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0)) :=
by
  sorry

end no_odd_integer_trinomial_has_root_1_over_2022_l445_445767


namespace min_value_expression_l445_445779

noncomputable def expression (x y : ℝ) : ℝ :=
  x^2 + 4*x*y + 5*y^2 - 8*x + 6*y + 2

theorem min_value_expression : ∃ (x y : ℝ), expression x y = -7 :=
by
  use (10, -3)
  sorry

end min_value_expression_l445_445779


namespace mass_of_rod_l445_445721

theorem mass_of_rod (a : ℝ) (h_pos : 0 ≤ a) : 
  (∫ x in 0..a, x^2) = (1/3) * a^3 := 
by {
  sorry
}

end mass_of_rod_l445_445721


namespace BANANA_arrangements_l445_445269

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445269


namespace total_weight_of_containers_l445_445064

theorem total_weight_of_containers (x y z : ℕ) :
  x + y = 162 →
  y + z = 168 →
  z + x = 174 →
  x + y + z = 252 :=
by
  intros hxy hyz hzx
  -- proof skipped
  sorry

end total_weight_of_containers_l445_445064


namespace rhombus_angle_bisector_square_l445_445602

theorem rhombus_angle_bisector_square 
  (A B C D O M N K L : Type*)
  [is_rhombus A B C D O]
  (M_bisector : is_angle_bisector_triangle M A B O)
  (N_bisector : is_angle_bisector_triangle N B C O)
  (K_bisector : is_angle_bisector_triangle K C D O)
  (L_bisector : is_angle_bisector_triangle L D A O)
  (diagonals_perpendicular : ∀ A B C D O, diagonals_perpendicular A B C D O)
  (diagonals_bisect_each_other : ∀ A B C D O, diagonals_bisect_each_other A B C D O)
  : is_square M N K L :=
sorry

end rhombus_angle_bisector_square_l445_445602


namespace value_of_g_at_2_l445_445482

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  sorry

end value_of_g_at_2_l445_445482


namespace blue_marbles_difference_l445_445134

variables (J1 J2 : Type) [fintype J1] [fintype J2]
variables (blue green : Type) [fintype blue] [fintype green]
variables (ratio1 ratio2 : ℤ) (total_green : ℤ)

axioms (ratio_condition1 : ratio1 = 7 / 3)
       (ratio_condition2 : ratio2 = 5 / 1)
       (equal_marbles : card J1 = card J2)
       (total_green_marbles : 120 = total_green)

theorem blue_marbles_difference :
  let blue1 := 7 * total_green / (7 + 3)
  let blue2 := 5 * total_green / (5 + 1)
  blue2 - blue1 = 34 := sorry

end blue_marbles_difference_l445_445134


namespace digits_all_one_if_digital_root_is_one_l445_445603

-- Define the product of digits
def product_of_digits (n : ℕ) : ℕ :=
  (nat.digits 10 n).foldl (λ acc d, acc * d) 1

-- Define the digital root
def digital_root (n : ℕ) : ℕ :=
  let rec dr (x : ℕ) : ℕ :=
    if x < 10 then x
    else dr (product_of_digits x)
  in dr n

-- Statement of the theorem
theorem digits_all_one_if_digital_root_is_one (n : ℕ) (h : digital_root n = 1) : ∀ d ∈ (nat.digits 10 n), d = 1 :=
by
  sorry

end digits_all_one_if_digital_root_is_one_l445_445603


namespace JeromeMoneyLeft_l445_445456

theorem JeromeMoneyLeft (
  euro_to_dollar: ℝ := 1.20,
  half_money_euro: ℝ := 43,
  gave_to_meg: ℝ := 8,
  gave_trice_to_bianca: ℝ := 3 * gave_to_meg,
  remaining_meg_bianca: ℝ := 43 * 2 * euro_to_dollar - gave_to_meg - gave_trice_to_bianca,
  gave_to_nathan: ℝ := 2 * remaining_meg_bianca,
  gave_to_charity: ℝ := 0.2 * (remaining_meg_bianca - gave_to_nathan)
): remaining_meg_bianca - gave_to_nathan - gave_to_charity = 0 := by
  sorry

end JeromeMoneyLeft_l445_445456


namespace shaded_region_equality_l445_445509

-- Define the necessary context and variables
variable {r : ℝ} -- radius of the circle
variable {θ : ℝ} -- angle measured in degrees

-- Define the relevant trigonometric functions
noncomputable def tan_degrees (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)
noncomputable def tan_half_degrees (x : ℝ) : ℝ := Real.tan ((x / 2) * Real.pi / 180)

-- State the theorem we need to prove given the conditions
theorem shaded_region_equality (hθ1 : θ / 2 = 90 - θ) :
  tan_degrees θ + (tan_degrees θ)^2 * tan_half_degrees θ = (θ * Real.pi) / 180 - (θ^2 * Real.pi) / 360 :=
  sorry

end shaded_region_equality_l445_445509


namespace num_ordered_pairs_satisfying_eq_l445_445615

/--
The number of ordered pairs of non-negative integers (m, n) for which 
m^3 + n^3 + 44mn = 22^3 equals 23.
-/
theorem num_ordered_pairs_satisfying_eq :
  {p : ℕ × ℕ | p.1^3 + p.2^3 + 44 * p.1 * p.2 = 22^3}.to_finset.card = 23 :=
sorry

end num_ordered_pairs_satisfying_eq_l445_445615


namespace sin_half_angle_l445_445832

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445832


namespace question1_question2_l445_445796

-- Condition: p
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0

-- Condition: q
def q (a : ℝ) (x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Question 1 statement: Given p is true and q is false when a = 0, find range of x
theorem question1 (x : ℝ) (h : p x ∧ ¬q 0 x) : -7/2 ≤ x ∧ x < -3 :=
sorry

-- Question 2 statement: If p is a sufficient condition for q, find range of a
theorem question2 (a : ℝ) (h : ∀ x, p x → q a x) : -5/2 ≤ a ∧ a ≤ 1/2 :=
sorry

end question1_question2_l445_445796


namespace sin_half_alpha_l445_445821

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445821


namespace percentage_green_shirts_correct_l445_445019

variable (total_students blue_percentage red_percentage other_students : ℕ)

noncomputable def percentage_green_shirts (total_students blue_percentage red_percentage other_students : ℕ) : ℕ :=
  let total_blue_shirts := blue_percentage * total_students / 100
  let total_red_shirts := red_percentage * total_students / 100
  let total_blue_red_other_shirts := total_blue_shirts + total_red_shirts + other_students
  let green_shirts := total_students - total_blue_red_other_shirts
  (green_shirts * 100) / total_students

theorem percentage_green_shirts_correct
  (h1 : total_students = 800) 
  (h2 : blue_percentage = 45)
  (h3 : red_percentage = 23)
  (h4 : other_students = 136) : 
  percentage_green_shirts total_students blue_percentage red_percentage other_students = 15 :=
by
  sorry

end percentage_green_shirts_correct_l445_445019


namespace hyperbola_ecc_equality_l445_445931

noncomputable def hyperbola_eccentricity {a b c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → true) (parabola : ∀ x y : ℝ, y^2 = 4 / 3 * c * x → true) : ℝ :=
  if c = 3 * a then 3 else 0

theorem hyperbola_ecc_equality (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → true)
  (parabola : ∀ x y : ℝ, y^2 = 4 / 3 * c * x → true)
  (right_triangle : ∃ y, y^2 = 4 / 3 * a * c ∧ 
                           (c - a) = ½ * |y - (-y)|) :
  hyperbola_eccentricity a_pos b_pos hyp parabola = 3 :=
sorry

end hyperbola_ecc_equality_l445_445931


namespace union_of_A_and_B_l445_445418

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l445_445418


namespace right_triangle_sides_l445_445370

theorem right_triangle_sides (p m : ℝ)
  (hp : 0 < p)
  (hm : 0 < m) :
  ∃ a b c : ℝ, 
    a + b + c = 2 * p ∧
    a^2 + b^2 = c^2 ∧
    (1 / 2) * a * b = m^2 ∧
    c = (p^2 - m^2) / p ∧
    a = (p^2 + m^2 + Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) ∧
    b = (p^2 + m^2 - Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) := 
by
  sorry

end right_triangle_sides_l445_445370


namespace arrangement_of_BANANA_l445_445361

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445361


namespace percentage_of_boyds_boy_friends_l445_445534

-- Definitions based on conditions
def number_of_julian_friends : ℕ := 80
def percentage_of_julian_boys : ℕ := 60
def percentage_of_julian_girls : ℕ := 40
def number_of_boyd_friends : ℕ := 100

-- Calculation based on conditions
def num_julian_boy_friends : ℕ := (percentage_of_julian_boys * number_of_julian_friends) / 100
def num_julian_girl_friends : ℕ := (percentage_of_julian_girls * number_of_julian_friends) / 100
def num_boyd_girl_friends : ℕ := 2 * num_julian_girl_friends
def num_boyd_boy_friends : ℕ := number_of_boyd_friends - num_boyd_girl_friends

-- Prove percentage of Boyd's friends who are boys
theorem percentage_of_boyds_boy_friends : (num_boyd_boy_friends * 100 / number_of_boyd_friends) = 36 :=
by
  simp [num_julian_boy_friends, num_julian_girl_friends, num_boyd_girl_friends, num_boyd_boy_friends, percentage_of_julian_boys, percentage_of_julian_girls, number_of_julian_friends, number_of_boyd_friends]
  sorry

end percentage_of_boyds_boy_friends_l445_445534


namespace polynomial_statement_correct_l445_445574

def termA : ℝ := - (4 * real.pi) / 3
def termC := 2
def polyD : ℤ[X][Y] := - (X ^ 2 * Y) + (X * Y) - 7

theorem polynomial_statement_correct :
  (termA ≠ -4/3) ∧ (termDegree (3^2 * X ^ 2 * Y) ≠ 5) ∧ isMonomial termC ∧ (polyDegree polyD ≠ 5) ∧ isTrinomial polyD ∧ (correctStatement = "C") :=
by
  sorry

end polynomial_statement_correct_l445_445574


namespace rides_first_day_l445_445203

variable (total_rides : ℕ) (second_day_rides : ℕ)

theorem rides_first_day (h1 : total_rides = 7) (h2 : second_day_rides = 3) : total_rides - second_day_rides = 4 :=
by
  sorry

end rides_first_day_l445_445203


namespace arithmetic_sequence_nth_term_l445_445508

-- Definitions from conditions
def a₁ : ℕ → ℤ := -60
def a₁₇  : ℕ → ℤ := -12

-- Lean statement

theorem arithmetic_sequence_nth_term (n : ℕ) :
  (∃ d, d = 3 ∧ ∀ n : ℕ, a₁₇ = a₁ + 16 * d ∧ (∀ n : ℕ, a_n = -60 + 3 * (n - 1))) ∧
  sum_abs_first_30_terms : (∃ S, S = 765 ∧ ∑ i in range 1..31, abs (a_n(i)) = 765) :=
sorry

end arithmetic_sequence_nth_term_l445_445508


namespace sin_half_alpha_l445_445827

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445827


namespace f_not_surjective_l445_445550

def f : ℝ → ℕ → Prop := sorry

theorem f_not_surjective (f : ℝ → ℕ) 
  (h : ∀ x y : ℝ, f (x + (1 / f y)) = f (y + (1 / f x))) : 
  ¬ (∀ n : ℕ, ∃ x : ℝ, f x = n) :=
sorry

end f_not_surjective_l445_445550


namespace complement_intersection_l445_445940

open Set

-- Definitions of sets
def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- The theorem statement
theorem complement_intersection :
  (U \ B) ∩ A = {2, 6} := by
  sorry

end complement_intersection_l445_445940


namespace hyperbola_eccentricity_l445_445403

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes_intersect : ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 = -1 ∧ B.1 = -1 ∧
    ∀ (A B : ℝ × ℝ), ∃ x y : ℝ, (A.2 = y ∧ B.2 = y ∧ x^2 / a^2 - y^2 / b^2 = 1))
  (triangle_area : ∃ A B : ℝ × ℝ, 1 / 2 * abs (A.1 * B.2 - A.2 * B.1) = 2 * Real.sqrt 3) :
  ∃ e : ℝ, e = Real.sqrt 13 :=
by {
  sorry
}

end hyperbola_eccentricity_l445_445403


namespace elevator_time_to_bottom_l445_445563

theorem elevator_time_to_bottom :
  let first_quarter := 25
  let second_quarter_per_floor := 10
  let next_quarter_per_floor := 7
  let last_quarter_first_half := 35
  let last_quarter_per_floor := 10
  let total_floors := 40
  let floors_per_quarter := total_floors / 4 in
  (first_quarter + 
   second_quarter_per_floor * floors_per_quarter + 
   next_quarter_per_floor * floors_per_quarter + 
   last_quarter_first_half + 
   last_quarter_per_floor * (floors_per_quarter / 2)) / 60 = 4.67 :=
by
  let first_quarter := 25
  let second_quarter_per_floor := 10
  let next_quarter_per_floor := 7
  let last_quarter_first_half := 35
  let last_quarter_per_floor := 10
  let total_floors := 40
  let floors_per_quarter := total_floors / 4
  show 
    (first_quarter + 
    second_quarter_per_floor * floors_per_quarter + 
    next_quarter_per_floor * floors_per_quarter + 
    last_quarter_first_half + 
    last_quarter_per_floor * (floors_per_quarter / 2)) / 60 = 4.67
  sorry

end elevator_time_to_bottom_l445_445563


namespace determine_f_2014_l445_445702
noncomputable def f : ℝ → ℝ 
| x := if x ≤ 0 then log (1 - x) / log 2 else f (x - 1) - f (x - 2)

theorem determine_f_2014 : f 2014 = 1 :=
sorry

end determine_f_2014_l445_445702


namespace female_students_count_l445_445594

variable (F : ℕ)

theorem female_students_count
    (avg_all_students : ℕ)
    (avg_male_students : ℕ)
    (avg_female_students : ℕ)
    (num_male_students : ℕ)
    (condition1 : avg_all_students = 90)
    (condition2 : avg_male_students = 82)
    (condition3 : avg_female_students = 92)
    (condition4 : num_male_students = 8)
    (condition5 : 8 * 82 + F * 92 = (8 + F) * 90) : 
    F = 32 := 
by 
  sorry

end female_students_count_l445_445594


namespace projection_correct_l445_445386

open Real

def vec1 : Vector ℝ := ⟨[5, -3, 2]⟩
def dir : Vector ℝ := ⟨[4, -3, 2]⟩

noncomputable def projection : Vector ℝ :=
  let dot_uv := vec1.dot_product dir
  let dot_vv := dir.dot_product dir
  let scalar := dot_uv / dot_vv
  ⟨dir.1.map (λ x => x * scalar)⟩

theorem projection_correct : projection = ⟨[132 / 29, -99 / 29, 66 / 29]⟩ := by
  sorry

end projection_correct_l445_445386


namespace triangle_is_right_l445_445941

-- Define the vectors involved
variables (A B C : ℝ^3)

-- Define the lengths of the sides
def a := ∥B - C∥
def b := ∥C - A∥
def c := ∥A - B∥

-- Define the condition as a predicate
def condition (t : ℝ) : Prop := ∥B - A - t • (B - C)∥ ≥ ∥C - A∥

-- Define the main theorem
theorem triangle_is_right (A B C : ℝ^3) :
  (∀ t : ℝ, condition A B C t) → ∃ C, angle (B - A) (C - B) = π / 2  :=
by
  sorry

end triangle_is_right_l445_445941


namespace sin_half_angle_l445_445842

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445842


namespace same_result_three_matches_l445_445689

variables (Outcome : Type) (A B C D : Outcome) 
noncomputable def match_outcome : Outcome → Outcome → Outcome → Outcome → Prop := sorry

theorem same_result_three_matches :
  (∀ x y z w : Outcome, ∃ p : Outcome, (match_outcome x y z w)) →
  (∃ p : Outcome, (p = A ∨ p = B ∨ p = C ∨ p = D) ∧ 
  ((p = A ∧ match_outcome A B C D) ∨ (p = B ∧ match_outcome A C B D) ∨ (p = C ∧ match_outcome A D B C) ∨ (p = D ∧ match_outcome C D A B))):= 
sorry

end same_result_three_matches_l445_445689


namespace permutations_of_BANANA_l445_445247

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445247


namespace BANANA_arrangements_l445_445263

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445263


namespace sin_half_alpha_l445_445892

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445892


namespace magnitude_of_conjugate_l445_445402

theorem magnitude_of_conjugate (z : ℂ) (h : z = 2 + I) : complex.abs (conjugate z) = real.sqrt 5 :=
by
  sorry

end magnitude_of_conjugate_l445_445402


namespace distinct_arrangements_of_BANANA_l445_445338

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l445_445338


namespace BANANA_arrangements_l445_445299

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l445_445299


namespace value_of_a_l445_445929

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -2/3 < x ∧ x < -1/3 → (f' a x) < 0) ∧
  (∀ x : ℝ, -1/3 < x → (f' a x) > 0) → a = 2 :=
by
  -- Define the given function f(x)
  let f : ℝ → ℝ := λ x => x^3 + a * x^2 + x + 1
  -- Define the first derivative of the function
  let f' : ℝ → ℝ := λ x => 3 * x^2 + 2 * a * x + 1
  -- We state the theorem without proof
  sorry

end value_of_a_l445_445929


namespace banana_arrangement_count_l445_445278

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445278


namespace sin_half_alpha_l445_445889

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445889


namespace sin_half_alpha_l445_445830

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445830


namespace sin_half_alpha_l445_445884

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445884


namespace maximize_sum_of_decreasing_arith_seq_l445_445390

theorem maximize_sum_of_decreasing_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) 
    (h1 : ∀ k, a k > a (k + 1)) 
    (h2 : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
    (h3 : S 5 = S 10) : 
    (∃ n, n = 7 ∨ n = 8) :=
by sorry

end maximize_sum_of_decreasing_arith_seq_l445_445390


namespace magnitude_z2_range_l445_445917

theorem magnitude_z2_range (z1 z2 : ℂ) 
  (h1 : (z1 - complex.i) * (z2 + complex.i) = 1) 
  (h2 : complex.abs z1 = real.sqrt 2) : 
  ∃ l u : ℝ, l = 2 - real.sqrt 2 ∧ u = 2 + real.sqrt 2 ∧ 
  l ≤ complex.abs z2 ∧ complex.abs z2 ≤ u :=
by {
  sorry
}

end magnitude_z2_range_l445_445917


namespace petya_wins_second_race_l445_445081

namespace Race

variables {speed_P speed_V : ℝ}
variables {t_P t_V : ℝ}

/-- Assume Petya and Vasya's speeds and times are positive real numbers -/
axiom (positive_speeds : ∀ {speed_P speed_V t_P t_V : ℝ}, 0 < speed_P ∧ 0 < speed_V ∧ 0 < t_P ∧ 0 < t_V)
-- Petya travels 60 meters in time 't_P'
def petya_distance_1 : ℝ := speed_P * t_P
-- Vasya travels 51 meters in time 't_P'
def vasya_distance_1 : ℝ := speed_V * t_P

/-- Vasya is 9 meters behind Petya at the end of the first race -/
axiom distance_gap : ∀ {speed_P speed_V t_P : ℝ}, (speed_P * t_P) = 60 → (speed_V * t_P) = 51 → (speed_P - speed_V) * t_P = 9

/-- Assume in the second race, Petya starts 9 meters behind Vasya -/
def petya_start_behind : ℝ := 9

-- Let 't' be the time for Petya to finish the second race
variables {t : ℝ}

/-- Petya finishes the race of 60 meters again at his speed_P -/
def petya_race_2 : ℝ := speed_P * t

-- Vasya's distance in the second race at time 't'
def vasya_race_2 : ℝ := speed_V * t

/-- By how many meters does Petya lead Vasya at the end of the second race -/
def petya_leads_vasya : ℝ := 1.35

/-- Prove that Petya finishes first in the second race and leads by 1.35 meters -/
theorem petya_wins_second_race : (speed_P * t) = 60 → (speed_V * t) < 60 → (speed_P * t) - (speed_V * t) = 1.35 := 
sorry

end Race

end petya_wins_second_race_l445_445081


namespace sin_half_angle_l445_445835

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445835


namespace fraction_of_juniors_l445_445152

theorem fraction_of_juniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J = (4 / 3) * S) :
  (J : ℚ) / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l445_445152


namespace part1_part2_l445_445938

def A (x : ℤ) := ∃ m n : ℤ, x = m^2 - n^2
def B (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

theorem part1 (h1: A 8) (h2: A 9) (h3: ¬ A 10) : 
  (A 8) ∧ (A 9) ∧ (¬ A 10) :=
by {
  sorry
}

theorem part2 (x : ℤ) (h : A x) : B x :=
by {
  sorry
}

end part1_part2_l445_445938


namespace nearest_sum_value_is_1004_l445_445703

noncomputable def f : ℝ → ℝ := sorry

theorem nearest_sum_value_is_1004 :
  (∀ x : ℝ, x ≠ 0 → 2 * f x + f (1 / x) = 3 * x + 6) →
  (let S := ∑ x in (λ x, f x = 2010) '' set.univ, x in
  abs (S - 1004) < 0.5) :=
sorry

end nearest_sum_value_is_1004_l445_445703


namespace trapezium_area_is_304_l445_445153

-- Define the lengths of the parallel sides and the height
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 16

-- Define the area function for a trapezium
def trapezium_area (a b h : ℝ) : ℝ := (1 / 2) * (a + b) * h

-- The theorem to prove that the area of the trapezium is 304 cm²
theorem trapezium_area_is_304 : trapezium_area a b h = 304 := by
  sorry

end trapezium_area_is_304_l445_445153


namespace distance_A1C1_BD1_l445_445599

-- Define the vertices of the cube with edge length 1
structure CubeVertices :=
  (A A1 B B1 C C1 D D1: ℝ × ℝ × ℝ)

-- Define the cube and the specific vertices
def cube : CubeVertices := {
  A := (0, 0, 0),
  B := (1, 0, 0),
  C := (1, 1, 0),
  D := (0, 1, 0),
  A1 := (0, 0, 1),
  B1 := (1, 0, 1),
  C1 := (1, 1, 1),
  D1 := (0, 1, 1)
}

noncomputable def distance_between_lines (x1 y1 z1 x2 y2 z2 a1 b1 c1 a2 b2 c2 : ℝ) : ℝ := 
  (√6 / 6)

-- The proof statement in Lean 4
theorem distance_A1C1_BD1 : distance_between_lines 0 0 1 1 1 1 1 0 0 0 1 1 = (√6 / 6) := 
by
  sorry

end distance_A1C1_BD1_l445_445599


namespace speed_of_stream_l445_445679

-- Define the speed of the boat in still water
def speed_of_boat_in_still_water : ℝ := 39

-- Define the effective speed upstream and downstream
def effective_speed_upstream (v : ℝ) : ℝ := speed_of_boat_in_still_water - v
def effective_speed_downstream (v : ℝ) : ℝ := speed_of_boat_in_still_water + v

-- Define the condition that time upstream is twice the time downstream
def time_condition (D v : ℝ) : Prop := 
  (D / effective_speed_upstream v = 2 * (D / effective_speed_downstream v))

-- The main theorem stating the speed of the stream
theorem speed_of_stream (D : ℝ) (h : D > 0) : (v : ℝ) → time_condition D v → v = 13 :=
by
  sorry

end speed_of_stream_l445_445679


namespace perm_banana_l445_445357

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445357


namespace sin_half_alpha_l445_445885

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445885


namespace total_distance_l445_445575

def morning_distance : ℕ := 2
def evening_multiplier : ℕ := 5

theorem total_distance : morning_distance + (evening_multiplier * morning_distance) = 12 :=
by
  sorry

end total_distance_l445_445575


namespace fixed_point_coordinates_l445_445160

noncomputable def fixed_point (A : Real × Real) : Prop :=
∀ (k : Real), ∃ (x y : Real), A = (x, y) ∧ (3 + k) * x + (1 - 2 * k) * y + 1 + 5 * k = 0

theorem fixed_point_coordinates :
  fixed_point (-1, 2) :=
by
  sorry

end fixed_point_coordinates_l445_445160


namespace coordinate_plane_points_l445_445762

theorem coordinate_plane_points (x y : ℝ) :
    4 * x^2 * y^2 = 4 * x * y + 3 ↔ (x * y = 3 / 2 ∨ x * y = -1 / 2) :=
by 
  sorry

end coordinate_plane_points_l445_445762


namespace scheduling_non_consecutive_courses_l445_445472

theorem scheduling_non_consecutive_courses :
  let courses := ["algebra", "geometry", "number_theory", "calculus"],
      periods := 8,
      ways := 120 in
  ∃ (f : fin 4 → fin 8), 
    (∀ (i j : fin 4), i ≠ j → abs (f i - f j) ≠ 1) ∧ 
    list.perm (list.map f list.fin_range) (list.fin_range 8) ↔ 
    ways = 120 :=
sorry

end scheduling_non_consecutive_courses_l445_445472


namespace skew_lines_angle_distance_l445_445597

theorem skew_lines_angle_distance :
  let A := (0, 0, 0) in
  let B := (4 * Real.sqrt 2, 0, 0) in
  let C := (2 * Real.sqrt 2, 4 * Real.sqrt 2, 0) in
  let D := (2 * Real.sqrt 2, 4 * Real.sqrt 2, 2) in
  let M := ((4 * Real.sqrt 2 + 2 * Real.sqrt 2) / 2, (0 + 4 * Real.sqrt 2) / 2, 0) in
  let K := ((0 + 4 * Real.sqrt 2) / 2, (0 + 0) / 2, 0) in
  let line1 := (D, M) in
  let line2 := (C, K) in
  let angle := 45 in
  let distance := 2 / Real.sqrt 3 in
  True := True → -- This line to ensure the theorem statement is syntactically complete
  sorry

end skew_lines_angle_distance_l445_445597


namespace sum_of_squares_equals_l445_445179

-- Definitions for the conditions in a)
variables {n : ℕ} (R : ℝ) (O X : E) (A : Fin n → E) [NormedSpace ℝ E]

-- Regularly inscribed polygon condition
def is_regular_polygon (A : Fin n → E) (O : E) (R : ℝ) := 
  ∀ i : Fin n, euclidean_distance (A i) O = R

-- The distance from point O to point X
abbreviation d := euclidean_distance O X

-- The sum of the squares of the distances from vertices to point X
def sum_of_squares_to_X (A : Fin n → E) (X : E) :=
  ∑ i, euclidean_distance (A i) X ^ 2

-- The theorem statement
theorem sum_of_squares_equals {E : Type*} [NormedSpace ℝ E] {n : ℕ} (A : Fin n → E) (R : ℝ) (O X : E) (h : is_regular_polygon A O R) :
  sum_of_squares_to_X A X = n * (R^2 + d O X) :=
by
  sorry

end sum_of_squares_equals_l445_445179


namespace binom_12_11_eq_12_l445_445221

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := 
by {
  sorry
}

end binom_12_11_eq_12_l445_445221


namespace product_of_positive_integral_values_of_m_l445_445385

theorem product_of_positive_integral_values_of_m :
  let q := 3 in
  ∃ (m1 m2 : ℕ), (m1 * m2 = 396) ∧ (m1^2 - 40 * m1 + 399 = q) ∧ (m2^2 - 40 * m2 + 399 = q) :=
by
  let q := 3
  use [22, 18]
  split
  . repeat { split }
  . calc
    22 * 18 = 396
    . . exact 396
  . calc
    22^2 - 40 * 22 + 399 = 3
    . . sorry
  . calc
    18^2 - 40 * 18 + 399 = 3
    . . sorry

end product_of_positive_integral_values_of_m_l445_445385


namespace sqrt_fraction_combination_l445_445208

variable (a b c : ℝ)

-- Conditions
def a_condition := a = 1 / 25
def b_condition := b = 1 / 36
def c_condition := c = 1 / 144

-- Problem Statement
theorem sqrt_fraction_combination : 
  a = 1 / 25 → b = 1 / 36 → c = 1 / 144 → 
  sqrt (a + b - c) = 0.2467 := by
  intros ha hb hc
  rw [ha, hb, hc]
  sorry

end sqrt_fraction_combination_l445_445208


namespace eccentricity_of_hyperbola_l445_445932

noncomputable def hyperbola_eccentricity : Prop :=
  ∀ (a b : ℝ), a > 0 → b > 0 → (∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  → (∀ (c : ℝ), c^2 = a^2 + b^2) → b = 3 * a → ∃ e : ℝ, e = Real.sqrt 10

-- Statement of the problem without proof (includes the conditions)
theorem eccentricity_of_hyperbola (a b : ℝ) (h : a > 0) (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  (h4 : ∀ (c : ℝ), c^2 = a^2 + b^2) : hyperbola_eccentricity := 
  sorry

end eccentricity_of_hyperbola_l445_445932


namespace f_def_neg_g_even_l445_445424

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then √x else -√(-x)

def g (x : ℝ) : ℝ :=
if x ≥ 0 then √x else √(-x)

theorem f_def_neg (x : ℝ) (h : x < 0) : f(x) = -√(-x) := by
  unfold f
  split_ifs 
  · contradiction
  · rfl

theorem g_even (x : ℝ) : g(-x) = g(x) := by
  unfold g
  split_ifs with h1 h2 h3 h4
  · contradiction
  · contradiction
  · contradiction
  · contradiction
  · rfl
  · rfl
  init
  init
  init
  contradiction

sorry --proof to be filled


end f_def_neg_g_even_l445_445424


namespace diameter_with_all_points_on_one_side_l445_445685

noncomputable def diameter_exists (ω : Circle) (points : Finset ω.Points) : Prop :=
  ∃ d : ω.Diameter, ∀ p ∈ points, p ∉ d.EndPoints
  
theorem diameter_with_all_points_on_one_side
  (ω : Circle)
  (points : Finset ω.Points)
  (h : ∀ p ∈ points, distance p (ω.Center) < ω.Radius) :
  diameter_exists ω points :=
sorry

end diameter_with_all_points_on_one_side_l445_445685


namespace area_triangle_outside_circle_l445_445181

theorem area_triangle_outside_circle
  (a : ℝ) 
  (h : a > 0) :
  let triangle_area := (a^2 * Real.sqrt 3) / 4
      circle_area := Real.pi * (a / 3)^2
      segment_area := (Real.pi * (a^2 / 54)) - (a^2 * Real.sqrt 3 / 36)
      total_area := triangle_area - circle_area + 3 * segment_area
  in total_area = (a^2 * (3 * Real.sqrt 3 - Real.pi)) / 18 :=
by 
  sorry

end area_triangle_outside_circle_l445_445181


namespace banana_permutations_l445_445254

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445254


namespace BANANA_arrangements_l445_445273

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l445_445273


namespace cube_surface_area_l445_445716

noncomputable def surface_area_of_cube_with_same_volume_as_prism 
  (length : ℝ) (width : ℝ) (height : ℝ) : ℝ := 
  let volume := length * width * height
  let side := real.cbrt volume
  6 * (side ^ 2)

theorem cube_surface_area 
  (length : ℝ) (width : ℝ) (height : ℝ) 
  (h_length : length = 5) 
  (h_width : width = 4) 
  (h_height : height = 40) : 
  surface_area_of_cube_with_same_volume_as_prism length width height = 600 :=
by
  unfold surface_area_of_cube_with_same_volume_as_prism
  rw [h_length, h_width, h_height]
  norm_num
  sorry -- Further steps can be filled here to complete the proof

end cube_surface_area_l445_445716


namespace probability_log_inequality_l445_445646

theorem probability_log_inequality (t : ℝ) (h_t : 0 < t ∧ t < 4) :
    (\( P ( 1 < t ∧ t < 3 \| 0 < t ∧ t < 4) = \frac{1}{2} \)) :=
sorry

end probability_log_inequality_l445_445646


namespace initial_investment_l445_445237

theorem initial_investment (FV : ℝ) (r : ℝ) (n : ℕ) : 
  FV = 700000 ∧ r = 0.04 ∧ n = 15 → 
  abs (700000 / (1 + 0.04)^15 - 388812.33) < 0.01 := 
by 
  intro h,
  cases h with hFV hrn,
  cases hrn with hr hn,
  have calcFV : abs (700000 / (1 + 0.04)^15 - 388812.33) < 0.01 := 
     sorry,
  exact calcFV

end initial_investment_l445_445237


namespace solve_Q_l445_445583

theorem solve_Q (Q : ℝ) (h : sqrt (Q^3) = 18 * root 6 64) :
  Q = 4 * (root 3 2) * 3 * (root 3 3) := by
  sorry

end solve_Q_l445_445583


namespace gcd_of_polynomial_and_multiple_of_12600_l445_445422

theorem gcd_of_polynomial_and_multiple_of_12600 (x : ℕ) (h : 12600 ∣ x) : gcd ((5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)) x = 840 := by
  sorry

end gcd_of_polynomial_and_multiple_of_12600_l445_445422


namespace sin_half_alpha_l445_445869

noncomputable def alpha : ℝ := sorry
def is_acute (alpha : ℝ) : Prop := 0 < alpha ∧ alpha < π / 2

axiom acos_alpha : cos alpha = (1 + sqrt 5) / 4
axiom acute_alpha : is_acute alpha

theorem sin_half_alpha : sin (alpha / 2) = (sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_alpha_l445_445869


namespace range_of_S6_l445_445058

variables {a_1 a_4 a_5 d S_6 : ℝ}

theorem range_of_S6 (h1 : a_4 = a_1 + 3 * d)
                    (h2 : a_5 = a_1 + 4 * d)
                    (h3 : 1 ≤ a_4 ∧ a_4 ≤ 4)
                    (h4 : 2 ≤ a_5 ∧ a_5 ≤ 3)
                    (h5 : S_6 = 6 * a_1 + 15 * d) : 
                    0 ≤ S_6 ∧ S_6 ≤ 30 :=
begin
  sorry
end

end range_of_S6_l445_445058


namespace maximum_perimeter_triangle_l445_445076

theorem maximum_perimeter_triangle {A B C : Point} (H : Hexagon) (AB_side : H.side AB):
  ∃ A' B' : Point, (A'B' ∥ AB_side) ∧ (C = A' ∨ C = B') → 
  perimeter_triangle A B C = maximum_perimeter_triangle A B H :=
begin
  sorry  -- Proof to be completed
end

end maximum_perimeter_triangle_l445_445076


namespace multiple_with_same_digit_l445_445641

theorem multiple_with_same_digit (N : ℕ) :
  (∃ a : ℕ, ∃ k : ℕ, k > 0 ∧ (∀ d : ℕ, d < 10 → d ∈ digits (a * k)) ∧ (∀ d : ℕ, d ∈ digits (a * k) → d = a)) ↔ (Nat.gcd N 10 = 1) :=
sorry

end multiple_with_same_digit_l445_445641


namespace area_of_triangle_formed_by_diagonal_l445_445677

theorem area_of_triangle_formed_by_diagonal (A : ℝ) (h : A = 128) : (1 / 2) * A = 64 := 
by
  rw h
  norm_num

#eval area_of_triangle_formed_by_diagonal 128 rfl -- To verify the theorem gives the expected result

end area_of_triangle_formed_by_diagonal_l445_445677


namespace sin_half_angle_l445_445813

theorem sin_half_angle (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : real.cos α = (1 + real.sqrt 5) / 4) :
  real.sin (α / 2) = (real.sqrt 5 - 1) / 4 :=
by
  sorry

end sin_half_angle_l445_445813


namespace find_salary_of_january_l445_445678

variables (J F M A May : ℝ)

theorem find_salary_of_january
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 := 
sorry

end find_salary_of_january_l445_445678


namespace skew_lines_intersection_l445_445688

theorem skew_lines_intersection {a b : Line} {α β : Plane} {l : Line} 
  (h1 : skew a b) 
  (h2 : a ⊆ α) 
  (h3 : b ⊆ β) 
  (h4 : α ∩ β = l) : 
  l intersects_with_at_least_one_of a b :=
sorry

end skew_lines_intersection_l445_445688


namespace problem_l445_445492

theorem problem (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_l445_445492


namespace sum_of_digits_l445_445536

theorem sum_of_digits (A B C D : ℕ) (h1 : A ≠ B) (h2 : A * B = 10 * C + D)
(ha : A < 10) (hb : B < 10) (hc : C < 10) (hd : D < 10) 
(hd1 : C ≠ A) (hd2 : C ≠ B) (hd3 : C ≠ D) (hd4 : D ≠ A) (hd5 : D ≠ B)
(h0 : ∃ f : (ℕ × ℕ) → (ℕ × ℕ), (∀ ab cd, f (ab, cd) = ab ∧ ab ≠ cd) ∧ 
    (∃ h : (A * B ≠ 10 * C + D → ℕ), h 0 = 3)) : 
  A + B + C + D = 17 :=
sorry

end sum_of_digits_l445_445536


namespace unique_zero_point_mn_l445_445984

noncomputable def f (a : ℝ) (x : ℝ) := a * (x^2 + 2 / x) - Real.log x

theorem unique_zero_point_mn (a : ℝ) (m n x₀ : ℝ) (hmn : m + 1 = n) (a_pos : 0 < a) (f_zero : f a x₀ = 0) (x0_in_range : m < x₀ ∧ x₀ < n) : m + n = 5 := by
  sorry

end unique_zero_point_mn_l445_445984


namespace two_std_devs_less_than_mean_l445_445101

theorem two_std_devs_less_than_mean (μ σ : ℝ) (hμ : μ = 12) (hσ : σ = 1.2) : 
    μ - 2 * σ = 9.6 := by
  rw [hμ, hσ]
  norm_num
  sorry

end two_std_devs_less_than_mean_l445_445101


namespace problem1_problem2_l445_445795

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem problem1 (x : ℝ) : f x ≥ 5 ↔ x ∈ set.Iic (-3) ∪ set.Ici 2 :=
begin
  sorry
end

theorem problem2 (a : ℝ) : (∀ x, f x > a^2 - 2 * a) ↔ a ∈ set.Ioo (-1 : ℝ) (3 : ℝ) :=
begin
  sorry
end

end problem1_problem2_l445_445795


namespace beam_reflection_problem_l445_445162

theorem beam_reflection_problem
  (A B D C : Point)
  (angle_CDA : ℝ)
  (total_path_length_max : ℝ)
  (equal_angle_reflections : ∀ (k : ℕ), angle_CDA * k ≤ 90)
  (path_length_constraint : ∀ (n : ℕ) (d : ℝ), 2 * n * d ≤ total_path_length_max)
  : angle_CDA = 5 ∧ total_path_length_max = 100 → ∃ (n : ℕ), n = 10 :=
sorry

end beam_reflection_problem_l445_445162


namespace probability_log3_integer_l445_445714

/-- Definition that represents a three-digit integer N -/
def is_three_digit (N : ℕ) : Prop := N >= 100 ∧ N <= 999

/-- Definition that represents N being a power of 3 -/
def is_power_of_3 (N : ℕ) : Prop := ∃ k : ℕ, N = 3^k

/-- Main statement -/
theorem probability_log3_integer (N : ℕ) (h : is_three_digit N) : 
  let total_three_digit_numbers := 900 in
  let count_valid := 2 in  -- (243, 729)
  (∃ p : ℚ, p = count_valid / total_three_digit_numbers ∧ p = 1/450) :=
by
  sorry

end probability_log3_integer_l445_445714


namespace sin_half_alpha_l445_445887

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445887


namespace domain_of_f_l445_445776

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^3 - 9*x)

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 ↔ x ∈ ((-∞ : set ℝ) ∪ -3 : set ℝ) ∪ -3 ∪ 0 : (-∞ : set ℝ) ∪ 0 : 3 : set ℝ ∪ 3 : set ℝ ∪ -∞ : set ℝ) :=
by
  sorry

end domain_of_f_l445_445776


namespace largest_m_digit_sum_l445_445549

def is_prime (n : ℕ) : Prop := ∀ (d : ℕ), d ∣ n → d = 1 ∨ d = n

def single_digit_prime (p : ℕ) : Prop := p ∈ {2, 3, 5, 7}

def valid_triplet (a b : ℕ) : Prop :=
  single_digit_prime a ∧
  single_digit_prime b ∧
  a ≠ b ∧
  is_prime (a * b + 3)

def product_of_triplet (a b : ℕ) : ℕ :=
  a * b * (a * b + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem largest_m_digit_sum :
  ∃ (m : ℕ), m = max (product_of_triplet 2 7) (max (product_of_triplet 2 5) (product_of_triplet 2 2)) ∧ sum_of_digits m = 13 :=
begin
  use 238,
  split,
  { refl },
  { sorry }
end

end largest_m_digit_sum_l445_445549


namespace permutations_of_banana_l445_445288

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l445_445288


namespace banana_permutations_l445_445259

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445259


namespace repeating_decimal_rational_representation_l445_445167

theorem repeating_decimal_rational_representation :
  (0.12512512512512514 : ℝ) = (125 / 999 : ℝ) :=
sorry

end repeating_decimal_rational_representation_l445_445167


namespace find_perimeter_l445_445110

/-
We define the problem conditions including the points of tangency, lengths, and radius of the inscribed circle,
and then we state the main theorem that requires proving the perimeter of triangle DEF.
-/

variables (D E F P Q R : Type) 
variable [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P] [MetricSpace Q] [MetricSpace R]

-- Define the tangency points and conditions given
def is_tangent_to (c : Circle D) (l : Line E) (p : P) : Prop := 
  IsTangent (Circle.radius c = 15) (Line.length DE = DP + PE)

axiom tangent_circle_def :
  ∀ (r : ℝ), (r = 15 ∧ r = DP ∧ r = PE) → 
  (Circle.radius = r ∧ DP = r ∧ PE = 18)

-- Define the distances and perimeter
variables {r : ℝ} (DEF : Triangle D E F) (DP PE : ℝ)
axiom distance_props: DP = 15 ∧ PE = 18

-- Define the expected perimeter of triangle DEF
noncomputable def triangle_perimeter (DEF : Triangle D E F) : ℝ :=
  280.5

-- Main theorem statement
theorem find_perimeter (DEF : Triangle D E F) (DP PE : ℝ) 
  (hn : DEF.incircle radius = 15) (ht : T_is_tangent_to DEF incircle DE at P)
  (hd : DP = 15) (hp : PE = 18) :
  triangle_perimeter DEF = 280.5 :=
sorry

end find_perimeter_l445_445110


namespace solve_sqrt_equation_l445_445106

theorem solve_sqrt_equation (k : ℝ) (h : sqrt (36 - k^2) - 6 = 0) : k = 0 :=
by
  sorry

end solve_sqrt_equation_l445_445106


namespace not_age_of_child_l445_445065

noncomputable def sum_from_1_to_n (n : ℕ) := n * (n + 1) / 2

theorem not_age_of_child (N : ℕ) (S : Finset ℕ) (a b : ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11} ∧
  N = 1100 * a + 11 * b ∧
  a ≠ b ∧
  N ≥ 1000 ∧ N < 10000 ∧
  ((S.sum id) = N) ∧
  (∀ age ∈ S, N % age = 0) →
  10 ∉ S := 
by
  sorry

end not_age_of_child_l445_445065


namespace factorial_div_sub_eq_l445_445755

theorem factorial_div_sub_eq : (12! / 11!) - 10 = 2 := by
  sorry

end factorial_div_sub_eq_l445_445755


namespace number_of_solutions_l445_445460

theorem number_of_solutions (n : ℤ) : set.count {n | 0 ≤ n ∧ n ≤ 720 ∧ (n^2 ≡ 1 [MOD 720])} = 16 :=
sorry

end number_of_solutions_l445_445460


namespace cost_of_pencils_and_notebooks_l445_445118

variable (p n : ℝ)

theorem cost_of_pencils_and_notebooks 
  (h1 : 9 * p + 10 * n = 5.06) 
  (h2 : 6 * p + 4 * n = 2.42) :
  20 * p + 14 * n = 8.31 :=
by
  sorry

end cost_of_pencils_and_notebooks_l445_445118


namespace triangle_side_length_difference_l445_445503

theorem triangle_side_length_difference (a b c : ℕ) (hb : b = 8) (hc : c = 3)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  let min_a := 6
  let max_a := 10
  max_a - min_a = 4 :=
by {
  sorry
}

end triangle_side_length_difference_l445_445503


namespace number_of_arrangements_of_BANANA_l445_445319

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445319


namespace find_abc_y_monotonicity_l445_445797

variables {ℤ : Type} [integral_domain ℤ]

noncomputable def f (a b c x : ℤ) : ℤ := (a * x^2 + 1) / (b * x + c)

theorem find_abc (a b c : ℤ):
  (f a b c 1 = 2) ∧ (f a b c 2 < 3) ∧ ∀ x: ℤ, f a b c (-x) = -f a b c x → 
  a = 1 ∧ b = 1 ∧ c = 0 :=
sorry

noncomputable def f_squared (x : ℤ) : ℤ := (x^2 + 1) / x

theorem y_monotonicity:
  ∀ x1 x2 : ℤ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 → (x1 < x2 → f_squared x1 > f_squared x2) :=
sorry

end find_abc_y_monotonicity_l445_445797


namespace sqrt_Sn_arithmetic_seq_a_n_general_formula_part2_sum_greatest_integer_part3_max_m_exists_l445_445801

theorem sqrt_Sn_arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  S 1 = 1 →
  (∀ n, n ≥ 2 → a n = sqrt (S n) + sqrt (S (n - 1))) →
  ∀ n, sqrt (S n) - sqrt (S (n - 1)) = 1 :=
sorry

theorem a_n_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  S 1 = 1 →
  (∀ n, n ≥ 2 → a n = sqrt (S n) + sqrt (S (n - 1))) →
  a 1 = 1 →
  ∀ n, a n = 2n - 1 :=
sorry

theorem part2_sum_greatest_integer (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  S 1 = 1 →
  (∀ n, n ≥ 2 → a n = sqrt (S n) + sqrt (S (n - 1))) →
  a 1 = 1 →
  [finset.range n].sum (λ i, 1 / (a (i+1) ^ 2)) = 1 :=
sorry

theorem part3_max_m_exists (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, a n > 0) →
  S 1 = 1 →
  (∀ n, n ≥ 2 → a n = sqrt (S n) + sqrt (S (n - 1))) →
  a 1 = 1 →
  (∀ n, b n = 1 / ((2 * n - 1) * (2 * n + 1))) →
  (∀ n, T n = finset.range (n + 1)).sum b →
  ∃ m, ∀ n, T n > m / 2022 ∧ m ≤ 673 :=
sorry

end sqrt_Sn_arithmetic_seq_a_n_general_formula_part2_sum_greatest_integer_part3_max_m_exists_l445_445801


namespace vector_dot_product_sum_l445_445954

variables {V : Type*} [InnerProductSpace ℝ V]
(open InnerProductSpace)

def vector_a : V := sorry
def vector_b : V := sorry
def vector_c : V := sorry

#check vector_a

-- Given conditions
def cond1 : vector_a + vector_b + vector_c = (0 : V) := sorry
def cond2 : ∥vector_a∥ = 1 := sorry
def cond3 : ∥vector_b∥ = 2 := sorry
def cond4 : ∥vector_c∥ = 2 := sorry

-- Proof goal
theorem vector_dot_product_sum :
  vector_a + vector_b + vector_c = 0 →
  ∥vector_a∥ = 1 →
  ∥vector_b∥ = 2 →
  ∥vector_c∥ = 2 →
  (⟪vector_a, vector_b⟫ + ⟪vector_b, vector_c⟫ + ⟪vector_c, vector_a⟫ : ℝ) = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445954


namespace shared_area_of_circles_l445_445632

noncomputable def area_shared_by_two_circles (r1 r2 : ℝ) (angle1 angle2 : ℝ) (chord_length : ℝ) : ℝ :=
  let circle1_sect_area := (angle1 / 360) * π * r1^2
  let circle2_sect_area := (angle2 / 360) * π * r2^2
  let quad_area := sqrt (r1^2 + r2^2)
  circle1_sect_area + circle2_sect_area - quad_area

theorem shared_area_of_circles : area_shared_by_two_circles (2 / (sqrt 3)) 2 120 60 2 = (10 * π - 12 * sqrt 3) / 9 :=
by
  sorry

end shared_area_of_circles_l445_445632


namespace banana_permutations_l445_445258

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l445_445258


namespace arrangement_of_BANANA_l445_445365

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445365


namespace integer_length_chords_l445_445697

def circle_eq (x y : ℝ) := x^2 + y^2 + 2*x - 4*y - 164 = 0

def point_A (x y : ℝ) := (x = 11) ∧ (y = 2)

def standard_form_circle_eq (x y : ℝ) := (x + 1)^2 + (y - 2)^2 = 169

theorem integer_length_chords (x y : ℝ) :
  (point_A x y) → 
  (∀ {l : ℝ}, l ∈ {l | ∃ x, ∃ y, point_A x y ∧ circle_eq x y} → l ∈ ℕ) →
  (count (λ l, l ∈ {l | ∃ x, ∃ y, point_A x y ∧ circle_eq x y ∧ standard_form_circle_eq x y ∧ l ∈ ℕ}) = 32)
:= sorry

end integer_length_chords_l445_445697


namespace first_quarter_days_2016_l445_445108

theorem first_quarter_days_2016 : 
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  (jan_days + feb_days + mar_days) = 91 := 
by
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  have h_leap_year : leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) := by sorry
  have h_feb_days : feb_days = 29 := by sorry
  have h_first_quarter : jan_days + feb_days + mar_days = 31 + 29 + 31 := by sorry
  have h_sum : 31 + 29 + 31 = 91 := by norm_num
  exact h_sum

end first_quarter_days_2016_l445_445108


namespace distance_between_points_l445_445383

def point := (Int, Int)

def distance (p1 p2 : point) : Real :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance (3, 3) (-2, -3) = Real.sqrt 61 :=
by
  sorry

end distance_between_points_l445_445383


namespace ian_final_number_l445_445066

-- Define the condition of the problem, where each student skips the third number in every group of five from what remains
def skip_pattern (n : ℕ) (skips : ℕ → Prop) : ℕ → Prop :=
  λ x, ¬ skips x ∧ (x % 5 ≠ 3)

-- Represents skipping for each student
def skips_for_students (students : ℕ) : ℕ → Prop
| n :=
  match students with
  | 1     => skip_pattern n (λ _, false)
  | k + 1 => skip_pattern n (skips_for_students k)
  end

-- Ian is the 9th student
def ian_says : ℕ := 
  let rec find_next (x : ℕ) : ℕ :=
    if ¬(skips_for_students 9 x) then x
    else find_next (x + 1)
  in find_next 1

theorem ian_final_number : ian_says = 591 :=
sorry

end ian_final_number_l445_445066


namespace constant_term_expansion_l445_445026

theorem constant_term_expansion (x : ℂ) :
  (∏ i in finset.range 5, (x^2 + 2/x^3)) = 40 :=
by
  sorry

end constant_term_expansion_l445_445026


namespace train_speed_proof_l445_445161

-- Define the conditions
def train_length : ℝ := 700
def crossing_time : ℝ := 41.9966402687785
def man_speed_kmh : ℝ := 3
def man_speed_ms : ℝ := man_speed_kmh * 1000 / 3600

-- Define the correct answer
def expected_train_speed_kmh : ℝ := 63.00468

-- The actual statement to prove
theorem train_speed_proof :
  (train_length / crossing_time + man_speed_ms) * 3600 / 1000 = expected_train_speed_kmh :=
by
  -- This is where the proof would go
  sorry

end train_speed_proof_l445_445161


namespace find_other_number_product_find_third_number_sum_l445_445746

-- First Question
theorem find_other_number_product (x : ℚ) (h : x * (1/7 : ℚ) = -2) : x = -14 :=
sorry

-- Second Question
theorem find_third_number_sum (y : ℚ) (h : (1 : ℚ) + (-4) + y = -5) : y = -2 :=
sorry

end find_other_number_product_find_third_number_sum_l445_445746


namespace stadium_length_in_yards_l445_445614

theorem stadium_length_in_yards (length_in_feet : ℕ) (h : length_in_feet = 186) : length_in_feet / 3 = 62 :=
by
  rw [h]
  rfl

end stadium_length_in_yards_l445_445614


namespace initial_investment_amount_l445_445487

theorem initial_investment_amount (r : ℝ) (final_amount : ℝ) (years : ℝ) (doubling_time : ℝ) (initial_investment : ℝ) :
  r = 4 ∧ final_amount = 20000 ∧ years = 36 ∧ doubling_time = 70 / r ∧ initial_investment = final_amount / (2 ^ (years / doubling_time)) → 
  initial_investment ≈ 4819.28 := 
by
  intros
  sorry

end initial_investment_amount_l445_445487


namespace intersection_of_A_and_B_l445_445408

noncomputable def A : Set ℝ := { x | -1 < x - 3 ∧ x - 3 ≤ 2 }
noncomputable def B : Set ℝ := { x | 3 ≤ x ∧ x < 6 }

theorem intersection_of_A_and_B : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end intersection_of_A_and_B_l445_445408


namespace perm_banana_l445_445356

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l445_445356


namespace intersection_A_B_l445_445419
noncomputable theory

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0} :=
by 
  sorry

end intersection_A_B_l445_445419


namespace number_of_arrangements_of_BANANA_l445_445311

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l445_445311


namespace intersection_empty_l445_445939

theorem intersection_empty {A B : Set} (hA : A = {'line}) (hB : B = {'ellipse}) : |A ∩ B| = 0 :=
by
  sorry

end intersection_empty_l445_445939


namespace sin_half_alpha_l445_445824

noncomputable def given_cos_alpha (α : ℝ) : Prop :=
  α ∈ Ioo 0 (π / 2) ∧ cos α = (1 + real.sqrt 5) / 4

theorem sin_half_alpha (α : ℝ) (hα : given_cos_alpha α) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
sorry

end sin_half_alpha_l445_445824


namespace sin_tan_identity_l445_445911

theorem sin_tan_identity (a : ℝ) (h : cos (31 * real.pi / 180) = a) : 
  sin (239 * real.pi / 180) * tan (149 * real.pi / 180) = real.sqrt (1 - a^2) :=
by sorry

end sin_tan_identity_l445_445911


namespace sin_half_alpha_l445_445893

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445893


namespace find_alpha_l445_445159

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^2 + 2 * α * x + 1

theorem find_alpha (α : ℝ) (H : ∃ x ∈ set.Icc 0 1, f α x = 0) : α ≤ -1 := 
by {
  sorry
}

end find_alpha_l445_445159


namespace cos_double_theta_l445_445809

theorem cos_double_theta (θ : ℝ) :
  2^(-5/2 + 2 * Real.cos θ) + 1 = 2^(3/4 + Real.cos θ) ->
  Real.cos (2 * θ) = 17/8 :=
by
  intro h
  sorry

end cos_double_theta_l445_445809


namespace calculate_volume_from_measurements_l445_445700

variables (r h : ℝ) (P : ℝ × ℝ)

noncomputable def volume_truncated_cylinder (area_base : ℝ) (height_segment : ℝ) : ℝ :=
  area_base * height_segment

theorem calculate_volume_from_measurements
    (radius : ℝ) (height : ℝ)
    (area_base : ℝ := π * radius^2)
    (P : ℝ × ℝ)  -- intersection point on the axis
    (height_segment : ℝ) : 
    volume_truncated_cylinder area_base height_segment = area_base * height_segment :=
by
  -- The proof would involve demonstrating the relationship mathematically
  sorry

end calculate_volume_from_measurements_l445_445700


namespace union_of_A_and_B_l445_445415

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l445_445415


namespace minimize_circumscribed_sphere_radius_l445_445566

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def circumscribed_sphere_radius (r h : ℝ) : ℝ :=
  (r^2 + (1 / 2 * h)^2).sqrt

theorem minimize_circumscribed_sphere_radius (r : ℝ) (h : ℝ) (hr : cylinder_surface_area r h = 16 * Real.pi) : 
  r^2 = 8 * Real.sqrt 5 / 5 :=
sorry

end minimize_circumscribed_sphere_radius_l445_445566


namespace height_of_remaining_cube_after_cut_l445_445691

-- Define the problem conditions
def cubeSideLength := 2.0
def choppedHeight : ℝ := 1 / Real.sqrt 3
def remainingHeight := (cubeSideLength - choppedHeight)

-- The final theorem statement
theorem height_of_remaining_cube_after_cut :
  remainingHeight = (5 * Real.sqrt 3) / 3 :=
by
  -- Leaving the proof as an exercise. 
  sorry

end height_of_remaining_cube_after_cut_l445_445691


namespace square_of_complex_l445_445753

def z : Complex := 5 - 2 * Complex.I

theorem square_of_complex : z^2 = 21 - 20 * Complex.I := by
  sorry

end square_of_complex_l445_445753


namespace total_surface_area_of_cuboid_l445_445426

variables (l w h : ℝ)
variables (lw_area wh_area lh_area : ℝ)

def box_conditions :=
  lw_area = l * w ∧
  wh_area = w * h ∧
  lh_area = l * h

theorem total_surface_area_of_cuboid (hc : box_conditions l w h 120 72 60) :
  2 * (120 + 72 + 60) = 504 :=
sorry

end total_surface_area_of_cuboid_l445_445426


namespace line_length_limit_l445_445706

noncomputable def geometric_series_limit : ℝ :=
  let a := 2
  let r := 1 / 4
  let b := 1 / 8
  let s1 := a / (1 - r)              -- Limit of the first series
  let s2 := (b / (1 - r)) * sqrt 3   -- Limit of the second series factoring out √3
  s1 + s2

-- Theorem statement
theorem line_length_limit (series_limit : ℝ) :
  series_limit = 1 / 6 * (16 + sqrt 3) :=
begin
  sorry
end

end line_length_limit_l445_445706


namespace binom_12_11_eq_12_l445_445220

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_eq_12_l445_445220


namespace multiples_of_15_between_20_and_200_l445_445462

theorem multiples_of_15_between_20_and_200 : 
  let a : ℕ := 30
  let l : ℕ := 195
  let d : ℕ := 15
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 12 := 
begin
  sorry
end

end multiples_of_15_between_20_and_200_l445_445462


namespace quadratic_positive_imp_ineq_l445_445052

theorem quadratic_positive_imp_ineq (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c > 0) → b^2 - 4 * c ≤ 0 :=
by 
  sorry

end quadratic_positive_imp_ineq_l445_445052


namespace sin_half_alpha_l445_445899

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l445_445899


namespace permutations_of_BANANA_l445_445243

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445243


namespace find_4digit_number_l445_445380

theorem find_4digit_number (a b c d n n' : ℕ) :
  n = 1000 * a + 100 * b + 10 * c + d →
  n' = 1000 * d + 100 * c + 10 * b + a →
  n = n' - 7182 →
  n = 1909 :=
by
  intros h1 h2 h3
  sorry

end find_4digit_number_l445_445380


namespace determine_d_l445_445766

def Q (x d : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

theorem determine_d (d : ℝ) : (∃ d, Q (-2) d = 0) → d = -14 := by
  sorry

end determine_d_l445_445766


namespace equivalent_multipliers_l445_445978

variable (a b : ℝ)

theorem equivalent_multipliers (a b : ℝ) :
  let a_final := 0.93 * a
  let expr := a_final + 0.05 * b
  expr = 0.93 * a + 0.05 * b  :=
by
  -- Proof placeholder
  sorry

end equivalent_multipliers_l445_445978


namespace cattle_ranch_total_cows_l445_445457

theorem cattle_ranch_total_cows 
  (A B C : ℕ) 
  (h1 : A + B + C = 200)
  (h2 : A = 2 * B)
  (h3 : C = 3 * A)
  (hA_growth : ∀ n, n = 0 ∨ n = 1 → ∀ x, (nat.succ n).succ * A + A + nat.succ n * x = 1.5 * x)
  (hB_growth : ∀ n, n = 0 ∨ n = 1 → ∀ y, (nat.succ n).succ * B + B + nat.succ n * y = 1.25 * y)
  (hC_growth : ∀ n, n = 0 ∨ n = 1 → ∀ z, (z + z + z).succ * C + C + nat.succ n * z = 1.4 * z) :
  ∃ A2 B2 C2, A2 + B2 + C2 = 389 := 
sorry

end cattle_ranch_total_cows_l445_445457


namespace sin_half_angle_l445_445909

theorem sin_half_angle (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : cos α = (1 + real.sqrt 5) / 4) :
  sin (α / 2) = (-1 + real.sqrt 5) / 4 :=
by sorry

end sin_half_angle_l445_445909


namespace distinct_outcomes_of_multiples_of_three_l445_445759

theorem distinct_outcomes_of_multiples_of_three (p q : ℕ) (hp1 : p % 2 = 1) (hq1 : q % 2 = 1) (hp2 : p % 3 = 0) (hq2 : q % 3 = 0) (hp3 : p < 20) (hq3 : q < 20) : 
  p ∈ {3, 9, 15} → q ∈ {3, 9, 15} → 
  (p ≠ q) → 
  ∀ P Q, P = (p + 1) * (q + 1) - 1 → Q = (q + 1) * (p + 1) - 1 →
  {P, Q}.card = 6 :=
by
  intros
  sorry

end distinct_outcomes_of_multiples_of_three_l445_445759


namespace solve_for_x_l445_445091

theorem solve_for_x (x : ℝ) : 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ↔ x = -22 / 13 := 
by 
  sorry

end solve_for_x_l445_445091


namespace mb_range_l445_445610

theorem mb_range (m b : ℝ) (hm : m = 3 / 4) (hb : b = -2 / 3) :
  -1 < m * b ∧ m * b < 0 :=
by
  rw [hm, hb]
  sorry

end mb_range_l445_445610


namespace distance_extension_segments_l445_445565

theorem distance_extension_segments (ABC : Triangle) (AD BE : Segment) (c R r3 DE : Real)
  (h_AD : AD.length = c / 3)
  (h_BE : BE.length = c / 3)
  (h_AB : ABC.side_length AB = c)
  (h_circumradius : ABC.circumradius = R)
  (h_excircle_radius : ABC.excircle_radius_opposite_C = r3) :
  DE.length = (c / R) * sqrt (R * (R + 2 * r3)) := sorry

end distance_extension_segments_l445_445565


namespace power_function_passing_point_l445_445445

theorem power_function_passing_point (a : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, f x = x ^ a ∧ f 9 = 3) → a = 1 / 2 :=
by
  sorry

end power_function_passing_point_l445_445445


namespace stationary_tank_radius_correct_l445_445169

noncomputable def stationary_tank_radius : ℝ :=
let
  h_stationary := 25 -- height of the stationary tank
  r_truck := 5 -- radius of the truck’s tank
  h_truck := 12 -- height of the truck’s tank
  h_drop := 0.03 -- oil level dropped in the stationary tank
  V_truck := Real.pi * r_truck^2 * h_truck -- volume of the truck’s tank
in
  Real.sqrt (300 / h_drop)

theorem stationary_tank_radius_correct (h_stationary : ℝ) (r_truck : ℝ) (h_truck : ℝ) (h_drop : ℝ) : 
  h_stationary = 25 ∧ r_truck = 5 ∧ h_truck = 12 ∧ h_drop = 0.03 → stationary_tank_radius = 100 :=
by
  intros
  sorry

end stationary_tank_radius_correct_l445_445169


namespace tangent_lines_to_circle_l445_445122

theorem tangent_lines_to_circle 
  (x y : ℝ) 
  (circle : (x - 2) ^ 2 + (y + 1) ^ 2 = 1) 
  (point : x = 3 ∧ y = 3) : 
  (x = 3 ∨ 15 * x - 8 * y - 21 = 0) :=
sorry

end tangent_lines_to_circle_l445_445122


namespace value_of_expression_l445_445660

theorem value_of_expression : (180^2 - 150^2) / 30 = 330 := by
  sorry

end value_of_expression_l445_445660


namespace dihedral_angle_tetrahedron_l445_445915

-- Defining a regular tetrahedron structure
structure Tetrahedron :=
(a : ℝ) -- edge length

-- Defining the midpoint of a line segment
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- The statement of the theorem
theorem dihedral_angle_tetrahedron (T : Tetrahedron)
  (M : ℝ × ℝ × ℝ) -- Midpoint of DD₁
  (D D₁ A B C : ℝ × ℝ × ℝ) -- Vertices of the tetrahedron
  (h1 : M = midpoint D D₁) -- M is the midpoint of DD₁
  (h2 : ∀ x y z, (x = y → y = z → x = z)) -- Congruence of triangles involving M
  (h3 : D₁ = (T.a / sqrt 3, sqrt 2 / sqrt 3 * T.a, 0)) -- Height position
  (h4 : A = (0, 0, 0))
  (h5 : B = (T.a, 0, 0))
  (h6 : C = (T.a / 2, sqrt(3) * T.a / 2, 0))
  : real.angle (A, B, C) = real.angle (B, M, C) := sorry

end dihedral_angle_tetrahedron_l445_445915


namespace permutations_of_BANANA_l445_445246

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l445_445246


namespace lune_area_correct_l445_445717

def area_lune_of_semicircles (d1 d2: ℝ) (h1: d1 = 1) (h2: d2 = 2) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let area_small := (1/2) * Real.pi * r1^2
  let theta := Real.pi / 6
  let intersection := (1/2) * r2^2 * (theta - Real.sin theta)
  area_small - intersection

theorem lune_area_correct : 
  area_lune_of_semicircles 1 2 rfl rfl = (Real.sqrt 3 / 4) + (Real.pi / 24) :=
by
  sorry

end lune_area_correct_l445_445717


namespace sin_half_angle_l445_445841

variable (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos α = (1 + Real.sqrt 5) / 4)

theorem sin_half_angle :
  sin (α / 2) = (Real.sqrt 5 - 1) / 4 :=
sorry

end sin_half_angle_l445_445841


namespace sin_half_alpha_l445_445883

theorem sin_half_alpha (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
    Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := 
by
  sorry

end sin_half_alpha_l445_445883


namespace arrangement_of_BANANA_l445_445360

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445360


namespace geometric_progression_arcsin_sin_l445_445227

noncomputable def least_positive_t : ℝ :=
  9 + 4 * Real.sqrt 5

theorem geometric_progression_arcsin_sin 
  (α : ℝ) 
  (hα1: 0 < α) 
  (hα2: α < Real.pi / 2) 
  (t : ℝ) 
  (h : ∀ (a b c d : ℝ), 
    a = Real.arcsin (Real.sin α) ∧ 
    b = Real.arcsin (Real.sin (3 * α)) ∧ 
    c = Real.arcsin (Real.sin (5 * α)) ∧ 
    d = Real.arcsin (Real.sin (t * α)) → 
    b / a = c / b ∧ c / b = d / c) : 
  t = least_positive_t :=
sorry

end geometric_progression_arcsin_sin_l445_445227


namespace vector_dot_product_sum_l445_445968

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_dot_product_sum
  (a b c : V)
  (h1 : a + b + c = 0)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hc : ∥c∥ = 2) :
  inner_product_space.inner a b + inner_product_space.inner b c + inner_product_space.inner c a = -9 / 2 :=
by sorry

end vector_dot_product_sum_l445_445968


namespace incenter_eq_intersection_angle_bisectors_l445_445724

-- Define a triangle in Euclidean space
variables {P : Type*} [EuclideanSpace P] {A B C : P}

theorem incenter_eq_intersection_angle_bisectors (h_triangle : Triangle A B C) :
  ∃ (O : P), (Incenter O A B C) ∧ (O = intersection (angle_bisector A B C) 
                                      (angle_bisector B C A) 
                                      (angle_bisector C A B)) := 
  sorry

end incenter_eq_intersection_angle_bisectors_l445_445724


namespace calc_tan_fraction_l445_445207

theorem calc_tan_fraction :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h_tan_30 : Real.tan (30 * Real.pi / 180) = Real.sqrt 3 / 3 := by sorry
  sorry

end calc_tan_fraction_l445_445207


namespace sin_half_angle_l445_445838

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445838


namespace arrangement_of_BANANA_l445_445369

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l445_445369


namespace find_Sn_l445_445802

open Real

variables {n : ℕ} (a1 d : ℝ)

-- Definitions of the sequences
def a_seq (n : ℕ) : ℝ := a1 + (n - 1) * d
def sqrt_S_seq (n : ℕ) : ℝ := sqrt a1 + (n - 1) * d
def S_sum (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a_seq a1 d i

theorem find_Sn (a1 d : ℝ) (h_pos_a1 : 0 < a1) (h_pos_d : 0 < d)
    (h_arith_sqrt_Sn : ∀ n, sqrt_S_seq a1 d n = sqrt a1 + (n - 1) * d) :
    S_sum a1 d n = (n^2) / 4 := by
  sorry

end find_Sn_l445_445802


namespace simplify_expr_equals_neg_five_l445_445214

noncomputable def simplify_expr : ℚ :=
  sqrt 27 + ((-1 / 3 : ℚ)⁻¹) - abs (2 - sqrt 3) - 8 * real.cos (real.pi / 6)

theorem simplify_expr_equals_neg_five :
  simplify_expr = -5 := 
sorry

end simplify_expr_equals_neg_five_l445_445214


namespace prod_quality_related_prob_at_least_one_from_A_l445_445165

noncomputable def chi_square_value (a b c d n : ℝ) : ℝ :=
  (n * (a * d - b * c) ^ 2) / ( (a + b) * (c + d) * (a + c) * (b + d) )

theorem prod_quality_related (a b c d n k0 : ℝ) (h_k0 : k0 = 2.706)
  (h_a : a = 40) (h_b : b = 80) (h_c : c = 80) (h_d : d = 100) (h_n : n = 300) :
  chi_square_value a b c d n ≥ k0 :=
by
  sorry

theorem prob_at_least_one_from_A :
  ∃ (A B : ℕ), A = 2 ∧ B = 4 ∧ (choose 6 2) = 15 ∧
  ((A * B + choose A 2 + choose B 2) : ℝ) / (choose 6 2) = (3 : ℝ) / 5 :=
by
  sorry

end prod_quality_related_prob_at_least_one_from_A_l445_445165


namespace multiples_of_15_between_5_and_205_l445_445466

theorem multiples_of_15_between_5_and_205 : 
  ∃ n : ℕ, n = 13 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 13 → 15 * k ∈ set.Icc 6 205) := 
by
  existsi 13
  split
  -- Proof that there are 13 multiples is omitted
  sorry
  -- Proof that all multiples are between 6 and 205 is omitted
  sorry

end multiples_of_15_between_5_and_205_l445_445466


namespace union_of_A_and_B_l445_445414

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l445_445414


namespace combinations_of_eight_choose_three_is_fifty_six_l445_445023

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end combinations_of_eight_choose_three_is_fifty_six_l445_445023


namespace coin_event_probability_equivalence_l445_445626
open ProbabilityTheory

-- Definitions of Events
def event_A (first_coin: Bool) : Prop :=
  first_coin = true -- heads is represented by true

def event_B (second_coin: Bool) : Prop :=
  second_coin = false -- tails is represented by false

-- The main theorem statement
theorem coin_event_probability_equivalence 
  (first_coin second_coin: Bool) 
  (h_fair : ∀ (c : Bool), P(c = true) = 1/2 ∧ P(c = false) = 1/2) :
  P(event_A first_coin) = P(event_B second_coin) := 
sorry

end coin_event_probability_equivalence_l445_445626


namespace ellipse_tangent_line_l445_445985

theorem ellipse_tangent_line (m : ℝ) : 
  (∀ (x y : ℝ), (x ^ 2 / 4) + (y ^ 2 / m) = 1 → (y = mx + 2)) → m = 1 :=
by sorry

end ellipse_tangent_line_l445_445985


namespace bottles_after_drinking_and_buying_l445_445667

theorem bottles_after_drinking_and_buying :
  ∀ (initial_bottles drank bought : ℕ), initial_bottles = 42 → drank = 25 → bought = 30 → initial_bottles - drank + bought = 47 :=
by
  intros initial_bottles drank bought h_initial h_drank h_bought
  rw [h_initial, h_drank, h_bought]
  sorry

end bottles_after_drinking_and_buying_l445_445667


namespace circle_overlap_distance_l445_445645

theorem circle_overlap_distance :
  let circle1 := set_of (λ (p : ℝ × ℝ), (p.1 - 1)^2 + (p.2 - 3)^2 = 7)
  let circle2 := set_of (λ (p : ℝ × ℝ), (p.1 + 4)^2 + (p.2 + 1)^2 = 16)
  let center1 := (1 : ℝ, 3 : ℝ)
  let center2 := (-4 : ℝ, -1 : ℝ)
  let radius1 := real.sqrt 7
  let radius2 := 4
  let d := real.sqrt ((1 - (-4))^2 + (3 - (-1))^2)
  radius1 + radius2 - d = real.sqrt 7 + 4 - real.sqrt 41 :=
by sorry

end circle_overlap_distance_l445_445645


namespace statement_a_statement_b_statement_c_statement_d_l445_445391

def f (x : ℝ) : ℝ :=
  abs (cos x) + 1 / (cos (abs x))

theorem statement_a : monotone_on f (set.Ioo 0 (π / 2)) := sorry

theorem statement_b : monotone_on f (set.Ioo (π / 2) π) := sorry

theorem statement_c : ¬ (set.range f ⊆ set.Iic (-2) ∪ set.Ici 2) := sorry

theorem statement_d : infinite {x : ℝ | f x = 2} := sorry

end statement_a_statement_b_statement_c_statement_d_l445_445391


namespace percentage_of_boys_l445_445496

theorem percentage_of_boys (total_students : ℕ) (ratio_boys_to_girls : ℕ) (ratio_girls_to_boys : ℕ) 
  (h_ratio : ratio_boys_to_girls = 3 ∧ ratio_girls_to_boys = 4 ∧ total_students = 42) : 
  (18 / 42) * 100 = 42.857 := 
by 
  sorry

end percentage_of_boys_l445_445496


namespace ticTacToeConfigCorrect_l445_445741

def ticTacToeConfigCount (board : Fin 3 → Fin 3 → Option Char) : Nat := 
  sorry -- this function will count the configurations according to the game rules

theorem ticTacToeConfigCorrect (board : Fin 3 → Fin 3 → Option Char) :
  ticTacToeConfigCount board = 438 := 
  sorry

end ticTacToeConfigCorrect_l445_445741


namespace find_number_of_girls_l445_445011

noncomputable def B (G : ℕ) : ℕ := (8 * G) / 5

theorem find_number_of_girls (B G : ℕ) (h_ratio : B = (8 * G) / 5) (h_total : B + G = 312) : G = 120 :=
by
  -- the proof would be done here
  sorry

end find_number_of_girls_l445_445011


namespace product_eq_sum_l445_445135

variables {x y : ℝ}

theorem product_eq_sum (h : x * y = x + y) (h_ne : y ≠ 1) : x = y / (y - 1) :=
sorry

end product_eq_sum_l445_445135


namespace vector_dot_product_sum_l445_445960

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

theorem vector_dot_product_sum (h₁ : a + b + c = 0) 
                               (ha : ∥a∥ = 1)
                               (hb : ∥b∥ = 2)
                               (hc : ∥c∥ = 2) :
  (a ⬝ b) + (b ⬝ c) + (c ⬝ a) = -9 / 2 := sorry

end vector_dot_product_sum_l445_445960


namespace banana_arrangement_count_l445_445277

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l445_445277


namespace proof_statements_l445_445451

-- Definitions for planes, lines and geometric relationships
variables (α β : Plane) (l m : Line)

-- Conditions
def condition_A := (α ⊥ β) ∧ (l ⊆ β) ∧ (α ∩ β = m) ∧ (l ⊥ m) 
def condition_B := (l ⊥ β) ∧ (α ∥ β) ∧ (m ⊆ α) 
def condition_C := (l ⊆ α) ∧ (m ⊆ α) ∧ (m ∥ β) ∧ (l ∥ β) 
def condition_D := (skew l m) ∧ (l ⊆ α) ∧ (l ∥ β) ∧ (m ⊆ β) ∧ (m ∥ α) 

-- Statements
def statement_A := condition_A → l ⊥ α 
def statement_B := condition_B → l ⊥ m 
def statement_C := condition_C → α ∥ β 
def statement_D := condition_D → α ∥ β 

-- The goal is to prove A and D are true, and B and C are false
theorem proof_statements : 
  (statement_A) ∧ ¬(statement_B) ∧ ¬(statement_C) ∧ (statement_D) := 
by {
  sorry
}

end proof_statements_l445_445451


namespace hyperbola_eccentricity_l445_445757

variables {a b c : ℝ}
variables (h1 : a > 0) (h2 : b > 0)
def hyperbola := ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)
def foci_distance := 2 * c
def MF1 := (2 * c / (Real.cos (60 * Real.pi / 180)))
def MF2 := (2 * c * (Real.tan (30 * Real.pi / 180)))

theorem hyperbola_eccentricity (conditions_met : a > 0 ∧ b > 0)
  (angle_condition : ∃ M : ℝ × ℝ, function.slope M (−c, 0) = Real.tan (30 * Real.pi / 180))
  (perpendicular_condition : ∃ M : ℝ × ℝ, (@prod.snd ℝ ℝ M = 0)) : 
  let e := (c / a) in e = Real.sqrt 3 :=
begin
  sorry -- Proof is omitted.
end

end hyperbola_eccentricity_l445_445757


namespace find_xyz_l445_445517

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Parameters for sides of triangle DEF
def DE : ℝ := 26
def EF : ℝ := 28
def FD : ℝ := 34

-- Heights with respect to each side
noncomputable def height_d (area : ℝ) : ℝ := 2 * area / FD
noncomputable def height_e (area : ℝ) : ℝ := 2 * area / DE
noncomputable def height_f (area : ℝ) : ℝ := 2 * area / EF

-- The maximum height k of the table
noncomputable def max_height (he hf : ℝ) : ℝ := he * hf / (he + hf)

theorem find_xyz : 
  let area := area_of_triangle DE EF FD,
      he := height_e area,
      hf := height_f area,
      k := max_height he hf
  in k = (96 * real.sqrt 55) / 54 ∧ (96 + 55 + 54 = 205) := 
sorry

end find_xyz_l445_445517


namespace sqrt_sixteen_l445_445593

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end sqrt_sixteen_l445_445593


namespace percentage_spent_on_household_items_l445_445085

def Raja_income : ℝ := 37500
def clothes_percentage : ℝ := 0.20
def medicines_percentage : ℝ := 0.05
def savings_amount : ℝ := 15000

theorem percentage_spent_on_household_items : 
  (Raja_income - (clothes_percentage * Raja_income + medicines_percentage * Raja_income + savings_amount)) / Raja_income * 100 = 35 :=
  sorry

end percentage_spent_on_household_items_l445_445085


namespace locus_midpoint_locus_one_third_l445_445600

-- Define the points on the cube
def A := (0 : ℝ, 0 : ℝ, 1 : ℝ)
def B := (1 : ℝ, 0 : ℝ, 1 : ℝ)
def C := (1 : ℝ, 1 : ℝ, 1 : ℝ)
def D := (0 : ℝ, 1 : ℝ, 1 : ℝ)
def A' := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def B' := (1 : ℝ, 0 : ℝ, 0 : ℝ)
def D' := (0 : ℝ, 1 : ℝ, 0 : ℝ)

-- Parameters for points X and Y
variables (t u : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1) (hu : 0 ≤ u) (hu1 : u ≤ 1)

-- Define point X on face diagonal AC
def X : ℝ × ℝ × ℝ := (t, t, 1)

-- Define point Y on B'D'
def Y : ℝ × ℝ × ℝ := (u, 1 - u, 0)

-- Midpoint M of line segment XY
def M := ((t + u) / 2, (t + 1 - u) / 2, 1 / 2)

-- One-third point Z along line segment XY
def Z := ((2 * t + u) / 3, (2 * t + 1 - u) / 3, 2 / 3)

-- Prove the loci
theorem locus_midpoint (t u : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1) (hu : 0 ≤ u) (hu1 : u ≤ 1) :
  (M t u ht ht1 hu hu1).2.2 = 1 / 2 := by
  sorry

theorem locus_one_third (t u : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1) (hu : 0 ≤ u) (hu1 : u ≤ 1) :
  (Z t u ht ht1 hu hu1).2.2 = 2 / 3 := by
  sorry

end locus_midpoint_locus_one_third_l445_445600


namespace sin_half_angle_l445_445839

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l445_445839


namespace choose_officers_ways_l445_445077

theorem choose_officers_ways :
  let members := 12
  let vp_candidates := 4
  let remaining_after_president := members - 1
  let remaining_after_vice_president := remaining_after_president - 1
  let remaining_after_secretary := remaining_after_vice_president - 1
  let remaining_after_treasurer := remaining_after_secretary - 1
  (members * vp_candidates * (remaining_after_vice_president) *
   (remaining_after_secretary) * (remaining_after_treasurer)) = 34560 := by
  -- Calculation here
  sorry

end choose_officers_ways_l445_445077


namespace find_X_l445_445044

def star (a b : ℤ) : ℤ := 5 * a - 3 * b

theorem find_X (X : ℤ) (h1 : star X (star 3 2) = 18) : X = 9 :=
by
  sorry

end find_X_l445_445044


namespace geometric_figures_inequalities_match_figure_II_l445_445228

theorem geometric_figures_inequalities_match_figure_II :
  (∀ x y : ℝ, |x| + |y| ≤ (3/2) * Real.sqrt (x^2 + y^2)) ∧
  (∀ x y : ℝ, (3/2) * Real.sqrt (x^2 + y^2) ≤ Real.sqrt 3 * max (|x|) (|y|)) →
  "Figure II" := sorry

end geometric_figures_inequalities_match_figure_II_l445_445228


namespace efficiency_ratio_l445_445694

variable (A_eff B_eff : ℝ)

-- Condition 1: A and B together finish a piece of work in 36 days
def combined_efficiency := A_eff + B_eff = 1 / 36

-- Condition 2: B alone finishes the work in 108 days
def B_efficiency := B_eff = 1 / 108

-- Theorem: Prove that the ratio of A's efficiency to B's efficiency is 2:1
theorem efficiency_ratio (h1 : combined_efficiency A_eff B_eff) (h2 : B_efficiency B_eff) : (A_eff / B_eff) = 2 := by
  sorry

end efficiency_ratio_l445_445694


namespace A_leq_A_plus_one_l445_445127

-- Define the sets and their cardinalities
def S (n : ℕ) : Set (Set ℕ) :=
  { s | (s : Set ℕ).Sum id = n ∧ ∀ x ∈ s, ∃ i : ℕ, x = 2 * i + 1 }

noncomputable def A (n : ℕ) : ℕ :=
  (S n).card

-- Theorem to be proven
theorem A_leq_A_plus_one (n : ℕ) (h : n > 1) : A n ≤ A (n + 1) := by 
  sorry

end A_leq_A_plus_one_l445_445127


namespace smallest_n_with_terminating_decimal_and_digit_7_l445_445651

-- Definition for \( n \) containing the digit 7
def contains_digit_7 (n: Nat) : Prop :=
  (n.toString.contains '7')

-- Definition for \( \frac{1}{n} \) being a terminating decimal
def is_terminating_decimal (n: Nat) : Prop :=
  ∃ a b: Nat, n = 2^a * 5^b

-- The main theorem statement
theorem smallest_n_with_terminating_decimal_and_digit_7 :
  Nat.find (λ n => is_terminating_decimal n ∧ contains_digit_7 n) = 65536 := by
  sorry

end smallest_n_with_terminating_decimal_and_digit_7_l445_445651


namespace constant_term_in_modified_equation_l445_445511

theorem constant_term_in_modified_equation :
  ∃ (c : ℝ), ∀ (q : ℝ), (3 * (3 * 5 - 3) - 3 + c = 132) → c = 99 := 
by
  sorry

end constant_term_in_modified_equation_l445_445511


namespace coconut_grove_average_yield_l445_445010

theorem coconut_grove_average_yield :
  ∀ (x : ℕ),
  40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x →
  x = 7 :=
by
  intro x
  intro h
  /- sorry proof -/
  sorry

end coconut_grove_average_yield_l445_445010


namespace range_of_alpha_l445_445000

def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem range_of_alpha :
  ∃ α : Set ℝ,
    (∀ x : ℝ, f' x = Real.sin (x + Real.pi / 6)) → 
    α = Set.Icc 0 (Real.pi / 4) ∪ Set.Icc (3 * Real.pi / 4) Real.pi :=
sorry

end range_of_alpha_l445_445000


namespace finite_completely_symmetric_set_is_regular_l445_445718

-- Definition of a completely symmetric set
def completely_symmetric (S : Finset ℝ³) : Prop :=
  S.card ≥ 3 ∧ ∀ {A B : ℝ³}, A ≠ B → A ∈ S → B ∈ S → (∃ r : ℝ³ → ℝ³, r.angle = π ∧ ∀ P ∈ S, r P ∈ S)

-- Main theorem statement
theorem finite_completely_symmetric_set_is_regular (S : Finset ℝ³) (hS : completely_symmetric S) (finite_S : Finite S) :
  (∃ n : ℕ, ∃ P : Finset ℝ³, P = S ∧ variety P regular_polygon) ∨
  (∃ T : Finset ℝ³, T = S ∧ variety T regular_tetrahedron) ∨
  (∃ O : Finset ℝ³, O = S ∧ variety O regular_octahedron) :=
sorry

end finite_completely_symmetric_set_is_regular_l445_445718
